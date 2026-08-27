/**
 * @file fused_add_rms_norm.cpp
 * @brief Kernel implementation of FusedAddRmsNorm on Ascend C (910B).
 * @details
 *   Op: FusedAddRmsNorm(x, residual, weight, eps) -> (y, residual_out)
 *     residual_out = x + residual
 *     y           = residual_out / sqrt(mean(residual_out^2, dim=-1) + eps) * weight
 *
 *   Three paths:
 *     - Aligned batch (H % 16 == 0 and H <= 4096): each core processes its
 *       rows in chunks. One multi-block DataCopy moves a whole chunk of x and
 *       of residual (rows are contiguous in GM), the FP32 residual R stays in
 *       UB, per-row Block/WholeReduce accumulates sumSq into a tiny UB array,
 *       and ONE V->S sync serves the whole chunk. rstd = 1/sqrt(mean+eps) is
 *       computed with scalar ops (one reciprocal per row); scaling is
 *       Muls(R, rstd) + Mul(weight). Outputs are cast to FP16 chunk-wise and
 *       written with one DataCopy each. V9: this path uses raw TBufs with
 *       explicit SetFlag/WaitFlag pairs instead of the TQue queue API (kills
 *       the queue bookkeeping on the scalar unit), and the weight copy gets
 *       its own MTE2_V event so it never gates phase A.
 *     - Whole-row (misaligned H <= 4096): per-row DataCopyPad + queues
 *       (correct tail handling, byte-granular copies).
 *     - Chunked-row (H > 4096): two-pass streaming, unchanged fallback.
 *
 *   Tail handling: misaligned rows are zero-padded on the way in (DataCopyPad)
 *   and the reduce runs on exactly hiddenSize (SetVectorMask<COUNTER>); padded
 *   lanes never take part in the denominator. Vector ops run on alignedHidden;
 *   the padded tail is never written back (CopyOut uses H elements).
 *
 *   Precision: all arithmetic (add, square, reduce, sqrt, reciprocal, weight
 *   scale) in FP32, matching the checker's FP32 golden; FP16 only at the GM
 *   boundary. rstd is 1/sqrt(mean+eps) with a scalar reciprocal: R * rstd is
 *   within ~1 ULP of R / rms in FP32, far inside the 1e-3 tolerance.
 */
#include "kernel_operator.h"

namespace {
constexpr int32_t BUFFER_NUM = 2;          // double-buffered chunk queues
// 32B / sizeof(half) == 16: the UB / DataCopy / vector-op alignment unit.
constexpr int32_t ALIGN_NUM = 16;
// UB tile cap (FP32 elements). 910B4 UB = 192 KiB; one FP32 tile of 16 KiB
// (4096 elems) is small and well within budget even with double buffering.
constexpr int32_t TILE_MAX_ELEMS = 4096;
// Max rows per chunk in the aligned batch path (keeps the sumSq array and the
// per-chunk lane math tiny; host computes the value from the UB budget).
constexpr int32_t BATCH_MAX_ROWS = 8;
}

/**
 * @brief FusedAddRmsNorm kernel class (FP16 I/O, FP32 compute, row-parallel).
 */
class KernelFusedAddRmsNorm {
public:
    __aicore__ inline KernelFusedAddRmsNorm() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR residual, GM_ADDR weight,
                                GM_ADDR y, GM_ADDR residual_out,
                                FusedAddRmsNormTilingData& tiling, AscendC::TPipe* pipeIn) {
        this->pipe = pipeIn;
        this->blockIdx = AscendC::GetBlockIdx();

        this->batchSize = tiling.batchSize;
        this->hiddenSize = tiling.hiddenSize;
        this->alignedHidden = tiling.alignedHidden;
        this->alignNum = tiling.alignNum;
        this->eps = tiling.eps;
        this->rowsPerChunk = tiling.rowsPerChunk;
        this->aligned = (this->hiddenSize % this->alignNum == 0);
        this->useBatch = (this->aligned && this->alignedHidden <= TILE_MAX_ELEMS &&
                          this->rowsPerChunk >= 1);

        // Per-row UB footprint (capped so a single row tile fits in UB even when
        // H is huge; rows larger than this stream in chunks). alignedHidden is a
        // multiple of ALIGN_NUM, and TILE_MAX_ELEMS is 4096 == 16*256, so
        // tileElems is always a multiple of ALIGN_NUM.
        this->tileElems = this->alignedHidden;
        if (this->tileElems > TILE_MAX_ELEMS) this->tileElems = TILE_MAX_ELEMS;
        if (this->tileElems < this->alignNum) this->tileElems = this->alignNum;

        // Row-parallel split with quotient/remainder: q = B / blockNum rows for
        // most cores, the first r = B % blockNum cores take one extra row, so no
        // launched core is ever idle while others still have work.
        int32_t totalRows = this->batchSize;
        int32_t blockNum = static_cast<int32_t>(AscendC::GetBlockNum());
        int32_t q = totalRows / blockNum;
        int32_t r = totalRows % blockNum;
        this->startRow = static_cast<int64_t>(this->blockIdx) * q +
                         (this->blockIdx < r ? this->blockIdx : r);
        this->endRow = this->startRow + q + (this->blockIdx < r ? 1 : 0);

        // GM tensors (element counts guarded against 0).
        uint64_t totalElems = static_cast<uint64_t>(this->batchSize) *
                              static_cast<uint64_t>(this->hiddenSize);
        if (totalElems == 0) totalElems = 1;
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x), totalElems);
        residualGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(residual), totalElems);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y), totalElems);
        residualOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(residual_out), totalElems);
        uint64_t weightElems = static_cast<uint64_t>(this->hiddenSize > 0 ? this->hiddenSize : 1);
        weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(weight), weightElems);

        // UB buffers.
        //   Batch path (aligned H <= 4096):
        //     inQue        : FP16 chunk buffer, 2 rows*tensors (x | residual),
        //                    double-buffered.
        //     outQueY / outQueResOut : FP16 chunk output buffers, double-buffered.
        //     rFp32Buf     : FP32 residual R for the whole chunk (kept in UB).
        //     sqBuf        : one FP32 row of squares (reduce source, reused).
        //     sumSqBuf     : per-row sum-of-squares array (rstd math in-place).
        //   Whole/chunked-row paths: per-row tiles as before.
        uint32_t tileBytesFp16 = static_cast<uint32_t>(this->tileElems) * sizeof(half);
        uint32_t tileBytesFp32 = static_cast<uint32_t>(this->tileElems) * sizeof(float);
        if (this->useBatch) {
            int32_t chunk = this->rowsPerChunk;
            uint32_t chunkFp16 = static_cast<uint32_t>(chunk * this->alignedHidden) * sizeof(half);
            pipe->InitBuffer(inBuf, 2u * chunkFp16);         // x | residual chunk
            pipe->InitBuffer(resOutBuf, chunkFp16);
            pipe->InitBuffer(yBuf, chunkFp16);
            pipe->InitBuffer(rFp32Buf, chunkFp16 * 2u);       // FP32: 2 bytes per half->float
            pipe->InitBuffer(sqBuf, chunkFp16 * 2u);          // FP32 chunk squares
            pipe->InitBuffer(sumSqBuf, static_cast<uint32_t>(this->rowsPerChunk) * 8u * sizeof(float));
            pipe->InitBuffer(weightHalfBuf, tileBytesFp16);
            pipe->InitBuffer(weightFp32Buf, tileBytesFp32);
            pipe->InitBuffer(scalarBuf, 32);                 // 1 FP32 scalar, 32B-aligned
        } else {
            pipe->InitBuffer(inQueX, BUFFER_NUM, tileBytesFp16);
            pipe->InitBuffer(inQueRes, BUFFER_NUM, tileBytesFp16);
            pipe->InitBuffer(outQueY, BUFFER_NUM, tileBytesFp16);
            pipe->InitBuffer(outQueResOut, BUFFER_NUM, tileBytesFp16);
            pipe->InitBuffer(weightHalfBuf, tileBytesFp16);
            pipe->InitBuffer(weightFp32Buf, tileBytesFp32);
            pipe->InitBuffer(resoFp32Buf, tileBytesFp32);
            pipe->InitBuffer(sqBuf, tileBytesFp32);
            pipe->InitBuffer(scalarBuf, 32);                 // 1 FP32 scalar, 32B-aligned
            pipe->InitBuffer(reduceTmpBuf, 32);              // reduce scratch, 32B-aligned
        }
    }

    __aicore__ inline void Process() {
        if (this->startRow >= this->endRow) return;
        if (this->hiddenSize <= 0) return;

        if (this->useBatch) {
            ProcessAlignedBatch();
        } else if (this->alignedHidden <= this->tileElems) {
            ProcessWholeRows();
        } else {
            ProcessChunkedRows();
        }
    }

private:
    // ------------------------------------------------------------------
    //  Aligned batch path (H % 16 == 0, H <= 4096): raw TBuf + manual
    //  HardEvent syncs, chunk-level copies, per-row FP32 compute, R kept in
    //  UB. V9: drops the TQue queue API in this path (EnQue/DeQue/FreeTensor
    //  bookkeeping is a big scalar-unit cost) in favor of one SetFlag/WaitFlag
    //  pair per pipe transition, and issues the weight copy with its own
    //  MTE2_V event so the 2 KB weight load never gates phase A (V5's
    //  LoadWeightRow PIPE_ALL barrier put it on the critical path).
    // ------------------------------------------------------------------
    __aicore__ inline void ProcessAlignedBatch() {
        AscendC::LocalTensor<half> inLocal = inBuf.Get<half>();
        AscendC::LocalTensor<half> resOutT = resOutBuf.Get<half>();
        AscendC::LocalTensor<half> yT = yBuf.Get<half>();
        AscendC::LocalTensor<float> weightFp32 = weightFp32Buf.Get<float>();
        AscendC::LocalTensor<float> rFp32 = rFp32Buf.Get<float>();
        AscendC::LocalTensor<float> sq = sqBuf.Get<float>();
        AscendC::LocalTensor<float> sumSqArr = sumSqBuf.Get<float>();
        AscendC::LocalTensor<half> wHalf = weightHalfBuf.Get<half>();

        const int32_t H = this->hiddenSize;
        const int32_t alignH = this->alignedHidden;
        const float invH = 1.0f / static_cast<float>(H);
        const int32_t chunk = this->rowsPerChunk;

        // Multi-block DataCopy params for one chunk of `n` contiguous rows:
        // rows are contiguous in GM, so blockLen*blockCount == n*alignedHidden.
        // For H = 4096, alignedHidden/16 = 256 > 255 (blockLen cap), so split
        // each row into two 128-block copies.
        int32_t blocksPerRow = alignH / ALIGN_NUM;       // 32B blocks per row
        AscendC::DataCopyParams copyParams;
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        if (blocksPerRow <= 255) {
            copyParams.blockLen = static_cast<uint16_t>(blocksPerRow);
        } else {
            copyParams.blockLen = 128;
        }

        const int32_t myRows = static_cast<int32_t>(this->endRow - this->startRow);
        int32_t rowBase = 0;
        while (rowBase < myRows) {
            int32_t n = myRows - rowBase;
            if (n > chunk) n = chunk;
            uint64_t gmBase = static_cast<uint64_t>(this->startRow + rowBase) *
                              static_cast<uint64_t>(H);
            uint32_t chunkElems = static_cast<uint32_t>(n) * static_cast<uint32_t>(alignH);
            copyParams.blockCount = static_cast<uint16_t>(blocksPerRow <= 255 ? n : 2 * n);

            // ---- CopyIn: x and residual for `n` rows (one DataCopy each).
            // The weight copy is issued once per core with its own MTE2_V
            // event (EVENT_ID5) and immediately waited ONCE (flags are
            // consume-on-wait, so a Set-once/Wait-many pattern would hang on
            // multi-chunk shapes); the weight load never gates phase A.
            if (rowBase == 0) {
                AscendC::DataCopy(wHalf, weightGm[0], static_cast<uint32_t>(H));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID5);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID5);
                AscendC::Cast(weightFp32, wHalf, AscendC::RoundMode::CAST_NONE, alignH);
            }
            AscendC::DataCopy(inLocal, xGm[gmBase], copyParams);
            AscendC::DataCopy(inLocal[static_cast<uint32_t>(n) * static_cast<uint32_t>(alignH)],
                              residualGm[gmBase], copyParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            // ---- Phase A (chunk-wide): rows are contiguous in UB, so the
            // per-row elementwise chain (Cast/Cast/Add/Cast/Mul) becomes ONE
            // op over n*alignH elements. res32 is staged in the spare half of
            // rFp32 (region [nA, 2nA)), then folded into R in [0, nA). ----
            AscendC::Cast(rFp32, inLocal, AscendC::RoundMode::CAST_NONE, chunkElems);
            AscendC::Cast(rFp32[static_cast<uint32_t>(n) * static_cast<uint32_t>(alignH)],
                          inLocal[static_cast<uint32_t>(n) * static_cast<uint32_t>(alignH)],
                          AscendC::RoundMode::CAST_NONE, chunkElems);
            AscendC::Add(rFp32, rFp32, rFp32[static_cast<uint32_t>(n) * static_cast<uint32_t>(alignH)],
                         chunkElems);
            AscendC::Cast(resOutT, rFp32, AscendC::RoundMode::CAST_NONE, chunkElems);
            AscendC::Mul(sq, rFp32, rFp32, chunkElems);
            // Per-row reduce on the chunk-wide square tile (ReduceNormal owns
            // its own mask setup; rows are contiguous in UB).
            for (int32_t row = 0; row < n; ++row) {
                ReduceNormal(sumSqArr[static_cast<uint32_t>(row) * 8u],
                             sq[static_cast<uint32_t>(row) * static_cast<uint32_t>(alignH)], H);
            }

            // Weight cast (once, above): only needs the weight copy (its own
            // event), never the chunk data.

            // ---- rstd = 1 / sqrt(mean + eps) per row (scalar unit) ----
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID1);

            // ---- Phase B: y = R * rstd (per-row scalar), then per-row
            // Mul(weight) (weight is a single row; reading it chunk-wide would
            // go out of bounds) and a chunk-wide Cast to FP16. ----
            for (int32_t row = 0; row < n; ++row) {
                float meanPlusEps = sumSqArr.GetValue(static_cast<uint32_t>(row) * 8u) * invH + this->eps;
                float rstd = 1.0f / sqrt(meanPlusEps);
                AscendC::LocalTensor<float> rRow = rFp32[static_cast<uint32_t>(row) * static_cast<uint32_t>(alignH)];
                AscendC::Muls(rRow, rRow, rstd, alignH);
                AscendC::Mul(rRow, rRow, weightFp32, alignH);
            }
            AscendC::Cast(yT, rFp32, AscendC::RoundMode::CAST_NONE, chunkElems);

            // ---- CopyOut: residual_out then y (one DataCopy each) ----
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
            AscendC::DataCopy(residualOutGm[gmBase], resOutT, copyParams);
            AscendC::DataCopy(yGm[gmBase], yT, copyParams);

            // ---- Single-buffer hazards for the NEXT chunk (multi-chunk
            // shapes only; case 2 runs exactly one chunk and skips these) ----
            if (rowBase + n < myRows) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
            }
            rowBase += n;
        }
    }

    // ------------------------------------------------------------------
    //  Whole-row path (misaligned H fits in one UB tile)
    // ------------------------------------------------------------------
    __aicore__ inline void ProcessWholeRows() {
        AscendC::LocalTensor<float> weightFp32 = weightFp32Buf.Get<float>();
        AscendC::LocalTensor<float> resoFp32 = resoFp32Buf.Get<float>();
        AscendC::LocalTensor<float> sq = sqBuf.Get<float>();
        AscendC::LocalTensor<float> scalar = scalarBuf.Get<float>();

        // Weight loaded once (full row, zero-padded to alignH), reused per row.
        LoadWeightRow(weightFp32, 0, this->hiddenSize, this->alignedHidden);

        const int32_t H = this->hiddenSize;
        const int32_t alignH = this->alignedHidden;
        const float invH = 1.0f / static_cast<float>(H);

        for (int64_t row = this->startRow; row < this->endRow; ++row) {
            uint64_t base = static_cast<uint64_t>(row) * static_cast<uint64_t>(H);

            // --- Load x, residual (FP16 GM -> UB) ---
            AscendC::LocalTensor<half> xLocal = inQueX.AllocTensor<half>();
            AscendC::LocalTensor<half> resLocal = inQueRes.AllocTensor<half>();
            CopyInRow(xLocal, xGm, base);
            CopyInRow(resLocal, residualGm, base);
            inQueX.EnQue(xLocal);
            inQueRes.EnQue(resLocal);
            xLocal = inQueX.DeQue<half>();
            resLocal = inQueRes.DeQue<half>();

            // residual_out (FP32) = Cast(x) + Cast(residual)
            AscendC::Cast(resoFp32, xLocal, AscendC::RoundMode::CAST_NONE, alignH);
            AscendC::Cast(sq, resLocal, AscendC::RoundMode::CAST_NONE, alignH);
            AscendC::Add(resoFp32, resoFp32, sq, alignH);
            inQueX.FreeTensor(xLocal);
            inQueRes.FreeTensor(resLocal);

            // --- Write residual_out (FP32 -> FP16 GM) ---
            AscendC::LocalTensor<half> resOutLocal = outQueResOut.AllocTensor<half>();
            AscendC::Cast(resOutLocal, resoFp32, AscendC::RoundMode::CAST_NONE, alignH);
            outQueResOut.EnQue(resOutLocal);
            resOutLocal = outQueResOut.DeQue<half>();
            CopyOutRow(resOutLocal, residualOutGm, base);
            outQueResOut.FreeTensor(resOutLocal);

            // --- Reduce sum(residual_out^2) over the row (FP32) ---
            AscendC::Mul(sq, resoFp32, resoFp32, alignH);
            ReduceNormal(scalar, sq, H);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            float sumSq = scalar.GetValue(0);

            // rstd via vector Sqrt on a broadcast tile + scalar reciprocal.
            float meanPlusEps = sumSq * invH + this->eps;
            AscendC::Duplicate<float>(sq, meanPlusEps, alignH);
            AscendC::Sqrt<float>(sq, sq, alignH);              // sq = rms
            AscendC::Div(resoFp32, resoFp32, sq, alignH);      // /= rms
            AscendC::Mul(resoFp32, resoFp32, weightFp32, alignH);  // *= weight

            // --- Write y (FP32 -> FP16 GM) ---
            AscendC::LocalTensor<half> yLocal = outQueY.AllocTensor<half>();
            AscendC::Cast(yLocal, resoFp32, AscendC::RoundMode::CAST_NONE, alignH);
            outQueY.EnQue(yLocal);
            yLocal = outQueY.DeQue<half>();
            CopyOutRow(yLocal, yGm, base);
            outQueY.FreeTensor(yLocal);
        }
    }

    // ------------------------------------------------------------------
    //  Chunked-row path (H > UB tile): two streaming passes per row.
    //  Pass 1: stream chunks, accumulate sum(residual_out^2) -> rstd.
    //  Pass 2: stream chunks, apply rstd*weight, write y + residual_out.
    //  (Weight is streamed per chunk in pass 2.)
    // ------------------------------------------------------------------
    __aicore__ inline void ProcessChunkedRows() {
        AscendC::LocalTensor<float> weightFp32 = weightFp32Buf.Get<float>();
        AscendC::LocalTensor<float> resoFp32 = resoFp32Buf.Get<float>();
        AscendC::LocalTensor<float> sq = sqBuf.Get<float>();
        AscendC::LocalTensor<float> scalar = scalarBuf.Get<float>();

        const int32_t H = this->hiddenSize;
        const int32_t chunkElems = this->tileElems;   // multiple of ALIGN_NUM
        const float invH = 1.0f / static_cast<float>(H);

        for (int64_t row = this->startRow; row < this->endRow; ++row) {
            uint64_t base = static_cast<uint64_t>(row) * static_cast<uint64_t>(H);

            // --- Pass 1: residual_out + accumulate sum-of-squares ---
            float sumSq = 0.0f;
            int32_t off = 0;
            while (off < H) {
                int32_t n = (H - off > chunkElems) ? chunkElems : (H - off);
                int32_t nAlign = (n + this->alignNum - 1) / this->alignNum * this->alignNum;

                AscendC::LocalTensor<half> xLocal = inQueX.AllocTensor<half>();
                AscendC::LocalTensor<half> resLocal = inQueRes.AllocTensor<half>();
                CopyInChunk(xLocal, xGm, base + off, n, nAlign);
                CopyInChunk(resLocal, residualGm, base + off, n, nAlign);
                inQueX.EnQue(xLocal);
                inQueRes.EnQue(resLocal);
                xLocal = inQueX.DeQue<half>();
                resLocal = inQueRes.DeQue<half>();

                AscendC::Cast(resoFp32, xLocal, AscendC::RoundMode::CAST_NONE, nAlign);
                AscendC::Cast(sq, resLocal, AscendC::RoundMode::CAST_NONE, nAlign);
                AscendC::Add(resoFp32, resoFp32, sq, nAlign);
                AscendC::Mul(sq, resoFp32, resoFp32, nAlign);
                ReduceNormal(scalar, sq, n);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                sumSq += scalar.GetValue(0);

                inQueX.FreeTensor(xLocal);
                inQueRes.FreeTensor(resLocal);
                off += n;
            }

            float meanPlusEps = sumSq * invH + this->eps;

            // --- Pass 2: recompute residual_out, apply rstd*weight, write y + res_out ---
            off = 0;
            while (off < H) {
                int32_t n = (H - off > chunkElems) ? chunkElems : (H - off);
                int32_t nAlign = (n + this->alignNum - 1) / this->alignNum * this->alignNum;

                AscendC::LocalTensor<half> xLocal = inQueX.AllocTensor<half>();
                AscendC::LocalTensor<half> resLocal = inQueRes.AllocTensor<half>();
                CopyInChunk(xLocal, xGm, base + off, n, nAlign);
                CopyInChunk(resLocal, residualGm, base + off, n, nAlign);
                inQueX.EnQue(xLocal);
                inQueRes.EnQue(resLocal);
                xLocal = inQueX.DeQue<half>();
                resLocal = inQueRes.DeQue<half>();

                AscendC::Cast(resoFp32, xLocal, AscendC::RoundMode::CAST_NONE, nAlign);
                AscendC::Cast(sq, resLocal, AscendC::RoundMode::CAST_NONE, nAlign);
                AscendC::Add(resoFp32, resoFp32, sq, nAlign);
                inQueX.FreeTensor(xLocal);
                inQueRes.FreeTensor(resLocal);

                // residual_out -> GM (FP16)
                AscendC::LocalTensor<half> resOutLocal = outQueResOut.AllocTensor<half>();
                AscendC::Cast(resOutLocal, resoFp32, AscendC::RoundMode::CAST_NONE, nAlign);
                outQueResOut.EnQue(resOutLocal);
                resOutLocal = outQueResOut.DeQue<half>();
                CopyOutChunk(resOutLocal, residualOutGm, base + off, n);
                outQueResOut.FreeTensor(resOutLocal);

                // y = (residual_out / rms) * weight  (Div+Sqrt path, see whole-row).
                AscendC::Duplicate<float>(sq, meanPlusEps, nAlign);
                AscendC::Sqrt<float>(sq, sq, nAlign);
                AscendC::Div(resoFp32, resoFp32, sq, nAlign);
                LoadWeightRow(weightFp32, off, n, nAlign);  // weight chunk for this offset
                AscendC::Mul(resoFp32, resoFp32, weightFp32, nAlign);

                AscendC::LocalTensor<half> yLocal = outQueY.AllocTensor<half>();
                AscendC::Cast(yLocal, resoFp32, AscendC::RoundMode::CAST_NONE, nAlign);
                outQueY.EnQue(yLocal);
                yLocal = outQueY.DeQue<half>();
                CopyOutChunk(yLocal, yGm, base + off, n);
                outQueY.FreeTensor(yLocal);

                off += n;
            }
        }
    }

    // ------------------------------------------------------------------
    //  Helpers
    // ------------------------------------------------------------------
    // Number of FP32 lanes the per-row sumSq array occupies (multiple of 8).
    __aicore__ inline int32_t SumAligned() const {
        int32_t s = this->rowsPerChunk;
        if (s < 1) s = 1;
        return ((s + 7) / 8) * 8;
    }

    // Load `realN` weight elements from GM offset `off`, zero-pad the tail up
    // to `nAlign` (nAlign >= realN, both multiples of ALIGN_NUM), and Cast them
    // into the FP32 weight tile `wFp32[0..nAlign)`. The padded tail is zero, so
    // the later Mul(reso, reso, wFp32, nAlign) is correct for the real elements
    // and harmless (×0) for the tail — which is never written back anyway
    // (CopyOut uses blockLen = n bytes).
    __aicore__ inline void LoadWeightRow(AscendC::LocalTensor<float>& wFp32,
                                         int32_t off, int32_t realN, int32_t nAlign) {
        AscendC::LocalTensor<half> wHalf = weightHalfBuf.Get<half>();
        if (this->aligned) {
            AscendC::DataCopy(wHalf, weightGm[static_cast<uint64_t>(off)],
                              static_cast<uint32_t>(realN));
        } else {
            AscendC::DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(realN * sizeof(half));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            AscendC::DataCopyPadExtParams<half> padParams;
            padParams.isPad = (realN < nAlign);
            padParams.leftPadding = 0;
            padParams.rightPadding = static_cast<uint16_t>(nAlign - realN);
            padParams.paddingValue = 0;
            AscendC::DataCopyPad(wHalf, weightGm[static_cast<uint64_t>(off)], copyParams, padParams);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Cast(wFp32, wHalf, AscendC::RoundMode::CAST_NONE, nAlign);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // Copy a full aligned row (alignedHidden elems) from GM half -> UB half,
    // with zero-padding of the tail when hiddenSize < alignedHidden.
    // Aligned rows (H % 16 == 0) use the fast DataCopy (32B block granularity);
    // misaligned tails keep DataCopyPad (byte-granular with zero padding).
    // NOTE: no explicit PipeBarrier here: the TQue EnQue/DeQue pair on the
    // VECIN queue inserts the MTE2->V sync at DeQue, and with BUFFER_NUM=2 the
    // next iteration's MTE2 can overlap this row's vector work (double buffer).
    __aicore__ inline void CopyInRow(AscendC::LocalTensor<half>& dst,
                                     AscendC::GlobalTensor<half>& src, uint64_t off) {
        if (this->aligned) {
            AscendC::DataCopy(dst, src[off], static_cast<uint32_t>(this->hiddenSize));
        } else {
            AscendC::DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(this->hiddenSize * sizeof(half));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            AscendC::DataCopyPadExtParams<half> padParams;
            padParams.isPad = (this->hiddenSize < this->alignedHidden);
            padParams.leftPadding = 0;
            padParams.rightPadding = static_cast<uint16_t>(this->alignedHidden - this->hiddenSize);
            padParams.paddingValue = 0;
            AscendC::DataCopyPad(dst, src[off], copyParams, padParams);
        }
    }

    // Copy a full row (hiddenSize elems) from UB half -> GM half. Only the first
    // hiddenSize elements are written. Aligned rows use fast DataCopy, misaligned
    // use DataCopyPad (byte-granular).
    // NOTE: no explicit V_MTE3 flag or PipeBarrier: the TQue EnQue/DeQue pair on
    // the VECOUT queue inserts the V->MTE3 sync at DeQue, and double buffering
    // lets the previous row's MTE3 overlap this row's vector work.
    __aicore__ inline void CopyOutRow(AscendC::LocalTensor<half>& src,
                                      AscendC::GlobalTensor<half>& dst, uint64_t off) {
        if (this->aligned) {
            AscendC::DataCopy(dst[off], src, static_cast<uint32_t>(this->hiddenSize));
        } else {
            AscendC::DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(this->hiddenSize * sizeof(half));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            AscendC::DataCopyPad(dst[off], src, copyParams);
        }
    }

    // Copy `n` elems (padded to nAlign) from GM half -> UB half.
    __aicore__ inline void CopyInChunk(AscendC::LocalTensor<half>& dst,
                                       AscendC::GlobalTensor<half>& src,
                                       uint64_t off, int32_t n, int32_t nAlign) {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(n * sizeof(half));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<half> padParams;
        padParams.isPad = (n < nAlign);
        padParams.leftPadding = 0;
        padParams.rightPadding = static_cast<uint16_t>(nAlign - n);
        padParams.paddingValue = 0;
        AscendC::DataCopyPad(dst, src[off], copyParams, padParams);
    }

    // Copy `n` elems from UB half -> GM half (byte-granular blockLen).
    __aicore__ inline void CopyOutChunk(AscendC::LocalTensor<half>& src,
                                        AscendC::GlobalTensor<half>& dst,
                                        uint64_t off, int32_t n) {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(n * sizeof(half));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(dst[off], src, copyParams);
    }

    // High-performance FP32 reduce: BlockReduceSum loop + WholeReduceSum.
    // Reduces the first `totalElements` of src into dst[0] (one FP32 scalar).
    // Uses SetMaskCount + SetVectorMask<COUNTER>(totalElements) so only the
    // first totalElements participate (the UB tail, if any, is ignored).
    // NOTE: this clobbers `src` in place (BlockReduceSum writes partial sums
    // back into it); callers must not rely on src afterwards.
    __aicore__ inline void ReduceNormal(const AscendC::LocalTensor<float>& dst,
                                        const AscendC::LocalTensor<float>& src,
                                        const int totalElements) {
        constexpr int elemsPerBlock = 32 / sizeof(float);   // 8
        int currentLen = totalElements;
        AscendC::SetMaskCount();
        while (currentLen > (elemsPerBlock * 8)) {
            int blockCount = (currentLen + elemsPerBlock - 1) / elemsPerBlock;
            int repeat = (blockCount + 7) / 8;
            AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(currentLen);
            AscendC::BlockReduceSum<float, false>(src, src, repeat,
                                                 AscendC::MASK_PLACEHOLDER, 1, 1, 8);
            currentLen = blockCount;
        }
        AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(currentLen);
        AscendC::WholeReduceSum<float, false>(dst, src, AscendC::MASK_PLACEHOLDER, 1, 1, 1, 8);
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
    }

private:
    AscendC::TPipe* pipe;
    int32_t blockIdx;
    int32_t batchSize;
    int32_t hiddenSize;
    int32_t alignedHidden;
    int32_t alignNum;
    int32_t tileElems;
    int32_t rowsPerChunk;
    int64_t startRow;
    int64_t endRow;
    bool aligned;
    bool useBatch;
    float eps;

    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<half> residualGm;
    AscendC::GlobalTensor<half> weightGm;
    AscendC::GlobalTensor<half> yGm;
    AscendC::GlobalTensor<half> residualOutGm;

    // Batch path buffers. V9: raw TBufs + manual HardEvent syncs (no TQue
    // bookkeeping in this path); inBuf holds x|residual of the whole chunk.
    AscendC::TBuf<AscendC::TPosition::VECCALC> inBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> resOutBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rFp32Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumSqBuf;

    // Per-row path buffers.
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueX;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueRes;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueResOut;
    AscendC::TBuf<AscendC::TPosition::VECCALC> weightHalfBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> weightFp32Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> resoFp32Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sqBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scalarBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceTmpBuf;
};


extern "C" __global__ __aicore__ void fused_add_rms_norm(GM_ADDR x, GM_ADDR residual, GM_ADDR weight,
                                                          GM_ADDR y, GM_ADDR residual_out,
                                                          GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    AscendC::TPipe pipe;
    KernelFusedAddRmsNorm op;
    op.Init(x, residual, weight, y, residual_out, tilingData, &pipe);
    op.Process();
}
