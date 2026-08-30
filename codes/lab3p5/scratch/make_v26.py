#!/usr/bin/env python3
# Build kernel_v26.cpp from kernel_v25.cpp:
#  - BlockReduceSum tree (BS1 8192->1024, BS2 1024->128, WRS 128->8 packed) replaces
#    the 8 per-row fold-Adds + WRS (9 vector instructions -> 3).
#  - weight pre-broadcast into wTile via UB->UB DataCopy (srcStride=0) so the
#    per-row weight Mul loop (8 calls) becomes ONE chunk-wide Mul.
import re

s = open("scratch/kernel_v25.cpp", "rb").read().decode()

# ---- 1) buffers: add sq2 (1024 fp32) + sumTmp (128 fp32) + wTile (n*alignH fp32) ----
old = """            pipe->InitBuffer(sumSqBuf, static_cast<uint32_t>(this->rowsPerChunk) * 8u * sizeof(float));
            pipe->InitBuffer(weightHalfBuf, tileBytesFp16);
            pipe->InitBuffer(weightFp32Buf, tileBytesFp32);
            pipe->InitBuffer(scalarBuf, 32);                 // 1 FP32 scalar, 32B-aligned"""
new = """            pipe->InitBuffer(sumSqBuf, static_cast<uint32_t>(this->rowsPerChunk) * 8u * sizeof(float));
            pipe->InitBuffer(sq2Buf, 1024 * sizeof(float));      // BS1 dst (group sums)
            pipe->InitBuffer(sumTmpBuf, 128 * sizeof(float));     // BS2 dst (per-row partials)
            pipe->InitBuffer(weightHalfBuf, tileBytesFp16);
            pipe->InitBuffer(weightFp32Buf, tileBytesFp32);
            pipe->InitBuffer(wTileBuf, static_cast<uint32_t>(chunk * this->alignedHidden) * sizeof(float));
            pipe->InitBuffer(scalarBuf, 32);                 // 1 FP32 scalar, 32B-aligned"""
assert old in s, "init buffers"
s = s.replace(old, new)

# ---- 2) reduce: BlockReduceSum tree replaces fold+WRS ----
old = """            // ---- Per-row fold (rows of both halves are contiguous in sq) ----
            const int32_t foldSegs = H >> 6;
            const int32_t foldTail = H & 63;
            if (foldSegs > 1) {
                for (int32_t row = 0; row < n; ++row) {
                    AscendC::LocalTensor<float> sqRow =
                        sq[static_cast<uint32_t>(row) * static_cast<uint32_t>(alignH)];
                    AscendC::Add<float>(sqRow, sqRow[64], sqRow, 64,
                                        static_cast<uint8_t>(foldSegs - 1),
                                        AscendC::BinaryRepeatParams(1, 1, 1, 0, 8, 0));
                }
            }
            if (foldSegs >= 1 && foldTail > 0) {
                const int32_t tailOff = H & ~63;
                for (int32_t row = 0; row < n; ++row) {
                    AscendC::LocalTensor<float> sqRow =
                        sq[static_cast<uint32_t>(row) * static_cast<uint32_t>(alignH)];
                    AscendC::Add<float>(sqRow, sqRow[tailOff], sqRow, foldTail, 1,
                                        AscendC::BinaryRepeatParams(1, 1, 1, 0, 8, 0));
                }
            }
            AscendC::WholeReduceSum<float>(sumSqArr, sq, foldSegs > 0 ? 64 : H, n, 1, 1,
                                           alignH / 8);"""
new = """            // ---- Per-row reduce via a BlockReduceSum tree (3 vector
            // instructions total instead of 8 fold-Adds + WRS). Works when the
            // chunk is exactly n rows of alignedHidden (n*alignH elements,
            // each row's length a multiple of 64); otherwise falls back to
            // the per-row fold. BS1: each 32B block (8 fp32) -> 1 sum;
            // repeat k writes dst[8k..8k+8). BS2 repeats the same on BS1's
            // output. WRS then sums each row's 16 partials (rows stay
            // block-aligned because alignedHidden is a multiple of 128).
            // Misaligned H never reaches this path.
            if (n * alignH >= 64 && (alignH & 127) == 0 && (n * alignH) % 64 == 0) {
                // V13 ReduceNormal idiom: SetMaskCount + 1-arg COUNTER mask
                // (total elements), repeatTime = total/64 per stage.
                AscendC::SetMaskCount();
                AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(n * alignH);
                AscendC::BlockReduceSum<float, false>(sq2, sq, (n * alignH) / 64,
                                                      AscendC::MASK_PLACEHOLDER, 1, 1, 8);
                AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(n * alignH / 8);
                AscendC::BlockReduceSum<float, false>(sumTmp, sq2, (n * alignH) / 512,
                                                      AscendC::MASK_PLACEHOLDER, 1, 1, 8);
                AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(16);
                AscendC::WholeReduceSum<float, false>(sumSqArr, sumTmp, 16, n, 1, 1, 2);
                AscendC::SetMaskNorm();
                AscendC::ResetMask();
            } else {
                const int32_t foldSegs = H >> 6;
                const int32_t foldTail = H & 63;
                if (foldSegs > 1) {
                    for (int32_t row = 0; row < n; ++row) {
                        AscendC::LocalTensor<float> sqRow =
                            sq[static_cast<uint32_t>(row) * static_cast<uint32_t>(alignH)];
                        AscendC::Add<float>(sqRow, sqRow[64], sqRow, 64,
                                            static_cast<uint8_t>(foldSegs - 1),
                                            AscendC::BinaryRepeatParams(1, 1, 1, 0, 8, 0));
                    }
                }
                if (foldSegs >= 1 && foldTail > 0) {
                    const int32_t tailOff = H & ~63;
                    for (int32_t row = 0; row < n; ++row) {
                        AscendC::LocalTensor<float> sqRow =
                            sq[static_cast<uint32_t>(row) * static_cast<uint32_t>(alignH)];
                        AscendC::Add<float>(sqRow, sqRow[tailOff], sqRow, foldTail, 1,
                                            AscendC::BinaryRepeatParams(1, 1, 1, 0, 8, 0));
                    }
                }
                AscendC::WholeReduceSum<float>(sumSqArr, sq, foldSegs > 0 ? 64 : H, n, 1, 1,
                                               alignH / 8);
            }"""
assert old in s, "fold block"
s = s.replace(old, new)

# ---- 3) phase B: wTile broadcast + chunk-wide weight mul ----
old = """            AscendC::PipeBarrier<PIPE_V>();
            const AscendC::BinaryRepeatParams binaryParams;
            for (int32_t row = 0; row < n; ++row) {
                AscendC::LocalTensor<float> rRow = rFp32[static_cast<uint32_t>(row) * static_cast<uint32_t>(alignH)];
                AscendC::Mul<float, false>(rRow, rRow, weightFp32, AscendC::MASK_PLACEHOLDER, 1, binaryParams);
            }
            AscendC::PipeBarrier<PIPE_V>();"""
new = """            AscendC::PipeBarrier<PIPE_V>();
            // Weight broadcast: wTile rows are identical copies of weightFp32
            // (ONE UB->UB DataCopy with srcStride=0), so the weight multiply
            // becomes a single chunk-wide instruction instead of n per-row
            // Mul calls.
            {
                AscendC::DataCopyParams wtParams;
                wtParams.blockCount = static_cast<uint16_t>(n);
                wtParams.blockLen = static_cast<uint16_t>(alignH / 8);
                wtParams.srcStride = 0;
                wtParams.dstStride = 0;
                AscendC::DataCopy(wTile, weightFp32, wtParams);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID5);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID5);
                AscendC::SetMaskCount();
                AscendC::SetVectorMask<uint8_t, AscendC::MaskMode::COUNTER>(0, chunkElems);
                const AscendC::BinaryRepeatParams binaryParams;
                AscendC::Mul<float, false>(rFp32, rFp32, wTile, AscendC::MASK_PLACEHOLDER, 1,
                                           binaryParams);
                AscendC::SetMaskNorm();
            }
            AscendC::PipeBarrier<PIPE_V>();"""
assert old in s, "phase B weight loop"
s = s.replace(old, new)

# locals: add wTile
old = """        AscendC::LocalTensor<float> weightFp32 = weightFp32Buf.Get<float>();
        AscendC::LocalTensor<half> wHalf = weightHalfBuf.Get<half>();"""
new = """        AscendC::LocalTensor<float> weightFp32 = weightFp32Buf.Get<float>();
        AscendC::LocalTensor<float> wTile = wTileBuf.Get<float>();
        AscendC::LocalTensor<float> sq2 = sq2Buf.Get<float>();
        AscendC::LocalTensor<float> sumTmp = sumTmpBuf.Get<float>();
        AscendC::LocalTensor<half> wHalf = weightHalfBuf.Get<half>();"""
assert old in s, "locals"
s = s.replace(old, new)

# members: add sq2Buf, sumTmpBuf, wTileBuf
old = """    AscendC::TBuf<AscendC::TPosition::VECCALC> rFp32Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumSqBuf;"""
new = """    AscendC::TBuf<AscendC::TPosition::VECCALC> rFp32Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumSqBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sq2Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumTmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> wTileBuf;"""
assert old in s, "members"
s = s.replace(old, new)

open("scratch/kernel_v26.cpp", "wb").write(s.encode())
print("v26 written, bytes:", len(s))
