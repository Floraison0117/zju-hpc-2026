#!/usr/bin/env python3
# Build kernel_v24.cpp from kernel_v23.cpp: two-half pipelined batch path.
import re

s = open("scratch/kernel_v23.cpp", "rb").read().decode()

start = s.find("    __aicore__ inline void ProcessAlignedBatch() {")
end = s.find("    // ------------------------------------------------------------------\n    //  Whole-row path")
assert start != -1 and end != -1, "anchors not found"

new_pab = '''    __aicore__ inline void ProcessAlignedBatch() {
        AscendC::LocalTensor<half> inLocal = inBuf.Get<half>();
        AscendC::LocalTensor<half> resOutT = resOutBuf.Get<half>();
        AscendC::LocalTensor<half> yT = yBuf.Get<half>();
        AscendC::LocalTensor<float> rFp32 = rFp32Buf.Get<float>();
        AscendC::LocalTensor<float> sq = sqBuf.Get<float>();
        AscendC::LocalTensor<float> sumSqArr = sumSqBuf.Get<float>();
        AscendC::LocalTensor<float> weightFp32 = weightFp32Buf.Get<float>();
        AscendC::LocalTensor<half> wHalf = weightHalfBuf.Get<half>();

        const int32_t H = this->hiddenSize;
        const int32_t alignH = this->alignedHidden;
        const float invH = 1.0f / static_cast<float>(H);
        const int32_t chunk = this->rowsPerChunk;

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

            // V24: split the chunk into two halves; issue BOTH loads up front
            // so MTE2 streams them back-to-back, then compute half0 while
            // MTE2 finishes half1. Halves are contiguous sub-ranges of the
            // same buffers. Odd n puts the extra row in half0.
            const bool split = (n >= 2);
            const int32_t n0 = split ? ((n + 1) / 2) : n;
            const int32_t n1 = split ? (n - n0) : 0;
            const uint32_t e0 = static_cast<uint32_t>(n0) * static_cast<uint32_t>(alignH);
            const uint32_t e1 = static_cast<uint32_t>(n1) * static_cast<uint32_t>(alignH);

            if (rowBase == 0) {
                AscendC::DataCopy(wHalf, weightGm[0], static_cast<uint32_t>(H));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID5);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID5);
                AscendC::Cast(weightFp32, wHalf, AscendC::RoundMode::CAST_NONE, alignH);
            }

            AscendC::DataCopyParams halfParams = copyParams;
            halfParams.blockCount = static_cast<uint16_t>(blocksPerRow <= 255 ? n0 : 2 * n0);
            AscendC::DataCopy(inLocal, xGm[gmBase], halfParams);
            AscendC::DataCopy(inLocal[e0], residualGm[gmBase], halfParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            if (split) {
                AscendC::DataCopyParams half1Params = copyParams;
                half1Params.blockCount = static_cast<uint16_t>(blocksPerRow <= 255 ? n1 : 2 * n1);
                AscendC::DataCopy(inLocal[2u * e0],
                                  xGm[gmBase + static_cast<uint64_t>(n0) * static_cast<uint64_t>(H)],
                                  half1Params);
                AscendC::DataCopy(inLocal[2u * e0 + e1],
                                  residualGm[gmBase + static_cast<uint64_t>(n0) * static_cast<uint64_t>(H)],
                                  half1Params);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            }

            // ---- Phase A on half 0 ----
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Add(resOutT, inLocal, inLocal[e0], e0);
            AscendC::Cast(rFp32, resOutT, AscendC::RoundMode::CAST_NONE, e0);
            AscendC::Mul(sq, rFp32, rFp32, e0);

            // Early residual_out copy for half 0: MTE3 streams it out while V
            // still works on half 1.
            if (split) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
                AscendC::DataCopyParams outH0 = copyParams;
                outH0.blockCount = static_cast<uint16_t>(blocksPerRow <= 255 ? n0 : 2 * n0);
                AscendC::DataCopy(residualOutGm[gmBase], resOutT, outH0);
            }

            // ---- Phase A on half 1 ----
            if (split) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
                AscendC::Add(resOutT[e0], inLocal[2u * e0], inLocal[2u * e0 + e1], e1);
                AscendC::Cast(rFp32[e0], resOutT[e0], AscendC::RoundMode::CAST_NONE, e1);
                AscendC::Mul(sq[e0], rFp32[e0], rFp32[e0], e1);
            }

            // ---- Per-row fold (rows of both halves are contiguous in sq) ----
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

            // ---- rstd = 1 / sqrt(mean + eps) per row (scalar unit) ----
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID1);

            // ---- Phase B, CANN DuplicateMulImpl style ----
            AscendC::SetMaskCount();
            AscendC::SetVectorMask<uint8_t, AscendC::MaskMode::COUNTER>(0, static_cast<uint32_t>(alignH));
            const AscendC::UnaryRepeatParams unaryParams;
            for (int32_t row = 0; row < n; ++row) {
                float meanPlusEps = sumSqArr.GetValue(static_cast<uint32_t>(row)) * invH + this->eps;
                float rstd = 1.0f / sqrt(meanPlusEps);
                AscendC::LocalTensor<float> rRow = rFp32[static_cast<uint32_t>(row) * static_cast<uint32_t>(alignH)];
                AscendC::Muls<float, false>(rRow, rRow, rstd, AscendC::MASK_PLACEHOLDER, 1, unaryParams);
            }
            AscendC::PipeBarrier<PIPE_V>();
            const AscendC::BinaryRepeatParams binaryParams;
            for (int32_t row = 0; row < n; ++row) {
                AscendC::LocalTensor<float> rRow = rFp32[static_cast<uint32_t>(row) * static_cast<uint32_t>(alignH)];
                AscendC::Mul<float, false>(rRow, rRow, weightFp32, AscendC::MASK_PLACEHOLDER, 1, binaryParams);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetMaskNorm();
            AscendC::Cast(yT, rFp32, AscendC::RoundMode::CAST_NONE, chunkElems);

            // ---- CopyOut: residual_out (half1 only when split) then y ----
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
            if (split) {
                AscendC::DataCopyParams outH1 = copyParams;
                outH1.blockCount = static_cast<uint16_t>(blocksPerRow <= 255 ? n1 : 2 * n1);
                AscendC::DataCopy(residualOutGm[gmBase + static_cast<uint64_t>(n0) * static_cast<uint64_t>(H)],
                                  resOutT[e0], outH1);
            } else {
                AscendC::DataCopy(residualOutGm[gmBase], resOutT, copyParams);
            }
            AscendC::DataCopy(yGm[gmBase], yT, copyParams);

            // ---- Single-buffer hazards for the NEXT chunk ----
            if (rowBase + n < myRows) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
            }
            rowBase += n;
        }
    }

'''
s = s[:start] + new_pab + s[end:]
open("scratch/kernel_v24.cpp", "wb").write(s.encode())
print("v24 written, bytes:", len(s))
