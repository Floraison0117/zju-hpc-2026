import io
p = 'op_kernel_fused_add_rms_norm.cpp'
s = open(p, encoding='utf-8').read()

old_start = '        const int32_t myRows = static_cast<int32_t>(this->endRow - this->startRow);'
old_end = '            rowBase += n;\n        }\n    }'
i0 = s.index(old_start)
i1 = s.index(old_end, i0) + len(old_end)
old = s[i0:i1]

new = '''        const int32_t myRows = static_cast<int32_t>(this->endRow - this->startRow);
        int32_t nChunks = (myRows + chunk - 1) / chunk;
        uint64_t gmBase = 0;
        int32_t n = 0;

        // Software-pipelined chunk loop: the MTE2 copy of chunk i+1 is issued
        // right after the phase-A vector ops of chunk i (before the V->S sync),
        // so GM->UB transfers of the next chunk overlap the vector/scalar work
        // of the current one (the inQue double buffer holds both chunks).
        // y = R * rstd * w is split as R*w (independent of rstd, issued before
        // the sync) followed by Muls(R, rstd) after the scalar has it, so the
        // post-sync serial region is only 2 vector ops per row.
        AscendC::LocalTensor<half> inLocal;

        // ---- prologue: issue the first chunk copy ----
        {
            gmBase = static_cast<uint64_t>(this->startRow) * static_cast<uint64_t>(H);
            n = (myRows > chunk) ? chunk : myRows;
            inLocal = inQue.AllocTensor<half>();
            copyParams.blockCount = static_cast<uint16_t>(blocksPerRow <= 255 ? n : 2 * n);
            AscendC::DataCopy(inLocal, xGm[gmBase], copyParams);
            AscendC::DataCopy(inLocal[static_cast<uint32_t>(n) * static_cast<uint32_t>(alignH)],
                              residualGm[gmBase], copyParams);
            inQue.EnQue(inLocal);
        }

        for (int32_t c = 0; c < nChunks; ++c) {
            int32_t rowBase = c * chunk;
            n = myRows - rowBase;
            if (n > chunk) n = chunk;
            gmBase = static_cast<uint64_t>(this->startRow + rowBase) *
                     static_cast<uint64_t>(H);

            inLocal = inQue.DeQue<half>();      // waits MTE2 of this chunk

            // ---- Phase A: R = x + residual (FP32), residual_out (FP16), sumSq ----
            AscendC::LocalTensor<half> resOutT = outQueResOut.AllocTensor<half>();
            for (int32_t row = 0; row < n; ++row) {
                AscendC::LocalTensor<half> xL = inLocal[static_cast<uint32_t>(row) * static_cast<uint32_t>(alignH)];
                AscendC::LocalTensor<half> resL = inLocal[static_cast<uint32_t>(n + row) * static_cast<uint32_t>(alignH)];
                AscendC::LocalTensor<float> rRow = rFp32[static_cast<uint32_t>(row) * static_cast<uint32_t>(alignH)];
                AscendC::Cast(rRow, xL, AscendC::RoundMode::CAST_NONE, alignH);
                AscendC::Cast(sq, resL, AscendC::RoundMode::CAST_NONE, alignH);
                AscendC::Add(rRow, rRow, sq, alignH);
                AscendC::Cast(resOutT[static_cast<uint32_t>(row) * static_cast<uint32_t>(alignH)], rRow,
                              AscendC::RoundMode::CAST_NONE, alignH);
                AscendC::Mul(sq, rRow, rRow, alignH);
                ReduceNormal(sumSqArr[static_cast<uint32_t>(row) * 8u], sq, H);
            }
            inQue.FreeTensor(inLocal);

            // ---- issue MTE2 for the next chunk NOW (overlaps this chunk) ----
            if (c + 1 < nChunks) {
                int32_t nNext = myRows - (c + 1) * chunk;
                if (nNext > chunk) nNext = chunk;
                uint64_t gmNext = static_cast<uint64_t>(this->startRow + (c + 1) * chunk) *
                                  static_cast<uint64_t>(H);
                AscendC::LocalTensor<half> inNext = inQue.AllocTensor<half>();
                copyParams.blockCount = static_cast<uint16_t>(blocksPerRow <= 255 ? nNext : 2 * nNext);
                AscendC::DataCopy(inNext, xGm[gmNext], copyParams);
                AscendC::DataCopy(inNext[static_cast<uint32_t>(nNext) * static_cast<uint32_t>(alignH)],
                                  residualGm[gmNext], copyParams);
                inQue.EnQue(inNext);
            }

            // ---- y pre-scale: R *= weight (independent of rstd) ----
            for (int32_t row = 0; row < n; ++row) {
                AscendC::LocalTensor<float> rRow = rFp32[static_cast<uint32_t>(row) * static_cast<uint32_t>(alignH)];
                AscendC::Mul(rRow, rRow, weightFp32, alignH);
            }

            // ---- rstd on the scalar unit: 1/sqrt(sum*invH + eps) ----
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            float rstd[BATCH_MAX_ROWS];
            for (int32_t row = 0; row < n; ++row) {
                float meanPlusEps = sumSqArr.GetValue(static_cast<uint32_t>(row) * 8u) * invH + this->eps;
                rstd[row] = 1.0f / sqrt(meanPlusEps);
            }

            // ---- Phase B: y *= rstd, cast FP16 ----
            AscendC::LocalTensor<half> yT = outQueY.AllocTensor<half>();
            for (int32_t row = 0; row < n; ++row) {
                AscendC::LocalTensor<float> rRow = rFp32[static_cast<uint32_t>(row) * static_cast<uint32_t>(alignH)];
                AscendC::Muls(rRow, rRow, rstd[row], alignH);
                AscendC::Cast(yT[static_cast<uint32_t>(row) * static_cast<uint32_t>(alignH)], rRow,
                              AscendC::RoundMode::CAST_NONE, alignH);
            }

            // ---- CopyOut: residual_out then y (one DataCopy each) ----
            outQueResOut.EnQue(resOutT);
            resOutT = outQueResOut.DeQue<half>();
            AscendC::DataCopy(residualOutGm[gmBase], resOutT, copyParams);
            outQueResOut.FreeTensor(resOutT);

            outQueY.EnQue(yT);
            yT = outQueY.DeQue<half>();
            AscendC::DataCopy(yGm[gmBase], yT, copyParams);
            outQueY.FreeTensor(yT);
        }
    }'''

s = s[:i0] + new + s[i1:]
open(p, 'w', encoding='utf-8').write(s)
print('V8 patched OK')
