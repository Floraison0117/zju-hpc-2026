#!/usr/bin/env python3
# Build kernel_v30.cpp from kernel_v28c.cpp: speculative prefetch pipeline.
# - TPipe + InitBuffer with FIXED max sizes (tiling-independent) so buffers
#   exist before the tiling read.
# - Before reading tiling: issue speculative DataCopies into the real buffers
#   with the case-2 layout guess (block k owns rows [8k, 8k+8), H=1024):
#     x  -> inLocal[0]      (flat 8192 halves from xGm[k*8192])
#     res -> inLocal[8192]  (flat 8192 halves from residualGm[k*8192])
#     w  -> wHalf           (flat 4096 halves)
#   All under EVENT_ID0 (MTE2_V). The scalar tiling read (~0.9 us HBM latency)
#   then overlaps this MTE2 streaming.
# - After tiling: if the guess exactly matches the real copy parameters
#   (H==1024, n==8, startRow==blockIdx*8, aligned, blockDim==32), the inputs
#   are already in UB (just Wait E0 once) and the copies are skipped.
#   Otherwise Wait E0 (spec copies must finish before UB reuse) and issue the
#   real copies as before. Correctness is unconditional.

s = open("scratch/kernel_v28c.cpp", "rb").read().decode()

# ---------------- 1) Init: split into buffer setup (fixed sizes) + late state
old_init_head = """    __aicore__ inline void Init(GM_ADDR x, GM_ADDR residual, GM_ADDR weight,
                                GM_ADDR y, GM_ADDR residual_out,
                                FusedAddRmsNormTilingData& tiling, AscendC::TPipe* pipeIn) {
        this->pipe = pipeIn;
        this->blockIdx = AscendC::GetBlockIdx();

        this->batchSize = tiling.batchSize;"""
new_init_head = """    // V30: fixed-max buffer setup, callable BEFORE the tiling values are read
    // (the speculative prefetch copies need real UB destinations while the
    // scalar unit is still stalled on the tiling GM read). Sizes cover every
    // shape the tiling can produce: rowsPerChunk*alignedHidden is bounded by
    // the host UB budget to ~8192 elements (or capped at 8 rows).
    __aicore__ inline void InitBuffers(AscendC::TPipe* pipeIn) {
        this->pipe = pipeIn;
        this->blockIdx = AscendC::GetBlockIdx();
        pipeIn->InitBuffer(inBuf, 32u * 1024);        // x | residual chunk (max 32 KB)
        pipeIn->InitBuffer(resOutBuf, 16u * 1024);    // fp16 chunk out (max 16 KB)
        pipeIn->InitBuffer(yBuf, 16u * 1024);
        pipeIn->InitBuffer(rFp32Buf, 32u * 1024);     // fp32 R (max 8192 elems)
        pipeIn->InitBuffer(sqBuf, 32u * 1024);        // fp32 squares
        pipeIn->InitBuffer(sumSqBuf, 8u * 8u * sizeof(float));
        pipeIn->InitBuffer(weightHalfBuf, 8u * 1024);   // 4096 halves
        pipeIn->InitBuffer(weightFp32Buf, 16u * 1024);  // 4096 floats
        pipeIn->InitBuffer(scalarBuf, 32);
        pipeIn->InitBuffer(inQueX, BUFFER_NUM, 8u * 1024);
        pipeIn->InitBuffer(inQueRes, BUFFER_NUM, 8u * 1024);
        pipeIn->InitBuffer(outQueY, BUFFER_NUM, 8u * 1024);
        pipeIn->InitBuffer(outQueResOut, BUFFER_NUM, 8u * 1024);
        pipeIn->InitBuffer(resoFp32Buf, 16u * 1024);
        pipeIn->InitBuffer(reduceTmpBuf, 32);
    }

    __aicore__ inline void InitLate(GM_ADDR x, GM_ADDR residual, GM_ADDR weight,
                                    GM_ADDR y, GM_ADDR residual_out,
                                    FusedAddRmsNormTilingData& tiling) {
        this->batchSize = tiling.batchSize;"""
assert old_init_head in s, "init head"
s = s.replace(old_init_head, new_init_head)

# ---------------- 2) remove the old InitBuffer blocks (batch + else branches)
i = s.find("        // UB buffers.\n")
j = s.find("        // Row-parallel split with quotient/remainder")
assert i != -1 and j != -1
s = s[:i] + s[j:]

# ---------------- 3) batch path: skip-copy fast path when guess matched
old_load = """            AscendC::DataCopy(inLocal, xGm[gmBase], copyParams);
            AscendC::DataCopy(inLocal[static_cast<uint32_t>(n) * static_cast<uint32_t>(alignH)],
                              residualGm[gmBase], copyParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);"""
new_load = """            // V30: when the speculative prefetch already moved this exact
            // chunk (guess matched), skip the copies; otherwise consume the
            // spec event (UB safety) and issue the real copies.
            const bool guessHit = this->specHit && (rowBase == 0) &&
                                  (n == this->specRows);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            if (!guessHit) {
                AscendC::DataCopy(inLocal, xGm[gmBase], copyParams);
                AscendC::DataCopy(inLocal[static_cast<uint32_t>(n) * static_cast<uint32_t>(alignH)],
                                  residualGm[gmBase], copyParams);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            }"""
assert old_load in s, "load block"
s = s.replace(old_load, new_load)

# ---------------- 4) weight: already prefetched by the spec copy
old_w = """            // V28a: the weight copy rides the SAME MTE2_V event as the chunk
            // inputs (one fewer Set/Wait pair on the critical path; the small
            // Cast runs on the V pipe right after phase A, well before phase B
            // first reads weightFp32).
            if (rowBase == 0) {
                AscendC::DataCopy(wHalf, weightGm[0], static_cast<uint32_t>(H));
            }
"""
new_w = """            // V30: weight is already in wHalf (speculative prefetch covers
            // up to 4096 halves; only the first alignH are ever read).
"""
assert old_w in s, "weight block"
s = s.replace(old_w, new_w)

# ---------------- 5) chunked path: consume the spec event at entry
old_chunked = """    __aicore__ inline void ProcessChunkedRows() {
        AscendC::LocalTensor<float> weightFp32 = weightFp32Buf.Get<float>();"""
new_chunked = """    __aicore__ inline void ProcessChunkedRows() {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);  // consume spec event
        AscendC::LocalTensor<float> weightFp32 = weightFp32Buf.Get<float>();"""
assert old_chunked in s, "chunked entry"
s = s.replace(old_chunked, new_chunked)

# ---------------- 6) members: specHit/specRows
old_mem = """    bool aligned;
    bool useBatch;
    float eps;"""
new_mem = """    bool aligned;
    bool useBatch;
    bool specHit;
    int32_t specRows;
    float eps;"""
assert old_mem in s, "members"
s = s.replace(old_mem, new_mem)

# ---------------- 7) InitLate tail: spec-hit check before the row split
old_tail = """        // Row-parallel split with quotient/remainder: q = B / blockNum rows for"""
new_tail = """        // V30: did the speculative prefetch land exactly this block's chunk?
        // (H = 1024, 8 rows starting at row 8*blockIdx, aligned, 32 blocks —
        // the scored 256x1024 tiling. Any other shape re-copies.)
        {
            int32_t blockNumL = static_cast<int32_t>(AscendC::GetBlockNum());
            int32_t qL = this->batchSize / blockNumL;
            int32_t rL = this->batchSize % blockNumL;
            int64_t expectStart = static_cast<int64_t>(this->blockIdx) * 8;
            int64_t actualStart = static_cast<int64_t>(this->blockIdx) * qL +
                                  (this->blockIdx < rL ? this->blockIdx : rL);
            this->specHit = this->useBatch && this->hiddenSize == 1024 &&
                            blockNumL == 32 && qL == 8 && rL == 0 &&
                            actualStart == expectStart;
            this->specRows = 8;
        }

        // Row-parallel split with quotient/remainder: q = B / blockNum rows for"""
assert old_tail in s, "tail anchor"
s = s.replace(old_tail, new_tail)

open("scratch/kernel_v30_stage1.cpp", "wb").write(s.encode())
print("v30 stage1 written, bytes:", len(s))