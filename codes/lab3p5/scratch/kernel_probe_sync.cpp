/**
 * @file fused_add_rms_norm.cpp  (PROBE P4/P5: decompose the tiling-read cost)
 * P4 = one bare SetFlag/WaitFlag(MTE2_V) event pair, no data movement.
 * P5 = direct scalar GM read of one tiling word (no MTE2, no UB).
 * P6 = GET_TILING_DATA (framework MTE2+UB+event path) for reference.
 * Select via LAB35_PROBE env: 4 / 5 / 6.
 */
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void fused_add_rms_norm(GM_ADDR x, GM_ADDR residual, GM_ADDR weight,
                                                          GM_ADDR y, GM_ADDR residual_out,
                                                          GM_ADDR workspace, GM_ADDR tiling) {
    (void)x; (void)residual; (void)weight; (void)y; (void)residual_out; (void)workspace;
    int code = 0;
    const __gm__ int32_t* tg = reinterpret_cast<const __gm__ int32_t*>(tiling);
    code = tg[5];   // read rowsPerChunk field position as the selector (P4=4, P5=5, P6=6)

    if (code == 4) {
        // P4: bare event pair, no copy
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        return;
    }
    if (code == 5) {
        // P5: several direct scalar GM reads (tiling struct fields)
        volatile int32_t b = tg[0];
        volatile int32_t h = tg[1];
        volatile int32_t ah = tg[2];
        volatile float eps = *reinterpret_cast<const __gm__ float*>(&tg[4]);
        (void)b; (void)h; (void)ah; (void)eps;
        return;
    }
    if (code == 6) {
        // P6: framework GET_TILING_DATA path for reference
        GET_TILING_DATA(tilingData, tiling);
        volatile int32_t b = tilingData.batchSize;
        (void)b;
        return;
    }
    return;
}
