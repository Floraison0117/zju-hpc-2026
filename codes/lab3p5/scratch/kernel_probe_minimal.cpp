/**
 * @file fused_add_rms_norm.cpp  (PROBE P3: minimal binary floor test)
 * Strips everything: no class, no tiling, no TPipe. Measures pure launch floor.
 */
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void fused_add_rms_norm(GM_ADDR x, GM_ADDR residual, GM_ADDR weight,
                                                          GM_ADDR y, GM_ADDR residual_out,
                                                          GM_ADDR workspace, GM_ADDR tiling) {
    (void)x; (void)residual; (void)weight; (void)y; (void)residual_out;
    (void)workspace; (void)tiling;
    return;
}
