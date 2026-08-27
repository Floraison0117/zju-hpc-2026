/**
 * @file fused_add_rms_norm.cpp
 * @brief Host tiling + registration for the FusedAddRmsNorm operator.
 *
 * Op: FusedAddRmsNorm(x, residual, weight, eps, enable_pdl) -> (y, residual_out)
 *   residual_out = x + residual
 *   y = residual_out / sqrt(mean(residual_out^2, dim=-1) + eps) * weight
 *
 * Inputs are FP16 ND tensors of shape (B, H); weight is (H,). Outputs mirror the
 * (B, H) shape. The kernel is row-parallel and reduces the H axis in FP32.
 *
 * Tiling carries B, H, the 32B-aligned H (for UB sizing), the align unit, and
 * eps. Multi-core split is over the B (row) axis: each AIV core owns a disjoint
 * contiguous range of rows — no cross-core sync or workspace needed.
 */
#include "../op_kernel/fused_add_rms_norm_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

constexpr int sizeFP16 = 2;
constexpr int alignSizeB = 32;   // 32B UB / DataCopy alignment unit
// ALIGN_NUM mirrors the kernel constant (16 FP16 / 32B). Kept local here so the
// host does not need the kernel's constexpr in scope.
constexpr int32_t kAlignNum = alignSizeB / sizeFP16;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    FusedAddRmsNormTilingData tiling;

    // --- Read shapes: x is (B, H); weight is (H,). ---
    const gert::StorageShape* x_shape = context->GetInputShape(0);
    if (x_shape == nullptr) { return ge::GRAPH_FAILED; }
    const auto& xs = x_shape->GetStorageShape();
    int32_t numDims = static_cast<int32_t>(xs.GetDimNum());

    int32_t B = 1;
    int32_t H = 1;
    if (numDims == 0) {
        // scalar input — treat as a single element (degenerate, but safe).
        B = 1; H = 1;
    } else if (numDims == 1) {
        // (H,) — a single row.
        B = 1;
        H = static_cast<int32_t>(xs.GetDim(0));
    } else {
        // (B, H) (or higher rank — collapse the leading axes into B).
        H = static_cast<int32_t>(xs.GetDim(numDims - 1));
        B = 1;
        for (int32_t i = 0; i < numDims - 1; ++i) {
            B *= static_cast<int32_t>(xs.GetDim(i));
        }
    }
    if (H <= 0) H = 1;
    if (B <= 0) B = 1;

    // 32B-aligned H (in FP16 elements). UB tiles and vector op counts use this;
    // the reduce runs on the exact H (tail ignored).
    int32_t alignedHidden = (H + kAlignNum - 1) / kAlignNum * kAlignNum;

    // UB budget for the aligned batch path (usable bytes; 910B4 UB = 192 KiB,
    // leave margin for the framework/stack). Per-row footprint (V9-6 kernel
    // layout, rows contiguous in UB): in x|res 8*H, R 4*H, squares 4*H,
    // out y|res_out 4*H -> ~20*H bytes per row. With 160 KiB we get
    // rowsPerChunk=8 for H=1024, so 256 rows / 32 blocks = exactly one
    // 8-row chunk per core (balanced, no cross-chunk syncs).
    constexpr int64_t kBatchUbBudget = 160 * 1024;
    int64_t perRowBytes = 20LL * alignedHidden;
    int32_t rowsPerChunk = 1;
    if (alignedHidden <= 4096 && (H % kAlignNum) == 0) {
        rowsPerChunk = static_cast<int32_t>(kBatchUbBudget / perRowBytes);
        if (rowsPerChunk < 1) rowsPerChunk = 1;
        if (rowsPerChunk > 8) rowsPerChunk = 8;
    }

    // --- Read eps (OPTIONAL, default 1e-6). enable_pdl is unused by the kernel. ---
    float eps = 1e-6f;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const float* epsPtr = attrs->GetFloat(0);   // eps (Float, index 0)
        if (epsPtr != nullptr) eps = *epsPtr;
    }

    tiling.set_batchSize(B);
    tiling.set_hiddenSize(H);
    tiling.set_alignedHidden(alignedHidden);
    tiling.set_alignNum(kAlignNum);
    tiling.set_eps(eps);
    tiling.set_rowsPerChunk(rowsPerChunk);

    // --- Multi-core: split rows across AIV cores (capped at B and at the
    // tiling constant below). 256 rows / 32 blocks = 8 rows/core exactly
    // (balanced; 40 blocks would leave 16 cores with 7 rows and 24 with 6,
    // and the 7-row cores become the stragglers). Fewer blocks also pay less
    // kernel-dispatch cost (the course's empty-kernel chart grows with block
    // count). Non-aligned / tiny shapes stay correct under any block count.
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t aivNum = static_cast<int32_t>(ascendcPlatform.GetCoreNumAiv());
    constexpr int32_t kTilingBlockCap = 32;
    int32_t blocks = aivNum < B ? aivNum : B;
    if (blocks > kTilingBlockCap) blocks = kTilingBlockCap;
    context->SetBlockDim(blocks);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
// Output shapes mirror the x/residual input (B, H). y and residual_out both have
// the same shape as x.
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* x_shape = context->GetInputShape(0);
    if (x_shape == nullptr) { return GRAPH_FAILED; }
    gert::Shape* y_shape = context->GetOutputShape(0);
    gert::Shape* resout_shape = context->GetOutputShape(1);
    if (y_shape == nullptr || resout_shape == nullptr) { return GRAPH_FAILED; }

    *y_shape = *x_shape;
    *resout_shape = *x_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    // y and residual_out share x's dtype (FP16).
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class FusedAddRmsNorm : public OpDef {
public:
    explicit FusedAddRmsNorm(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("residual")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("residual_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("eps").AttrType(OPTIONAL).Float(1e-06);
        this->Attr("enable_pdl").AttrType(OPTIONAL).Bool(false);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(FusedAddRmsNorm);
}
