#!/usr/bin/env python3
# P31: global_interp variable-batched launch (发射结构杠杆).
#
# Mechanism: global_interp_kernel is launched per (block, variable) with median
# 1 block/launch (nsys reprofile-20260826: blocks/launch min/med/max = 1/1/144,
# 34,400 launches, 48.83s, avg 1.42ms).  With ~14 SMs available, a 1-block
# launch (256 threads = 8 warps) leaves the GPU almost idle while each thread
# runs a deep latency chain (d_decide3d 216 masked loads + d_polin3_1b Neville
# with ya[216] in local memory).  The host loops over variables launch
# num_var nearly-identical kernels back to back (MassPAng: 17 vars, Wave: 2,
# BH path: 3).
#
# Fix: one kernel per BLOCK covering all variables: 2-D grid
# (ceil(NN/256), num_var); blockIdx.y = var_idx.  Each (j, var_idx) thread
# computes exactly what the per-variable launch's thread j computed:
#   - identical bbox tolerance checks (early return preserved)
#   - identical global_interp_device(...) call with identical SoA values
#   - atomicAdd to the identical, (j,var_idx)-distinct address
#     d_shellf[j*num_var + var_idx]
#   - d_weight atomicAdd from var_idx == 0 only
# Per-block stream / launch order across blocks is untouched, so the set and
# order of atomic contributions per address is unchanged => bit-exact by
# construction.
#
# Files changed (candidate tree only):
#   src/fmisc_gpu.cu       : add global_interp_multi_kernel + launcher
#   src/fmisc.h            : declare gpu_global_interp_multi_launch
#   src/MPatch_gpu.cu      : Interp_Points_GPU + Interp_N_Points_GPU use it
#   src/Parallel_GPU.cpp   : PatList_Interp_Points_GPU uses it
#
# Not touched: global_interp_amr_kernel (1.43s total, negligible), the
# original global_interp_kernel/launcher (kept for reference/base builds).
import hashlib
import sys

GUARDS = {
    "src/fmisc_gpu.cu": "c7cc9f07a708b0afaffa4ff757fcd7913d00a71aa5a0252357cbfaa7abb3cf61",
    "src/fmisc.h": "d5c4f4a2834110a44fa3ab4e5209716031982ec68e415e0b21e461ad8e824685",
    "src/MPatch_gpu.cu": "8474c9973795ace629be5ecb15cc792458d6bd542fb142ded22529afae01a277",
    "src/Parallel_GPU.cpp": "1f7b26175c79c7864c2228bdf9491e6d6f1de3591eadc5eaf3ec7ac861f03db8",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main(root):
    for path, want in GUARDS.items():
        got = sha256(f"{root}/{path}")
        if got != want:
            print(f"GUARD_FAIL {path}: {got} != {want}")
            sys.exit(1)
    print("guards ok")

    # ---------------- 1. fmisc_gpu.cu: add multi kernel + launcher ----------------
    p = f"{root}/src/fmisc_gpu.cu"
    s = open(p).read()

    anchor = "__forceinline__ __device__ double warpReduceSum(double val) {"
    if "global_interp_multi_kernel" in s:
        print("ALREADY_PATCHED fmisc_gpu.cu")
        sys.exit(1)
    assert s.count(anchor) == 1, "fmisc_gpu.cu anchor not unique"

    multi = r'''// ---------------------------------------------------------------------------
// P31: variable-batched global interpolation.
// One kernel per (block) covers all variables: grid = (ceil(NN/256), num_var),
// blockIdx.y == var_idx.  Bit-exact vs the per-variable global_interp_kernel
// launches: thread (j, var_idx) performs the identical bbox check, the
// identical global_interp_device call (same SoA values, same field pointer)
// and atomics to the identical distinct address d_shellf[j*num_var+var_idx].
// ---------------------------------------------------------------------------
__global__ void global_interp_multi_kernel(
    int NN, int DIM,
    double* d_XX_0, double* d_XX_1, double* d_XX_2,
    int ex0, int ex1, int ex2,
    double* d_X_0, double* d_X_1, double* d_X_2,
    double* const* d_fields,
    const double* d_SoA_all,
    double llb_0, double llb_1, double llb_2,
    double uub_0, double uub_1, double uub_2,
    double DH_0, double DH_1, double DH_2,
    int ordn, int Symmetry, int num_var,
    double* d_shellf, int* d_weight
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int var_idx = blockIdx.y;
    if (j >= NN) return;

    // 获取当前点的坐标
    double px = d_XX_0[j];
    double py = d_XX_1[j];
    double pz = d_XX_2[j];

    // 边界检查（Bounding Box 判断）— 使用 DH/2 容差以匹配 CPU 浮点比较行为
    double tol_0 = DH_0 / 2.0;
    double tol_1 = DH_1 / 2.0;
    double tol_2 = DH_2 / 2.0;

    if (px - llb_0 < -tol_0 || px - uub_0 > tol_0) return;
    if (DIM > 1 && (py - llb_1 < -tol_1 || py - uub_1 > tol_1)) return;
    if (DIM > 2 && (pz - llb_2 < -tol_2 || pz - uub_2 > tol_2)) return;

    // 本变量的场指针与 SoA（与逐变量 launch 传入的值完全一致）
    double* d_field = d_fields[var_idx];
    double SoA_0 = d_SoA_all[3 * var_idx + 0];
    double SoA_1 = d_SoA_all[3 * var_idx + 1];
    double SoA_2 = d_SoA_all[3 * var_idx + 2];

    // 组装传给已有插值库的指针
    double* d_X_arr[3] = {d_X_0, d_X_1, d_X_2};
    double SoA_arr[3] = {SoA_0, SoA_1, SoA_2};
    const int ex[3] = {ex0, ex1, ex2};

    // 调用你们原有的设备端插值函数
    double val = 0.0;
    global_interp_device(
        ex, d_X_arr[0], d_X_arr[1], d_X_arr[2],
        d_field, &val,
        px, py, pz,
        ordn, SoA_arr, Symmetry
    );

    // 将结果原子累加到对应位置（处理 Ghost Zone 多个 Block 重叠的情况）
    atomicAdd(&d_shellf[j * num_var + var_idx], val);

    if (var_idx == 0) {
        atomicAdd(&d_weight[j], 1);
    }
}

void gpu_global_interp_multi_launch(
    cudaStream_t stream,
    int NN, int DIM,
    double* d_XX_0, double* d_XX_1, double* d_XX_2,
    int shape_0, int shape_1, int shape_2,
    double* d_X_0, double* d_X_1, double* d_X_2,
    double* const* d_fields, const double* d_SoA_all,
    double llb_0, double llb_1, double llb_2,
    double uub_0, double uub_1, double uub_2,
    double DH_0, double DH_1, double DH_2,
    int ordn, int Symmetry, int num_var,
    double* d_shellf, int* d_weight
) {
    if (NN <= 0 || num_var <= 0) return;
    int blockSize = 256;
    dim3 gridSize((NN + blockSize - 1) / blockSize, num_var);

    global_interp_multi_kernel<<<gridSize, blockSize, 0, stream>>>(
        NN, DIM,
        d_XX_0, d_XX_1, d_XX_2,
        shape_0, shape_1, shape_2, d_X_0, d_X_1, d_X_2,
        d_fields, d_SoA_all,
        llb_0, llb_1, llb_2, uub_0, uub_1, uub_2,
        DH_0, DH_1, DH_2,
        ordn, Symmetry, num_var,
        d_shellf, d_weight
    );
}

'''
    s = s.replace(anchor, multi + anchor)
    open(p, "w").write(s)
    print("patched fmisc_gpu.cu")

    # ---------------- 2. fmisc.h: declaration ----------------
    p = f"{root}/src/fmisc.h"
    s = open(p).read()
    anchor = "void gpu_l2normhelper_launch("
    if "gpu_global_interp_multi_launch" in s:
        print("ALREADY_PATCHED fmisc.h")
        sys.exit(1)
    assert s.count(anchor) == 1, "fmisc.h anchor not unique"
    decl = r'''void gpu_global_interp_multi_launch(
	cudaStream_t stream,
    int NN, int DIM,
    double* d_XX_0, double* d_XX_1, double* d_XX_2,
    int shape_0, int shape_1, int shape_2,
    double* d_X_0, double* d_X_1, double* d_X_2,
    double* const* d_fields, const double* d_SoA_all,
    double llb_0, double llb_1, double llb_2,
    double uub_0, double uub_1, double uub_2,
    double DH_0, double DH_1, double DH_2,
    int ordn, int Symmetry, int num_var,
    double* d_shellf, int* d_weight
);

'''
    s = s.replace(anchor, decl + anchor)
    open(p, "w").write(s)
    print("patched fmisc.h")

    # ---------------- 3. MPatch_gpu.cu: two call sites ----------------
    p = f"{root}/src/MPatch_gpu.cu"
    s = open(p).read()
    if "gpu_global_interp_multi_launch" in s:
        print("ALREADY_PATCHED MPatch_gpu.cu")
        sys.exit(1)

    # 3a. Interp_Points_GPU: per-block staging buffers, freed after sync.
    old_a = r'''            double DH_0 = DH[0];
            double DH_1 = (dim > 1) ? DH[1] : 0.0;
            double DH_2 = (dim > 2) ? DH[2] : 0.0;
            varl = VarList;
            int k = 0;
            while (varl) {
                gpu_global_interp_launch(
                    BP->stream,
                    NN, dim,
                    d_XX[0], d_XX[1], d_XX[2],
                    shape_0, shape_1, shape_2,
                    BP->d_X[0], BP->d_X[1], BP->d_X[2],
                    BP->d_fgfs[varl->data->sgfn],
                    llb_0, llb_1, llb_2,
                    uub_0, uub_1, uub_2,
                    DH_0, DH_1, DH_2,
                    ordn, varl->data->SoA[0], varl->data->SoA[1], varl->data->SoA[2],
                    Symmetry, k, num_var, d_local_shellf, d_local_weight
                );
                varl = varl->next;
                k++;
            }'''
    new_a = r'''            double DH_0 = DH[0];
            double DH_1 = (dim > 1) ? DH[1] : 0.0;
            double DH_2 = (dim > 2) ? DH[2] : 0.0;
            {
                // P31: batch all variables of this block into one launch.
                double** h_fields = new double*[num_var];
                double* h_SoA = new double[3 * num_var];
                varl = VarList;
                int k = 0;
                while (varl) {
                    h_fields[k] = BP->d_fgfs[varl->data->sgfn];
                    h_SoA[3 * k + 0] = varl->data->SoA[0];
                    h_SoA[3 * k + 1] = varl->data->SoA[1];
                    h_SoA[3 * k + 2] = varl->data->SoA[2];
                    varl = varl->next;
                    k++;
                }
                double** d_fields = nullptr;
                double* d_SoA_all = nullptr;
                CUDA_CHECK(cudaMalloc(&d_fields, num_var * sizeof(double*)));
                CUDA_CHECK(cudaMalloc(&d_SoA_all, 3 * num_var * sizeof(double)));
                CUDA_CHECK(cudaMemcpyAsync(d_fields, h_fields, num_var * sizeof(double*),
                                            cudaMemcpyHostToDevice, BP->stream));
                CUDA_CHECK(cudaMemcpyAsync(d_SoA_all, h_SoA, 3 * num_var * sizeof(double),
                                            cudaMemcpyHostToDevice, BP->stream));
                gpu_global_interp_multi_launch(
                    BP->stream,
                    NN, dim,
                    d_XX[0], d_XX[1], d_XX[2],
                    shape_0, shape_1, shape_2,
                    BP->d_X[0], BP->d_X[1], BP->d_X[2],
                    d_fields, d_SoA_all,
                    llb_0, llb_1, llb_2,
                    uub_0, uub_1, uub_2,
                    DH_0, DH_1, DH_2,
                    ordn, Symmetry, num_var, d_local_shellf, d_local_weight
                );
                delete[] h_fields;
                delete[] h_SoA;
                p31_free_fields.push_back(d_fields);
                p31_free_soa.push_back(d_SoA_all);
            }'''
    assert s.count(old_a) == 1, "Interp_Points_GPU var-loop anchor not unique"
    s = s.replace(old_a, new_a)

    # 3b. declare staging-vector before the block loop of Interp_Points_GPU.
    anchor_b = r'''    double *d_local_shellf = GPUManager::getInstance().allocate_device_memory(NN * num_var);
    int *d_local_weight; CUDA_CHECK(cudaMalloc(&d_local_weight, NN * sizeof(int)));
    
    cudaMemset(d_local_shellf, 0, NN * num_var * sizeof(double));
    cudaMemset(d_local_weight, 0, NN * sizeof(int));'''
    assert s.count(anchor_b) == 1, "Interp_Points_GPU staging anchor not unique"
    new_b = anchor_b + r'''

    // P31: per-block device staging buffers (field pointers + SoA), freed
    // after the device sync below so block streams stay concurrent.
    std::vector<double**> p31_free_fields;
    std::vector<double*> p31_free_soa;'''
    s = s.replace(anchor_b, new_b)

    # 3c. free staging after synchronize_all() in Interp_Points_GPU.
    anchor_c = r'''    GPUManager::getInstance().synchronize_all();

    // =================================================================================
    // 3. MPI Allreduce 规约全局数据'''
    assert s.count(anchor_c) == 1, "Interp_Points_GPU sync anchor not unique"
    new_c = r'''    GPUManager::getInstance().synchronize_all();
    for (size_t q = 0; q < p31_free_fields.size(); ++q) CUDA_CHECK(cudaFree(p31_free_fields[q]));
    for (size_t q = 0; q < p31_free_soa.size(); ++q) CUDA_CHECK(cudaFree(p31_free_soa[q]));

    // =================================================================================
    // 3. MPI Allreduce 规约全局数据'''
    s = s.replace(anchor_c, new_c)

    # 3d. Interp_N_Points_GPU: same treatment.
    old_d = r'''            double DH_0 = DH[0];
            double DH_1 = (dim > 1) ? DH[1] : 0.0;
            double DH_2 = (dim > 2) ? DH[2] : 0.0;
            varl = VarList;
            int k = 0;
            while (varl) {
                // 启动你的 GPU Batch Kernel
                gpu_global_interp_launch(
                    BP->stream, NN, dim,
                    d_XX_0, d_XX_1, d_XX_2,
                    BP->shape[0], BP->shape[1], BP->shape[2],
                    BP->d_X[0], BP->d_X[1], BP->d_X[2],
                    BP->d_fgfs[varl->data->sgfn],
                    llb[0], llb[1], llb[2], uub[0], uub[1], uub[2],
                    DH_0, DH_1, DH_2,
                    ordn, varl->data->SoA[0], varl->data->SoA[1], varl->data->SoA[2],
                    Symmetry, k, num_var, d_shellf, d_weight
                );
                varl = varl->next;
                k++;
            }'''
    new_d = r'''            double DH_0 = DH[0];
            double DH_1 = (dim > 1) ? DH[1] : 0.0;
            double DH_2 = (dim > 2) ? DH[2] : 0.0;
            {
                // P31: batch all variables of this block into one launch.
                double** h_fields = new double*[num_var];
                double* h_SoA = new double[3 * num_var];
                varl = VarList;
                int k = 0;
                while (varl) {
                    h_fields[k] = BP->d_fgfs[varl->data->sgfn];
                    h_SoA[3 * k + 0] = varl->data->SoA[0];
                    h_SoA[3 * k + 1] = varl->data->SoA[1];
                    h_SoA[3 * k + 2] = varl->data->SoA[2];
                    varl = varl->next;
                    k++;
                }
                double** d_fields = nullptr;
                double* d_SoA_all = nullptr;
                CUDA_CHECK(cudaMalloc(&d_fields, num_var * sizeof(double*)));
                CUDA_CHECK(cudaMalloc(&d_SoA_all, 3 * num_var * sizeof(double)));
                CUDA_CHECK(cudaMemcpyAsync(d_fields, h_fields, num_var * sizeof(double*),
                                            cudaMemcpyHostToDevice, BP->stream));
                CUDA_CHECK(cudaMemcpyAsync(d_SoA_all, h_SoA, 3 * num_var * sizeof(double),
                                            cudaMemcpyHostToDevice, BP->stream));
                gpu_global_interp_multi_launch(
                    BP->stream, NN, dim,
                    d_XX_0, d_XX_1, d_XX_2,
                    BP->shape[0], BP->shape[1], BP->shape[2],
                    BP->d_X[0], BP->d_X[1], BP->d_X[2],
                    d_fields, d_SoA_all,
                    llb[0], llb[1], llb[2], uub[0], uub[1], uub[2],
                    DH_0, DH_1, DH_2,
                    ordn, Symmetry, num_var, d_shellf, d_weight
                );
                delete[] h_fields;
                delete[] h_SoA;
                p31_free_fields.push_back(d_fields);
                p31_free_soa.push_back(d_SoA_all);
            }'''
    assert s.count(old_d) == 1, "Interp_N_Points_GPU var-loop anchor not unique"
    s = s.replace(old_d, new_d)

    # 3e. staging vectors + free for Interp_N_Points_GPU.
    anchor_e = r'''    MyList<Block> *Bp = blb;
    while (Bp) {
        Block *BP = Bp->data;
        if (myrank == BP->rank) {
            double llb[3], uub[3];'''
    assert s.count(anchor_e) == 1, "Interp_N_Points_GPU loop anchor not unique"
    new_e = r'''    // P31: per-block device staging buffers, freed after the device sync below.
    std::vector<double**> p31_free_fields;
    std::vector<double*> p31_free_soa;
    MyList<Block> *Bp = blb;
    while (Bp) {
        Block *BP = Bp->data;
        if (myrank == BP->rank) {
            double llb[3], uub[3];'''
    s = s.replace(anchor_e, new_e)

    anchor_f = r'''    GPUManager::getInstance().synchronize_all();
    
    double *h_shellf_local = new double[NN * num_var];'''
    assert s.count(anchor_f) == 1, "Interp_N_Points_GPU sync anchor not unique"
    new_f = r'''    GPUManager::getInstance().synchronize_all();
    for (size_t q = 0; q < p31_free_fields.size(); ++q) CUDA_CHECK(cudaFree(p31_free_fields[q]));
    for (size_t q = 0; q < p31_free_soa.size(); ++q) CUDA_CHECK(cudaFree(p31_free_soa[q]));
    
    double *h_shellf_local = new double[NN * num_var];'''
    s = s.replace(anchor_f, new_f)

    # ensure <vector> is available
    if "#include <vector>" not in s:
        s = s.replace('#include "gpu_manager.h"', '#include <vector>\n#include "gpu_manager.h"', 1)
    open(p, "w").write(s)
    print("patched MPatch_gpu.cu")

    # ---------------- 4. Parallel_GPU.cpp: PatList_Interp_Points_GPU ----------------
    p = f"{root}/src/Parallel_GPU.cpp"
    s = open(p).read()
    if "gpu_global_interp_multi_launch" in s:
        print("ALREADY_PATCHED Parallel_GPU.cpp")
        sys.exit(1)

    old_g = r'''                varl = VarList;
                int k = 0;
                while (varl) {
                    gpu_global_interp_launch(
                        stream,
                        NN, dim,
                        d_XX[0], d_XX[1], d_XX[2],
                        shape_0, shape_1, shape_2,
                        BP->d_X[0], BP->d_X[1], BP->d_X[2],
                        BP->d_fgfs[varl->data->sgfn],
                        llb[0], llb[1], llb[2],
                        uub[0], uub[1], uub[2],
                        DH_0, DH_1, DH_2,
                        ordn, varl->data->SoA[0], varl->data->SoA[1], varl->data->SoA[2],
                        Symmetry, k, num_var, d_local_shellf, d_local_weight
                    );
                    varl = varl->next;
                    k ++;
                }'''
    new_g = r'''                {
                    // P31: batch all variables of this block into one launch.
                    double** h_fields = new double*[num_var];
                    double* h_SoA = new double[3 * num_var];
                    varl = VarList;
                    int k = 0;
                    while (varl) {
                        h_fields[k] = BP->d_fgfs[varl->data->sgfn];
                        h_SoA[3 * k + 0] = varl->data->SoA[0];
                        h_SoA[3 * k + 1] = varl->data->SoA[1];
                        h_SoA[3 * k + 2] = varl->data->SoA[2];
                        varl = varl->next;
                        k ++;
                    }
                    double** d_fields = nullptr;
                    double* d_SoA_all = nullptr;
                    cudaMalloc(&d_fields, num_var * sizeof(double*));
                    cudaMalloc(&d_SoA_all, 3 * num_var * sizeof(double));
                    cudaMemcpyAsync(d_fields, h_fields, num_var * sizeof(double*),
                                    cudaMemcpyHostToDevice, stream);
                    cudaMemcpyAsync(d_SoA_all, h_SoA, 3 * num_var * sizeof(double),
                                    cudaMemcpyHostToDevice, stream);
                    gpu_global_interp_multi_launch(
                        stream,
                        NN, dim,
                        d_XX[0], d_XX[1], d_XX[2],
                        shape_0, shape_1, shape_2,
                        BP->d_X[0], BP->d_X[1], BP->d_X[2],
                        d_fields, d_SoA_all,
                        llb[0], llb[1], llb[2],
                        uub[0], uub[1], uub[2],
                        DH_0, DH_1, DH_2,
                        ordn, Symmetry, num_var, d_local_shellf, d_local_weight
                    );
                    delete[] h_fields;
                    delete[] h_SoA;
                    p31_free_fields.push_back(d_fields);
                    p31_free_soa.push_back(d_SoA_all);
                }'''
    assert s.count(old_g) == 1, "PatList var-loop anchor not unique"
    s = s.replace(old_g, new_g)

    # staging vectors: insert before the PL walk (after d_local_shellf alloc).
    anchor_h = r'''    double *d_local_shellf = GPUManager::getInstance().allocate_device_memory(NN * num_var);
    int *d_local_weight; cudaMalloc(&d_local_weight, NN * sizeof(int));
    
    cudaMemset(d_local_shellf, 0, NN * num_var * sizeof(double));
    cudaMemset(d_local_weight, 0, NN * sizeof(int));'''
    assert s.count(anchor_h) == 1, "PatList staging anchor not unique"
    new_h = anchor_h + r'''

    // P31: per-block device staging buffers, freed after the device sync below.
    std::vector<double**> p31_free_fields;
    std::vector<double*> p31_free_soa;'''
    s = s.replace(anchor_h, new_h)

    anchor_i = r'''    GPUManager::getInstance().synchronize_all();

#if MPI_CUDA_AWARE'''
    assert s.count(anchor_i) == 1, "PatList sync anchor not unique"
    new_i = r'''    GPUManager::getInstance().synchronize_all();
    for (size_t q = 0; q < p31_free_fields.size(); ++q) cudaFree(p31_free_fields[q]);
    for (size_t q = 0; q < p31_free_soa.size(); ++q) cudaFree(p31_free_soa[q]);

#if MPI_CUDA_AWARE'''
    s = s.replace(anchor_i, new_i)

    if "#include <vector>" not in s:
        # find an existing #include to prepend after
        idx = s.find("#include")
        eol = s.find("\n", idx)
        s = s[: eol + 1] + "#include <vector>\n" + s[eol + 1 :]
    open(p, "w").write(s)
    print("patched Parallel_GPU.cpp")

    print("P31 PATCH DONE")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
