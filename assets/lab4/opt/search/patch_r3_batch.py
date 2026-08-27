#!/usr/bin/env python3
# P-R3BATCH: restrict3 variable-batched launch (发射结构杠杆).
#
# Mechanism: restrict3_kernel is launched once per (block-pair, variable) in
# Parallel_GPU.cpp gpu_data_packer case 2: 294,975 launches, 24.58s/3.98%,
# avg 0.083ms, blocks/launch min/med/max = 1/29/72, 128 regs -> 25% occ,
# waves_per_sm = 1.04 (grid 29 不足两波).  The grid is tiny and the SM is
# idle for most of the launch; the host also recomputes the geometry
# (CD/FD/base/i_start..) once per variable (identical across vars).
#
# Fix: one kernel per (block-pair) covering all variables: 2-D grid
# (ceil(total/256), num_var); blockIdx.y = var_idx.  Geometry scalars
# (computed once, identical across vars) + per-var arrays (d_dst_c[v],
# d_src_f[v], SoA[3v..3v+2]).  Each thread (idx, v) computes exactly what the
# per-variable launch's thread idx computed for variable v.  Bit-exact by
# construction (same arithmetic, same addresses).
#
# Files changed:
#   src/prolongrestrict_cell_gpu.cu : add restrict3_multi_kernel + launcher
#   src/prolongrestrict.h           : declare launcher
#   src/Parallel_GPU.cpp            : case 2 gather vars -> one multi launch
import hashlib
import sys

GUARDS = {
    "src/prolongrestrict_cell_gpu.cu": "44b8dc55b7a7421dd6389690c7ca363dd9a476dbce2954f787d840cb895382a0",
    "src/prolongrestrict.h": "69b82da7b76d045eec966ad1d785ee925562766b191705958007a2b97dcdde8d",
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

    # ---------------- 1. prolongrestrict_cell_gpu.cu: multi kernel + launcher ----------------
    p = f"{root}/src/prolongrestrict_cell_gpu.cu"
    s = open(p).read()
    if "restrict3_multi_kernel" in s:
        print("ALREADY_PATCHED prolongrestrict_cell_gpu.cu")
        sys.exit(1)

    # Insert before the original restrict3_kernel definition.
    anchor = "__global__ __launch_bounds__(256, 2) void restrict3_kernel("
    assert s.count(anchor) == 1, "restrict3_kernel anchor"

    multi = r'''// ---------------------------------------------------------------------------
// P-R3BATCH: variable-batched restrict3.
// One launch per (block-pair) covering all variables: grid = (ceil(total/256),
// num_var); blockIdx.y = var_idx.  Geometry scalars are identical across
// variables (computed once on host); per-variable arrays select the field.
// Thread (idx, v) performs the identical computation as the per-variable
// launch's thread idx for variable v.  Bit-exact by construction.
// ---------------------------------------------------------------------------
#define R3_MAXVARS 24
struct R3Batch {
    double* d_dst_c[R3_MAXVARS];
    const double* d_src_f[R3_MAXVARS];
    double SoA[R3_MAXVARS][3];
};

__global__ __launch_bounds__(256, 2) void restrict3_multi_kernel(
    int ni, int nj, int nk,
    int i_start, int j_start, int k_start,
    double llbc0, double llbc1, double llbc2,
    double uubc0, double uubc1, double uubc2,
    int extc0, int extc1, int extc2,
    double llbf0, double llbf1, double llbf2,
    double uubf0, double uubf1, double uubf2,
    int extf0, int extf1, int extf2,
    double llbt0, double llbt1, double llbt2,
    double uubt0, double uubt1, double uubt2,
    R3Batch b, int num_var, int Symmetry
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y;
    int total = ni * nj * nk;
    if (idx >= total || v >= num_var) return;

    int k_local = idx / (ni * nj);
    int rem     = idx % (ni * nj);
    int j_local = rem / ni;
    int i_local = rem % ni;

    int i = i_start + i_local;
    int j = j_start + j_local;
    int k = k_start + k_local;

    double arr_llbc[3] = {llbc0, llbc1, llbc2};
    double arr_uubc[3] = {uubc0, uubc1, uubc2};
    int    arr_extc[3] = {extc0, extc1, extc2};

    double arr_llbf[3] = {llbf0, llbf1, llbf2};
    double arr_uubf[3] = {uubf0, uubf1, uubf2};
    int    arr_extf[3] = {extf0, extf1, extf2};

    double arr_llbt[3] = {llbt0, llbt1, llbt2};
    double arr_uubt[3] = {uubt0, uubt1, uubt2};
    double arr_SoA[3]  = {b.SoA[v][0], b.SoA[v][1], b.SoA[v][2]};

    d_restrict3_device(
        i, j, k,
        arr_llbc, arr_uubc, arr_extc, b.d_dst_c[v],
        arr_llbf, arr_uubf, arr_extf, b.d_src_f[v],
        arr_llbt, arr_uubt,
        arr_SoA, Symmetry
    );
}

'''
    s = s.replace(anchor, multi + anchor)

    # Add launcher after gpu_restrict3_launch.
    old_launcher_tail = r'''    restrict3_kernel<<<grid, block, 0, stream>>>(
        ni, nj, nk, i_start, j_start, k_start,
        llbc[0], llbc[1], llbc[2],
        uubc[0], uubc[1], uubc[2],
        extc[0], extc[1], extc[2],
        d_dst_c,
        llbf[0], llbf[1], llbf[2],
        uubf[0], uubf[1], uubf[2],
        extf[0], extf[1], extf[2],
        d_src_f,
        llbt[0], llbt[1], llbt[2],
        uubt[0], uubt[1], uubt[2],
        SoA[0], SoA[1], SoA[2],
        Symmetry
    );
}'''
    assert s.count(old_launcher_tail) == 1, "restrict3 launcher tail"

    new_launcher = r'''    restrict3_kernel<<<grid, block, 0, stream>>>(
        ni, nj, nk, i_start, j_start, k_start,
        llbc[0], llbc[1], llbc[2],
        uubc[0], uubc[1], uubc[2],
        extc[0], extc[1], extc[2],
        d_dst_c,
        llbf[0], llbf[1], llbf[2],
        uubf[0], uubf[1], uubf[2],
        extf[0], extf[1], extf[2],
        d_src_f,
        llbt[0], llbt[1], llbt[2],
        uubt[0], uubt[1], uubt[2],
        SoA[0], SoA[1], SoA[2],
        Symmetry
    );
}

void gpu_restrict3_multi_launch(
    cudaStream_t stream,
    const double** h_src_fs, double** h_dst_cs, const double** h_SoAs,
    int num_var,
    const double* llbc, const double* uubc, const int* extc,
    const double* llbf, const double* uubf, const int* extf,
    const double* llbt, const double* uubt,
    int Symmetry
) {
    if (num_var <= 0 || num_var > R3_MAXVARS) {
        // fallback: per-variable launches via the original launcher
        for (int v = 0; v < num_var; ++v) {
            gpu_restrict3_launch(stream, h_src_fs[v], h_dst_cs[v],
                llbc, uubc, extc, llbf, uubf, extf, llbt, uubt,
                h_SoAs[v], Symmetry);
        }
        return;
    }

    double CD[3], FD[3], base[3];
    for(int d = 0; d < 3; d++) {
        CD[d] = (uubc[d] - llbc[d]) / (double)extc[d];
        FD[d] = (uubf[d] - llbf[d]) / (double)extf[d];
        if (llbc[d] <= llbf[d]) {
            base[d] = llbc[d];
        } else {
            int j_val = (int)std::trunc((llbc[d] - llbf[d]) / FD[d] + 0.4);
            if ((j_val / 2) * 2 == j_val) base[d] = llbf[d];
            else base[d] = llbf[d] - CD[d] / 2.0;
        }
    }

    int i_start, i_end, j_start, j_end, k_start, k_end;
    for(int d = 0; d < 3; d++) {
        int lbr = (int)std::trunc((llbt[d] - base[d]) / CD[d] + 0.4) + 1;
        int ubr = (int)std::trunc((uubt[d] - base[d]) / CD[d] + 0.4);
        int lbc = (int)std::trunc((llbc[d] - base[d]) / CD[d] + 0.4) + 1;

        if (d == 0) { i_start = lbr - lbc; i_end = ubr - lbc; }
        if (d == 1) { j_start = lbr - lbc; j_end = ubr - lbc; }
        if (d == 2) { k_start = lbr - lbc; k_end = ubr - lbc; }
    }

    int ni = i_end - i_start + 1;
    int nj = j_end - j_start + 1;
    int nk = k_end - k_start + 1;

    if (ni <= 0 || nj <= 0 || nk <= 0) return;

    int total_points = ni * nj * nk;
    int block = 256;
    int grid_x = (total_points + block - 1) / block;

    R3Batch b;
    for (int v = 0; v < num_var; ++v) {
        b.d_dst_c[v] = h_dst_cs[v];
        b.d_src_f[v] = h_src_fs[v];
        b.SoA[v][0] = h_SoAs[v][0];
        b.SoA[v][1] = h_SoAs[v][1];
        b.SoA[v][2] = h_SoAs[v][2];
    }

    dim3 grid3(grid_x, num_var);
    restrict3_multi_kernel<<<grid3, block, 0, stream>>>(
        ni, nj, nk, i_start, j_start, k_start,
        llbc[0], llbc[1], llbc[2],
        uubc[0], uubc[1], uubc[2],
        extc[0], extc[1], extc[2],
        llbf[0], llbf[1], llbf[2],
        uubf[0], uubf[1], uubf[2],
        extf[0], extf[1], extf[2],
        llbt[0], llbt[1], llbt[2],
        uubt[0], uubt[1], uubt[2],
        b, num_var, Symmetry
    );
}'''
    s = s.replace(old_launcher_tail, new_launcher)
    open(p, "w").write(s)
    print("patched prolongrestrict_cell_gpu.cu")

    # ---------------- 2. prolongrestrict.h: declaration ----------------
    p = f"{root}/src/prolongrestrict.h"
    s = open(p).read()
    if "gpu_restrict3_multi_launch" in s:
        print("ALREADY_PATCHED prolongrestrict.h")
        sys.exit(1)
    anchor = "void gpu_restrict3_launch("
    assert s.count(anchor) == 1, "prolongrestrict.h anchor"
    decl = r'''void gpu_restrict3_multi_launch(
    cudaStream_t stream,
    const double** h_src_fs, double** h_dst_cs, const double** h_SoAs,
    int num_var,
    const double* llbc, const double* uubc, const int* extc,
    const double* llbf, const double* uubf, const int* extf,
    const double* llbt, const double* uubt,
    int Symmetry
);
'''
    s = s.replace(anchor, decl + anchor)
    open(p, "w").write(s)
    print("patched prolongrestrict.h")

    # ---------------- 3. Parallel_GPU.cpp: case 2 -> gather + multi launch ----------------
    p = f"{root}/src/Parallel_GPU.cpp"
    s = open(p).read()
    if "gpu_restrict3_multi_launch" in s:
        print("ALREADY_PATCHED Parallel_GPU.cpp")
        sys.exit(1)

    # Case 2 currently launches per variable.  Restructure: the outer while(src&&dst)
    # loop iterates block pairs; for each pair with dir==PACK && type==2, gather all
    # variables' (src, dst, SoA) and launch one multi kernel.
    old2 = r'''                        case 2: {
                            gpu_restrict3_launch(
                                src->data->Bg->stream,
                                d_src_ptr, d_dst_ptr, // src_f, dst_c
                                dst->data->llb, dst->data->uub, dst->data->shape,        
                                src->data->Bg->bbox, src->data->Bg->bbox + dim, src->data->Bg->shape, 
                                dst->data->llb, dst->data->uub, 
                                varls->data->SoA, Symmetry
                            );
                            break;
                        }'''
    new2 = r'''                        case 2: {
                            // P-R3BATCH: gather all vars of this block pair, one launch
                            // (fired only on the first variable iteration of this block
                            // pair; the per-var loop still advances size_out normally).
                            if (varls == VarLists) {
                                const double* h_src_fs[R3_MAXVARS];
                                double* h_dst_cs[R3_MAXVARS];
                                const double* h_SoAs[R3_MAXVARS];
                                int nvb = 0;
                                MyList<var> *vb = VarLists;
                                int size_b = size_out;
                                while (vb) {
                                    double* ddst = d_data + size_b;
                                    double* dsrc = src->data->Bg->d_fgfs[vb->data->sgfn];
                                    h_src_fs[nvb] = dsrc;
                                    h_dst_cs[nvb] = ddst;
                                    h_SoAs[nvb] = vb->data->SoA;
                                    nvb++;
                                    size_b += dst->data->shape[0] * dst->data->shape[1] * dst->data->shape[2];
                                    vb = vb->next;
                                }
                                gpu_restrict3_multi_launch(
                                    src->data->Bg->stream,
                                    h_src_fs, h_dst_cs, h_SoAs, nvb,
                                    dst->data->llb, dst->data->uub, dst->data->shape,
                                    src->data->Bg->bbox, src->data->Bg->bbox + dim, src->data->Bg->shape,
                                    dst->data->llb, dst->data->uub,
                                    Symmetry
                                );
                            }
                            break;
                        }'''
    assert s.count(old2) == 1, "case 2 anchor"
    s = s.replace(old2, new2)
    open(p, "w").write(s)
    print("patched Parallel_GPU.cpp")
    print("R3 BATCH PATCH DONE")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
