#!/usr/bin/env python3
# P32: sommerfeld boundary compact launch domain (26a mechanism transplant).
#
# Mechanism: sommerfeld_rout_kernel launches the FULL block volume
# (blocks/launch 343/512/1120, same as rhs) but only points on the 6 single-
# cell-thick face layers that coincide with the PATCH outer bbox do any work
# (is_sommerfeld_boundary early-return otherwise, ~93% of threads).  With 160
# regs -> 12.5% max occupancy the block-serialization waste is large: a block
# whose 7 boundary threads run the 216-load stencil holds its SM slot while
# 249 sibling threads idle.
#
# Fix (host-side): evaluate the six FACE conditions of is_sommerfeld_boundary
# on the host (h_X/h_Y/h_Z are bit-identical host copies of d_X/d_Y/d_Z,
# uploaded once at Block construction).  If no face is active, skip the launch
# entirely.  Otherwise launch ONE compact kernel whose flat 1-D domain covers
# exactly the union of the active face layers (grid = ceil(nBnd/256)).
#
# Bit-exactness: the processed point set of the original launch is exactly
# the union of active face layers; the compact kernel enumerates the same
# union (edge/corner points may be enumerated once per containing active
# layer; each occurrence computes the identical value from the identical
# inputs and writes the identical bits to the same f[ex_idx], so duplicate
# writes are benign).  The per-point kernel body is VERBATIM from the
# original (including the is_sommerfeld_boundary check).
#
# Files changed (candidate tree only):
#   src/sommerfeld_rout_gpu.cu : add sommerfeld_rout_compact_kernel; rewrite
#                               gpu_sommerfeld_rout_launch (host face eval).
#   src/sommerfeld_rout.h      : extend launcher declaration (h_X/h_Y/h_Z).
#   src/bssn_step_gpu.C        : 2 call sites pass cg->X[0..2].
import hashlib
import re
import sys

GUARDS = {
    "src/sommerfeld_rout_gpu.cu": "e9ea7810e7e2a2240dd4e7e114df0ffc8908b1bd1b556d2bec9f9b2604590920",
    "src/sommerfeld_rout.h": "08ba52697681c369f19544bb0d014a8bc6fe22596167424bece8f2fc33c5fece",
    "src/bssn_step_gpu.C": "2bd3b2d9bd8ffe6c4fa3f9250ca6e08c60685bf0d60670493dbe7f58fab3c548",
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

    # ---------------- 1. sommerfeld_rout_gpu.cu ----------------
    p = f"{root}/src/sommerfeld_rout_gpu.cu"
    s = open(p).read()
    if "sommerfeld_rout_compact_kernel" in s:
        print("ALREADY_PATCHED sommerfeld_rout_gpu.cu")
        sys.exit(1)

    # Extract the original kernel text (from '__global__ void sommerfeld_rout_kernel('
    # to the line before '__global__ void sommerfeld_routbam_kernel(').
    m_start = s.index("__global__ void sommerfeld_rout_kernel(")
    m_end = s.index("__global__ void sommerfeld_routbam_kernel(")
    orig_kernel = s[m_start:m_end]

    # Extract the verbatim body: everything from the Fortran 1-based conversion
    # comment onward (kept unchanged in the compact kernel).
    body_anchor = "    // 转换为 Fortran 的 1-based 索引"
    assert orig_kernel.count(body_anchor) == 1
    body = orig_kernel[orig_kernel.index(body_anchor):]

    # Sanity: the prologue we replace is exactly the 3-D decode + bounds check.
    prologue_old = """    int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    int k0 = blockIdx.z * blockDim.z + threadIdx.z;

    if (i0 >= ex0 || j0 >= ex1 || k0 >= ex2) return;

"""
    assert orig_kernel.count(prologue_old) == 1, "sommerfeld prologue anchor missing"

    compact_kernel = r"""// ---------------------------------------------------------------------------
// P32: sommerfeld boundary compact launch domain.
// Flat 1-D domain over the union of the ACTIVE face layers (i==1 / i==ex0 /
// j==1 / j==ex1 / k==1 / k==ex2 that coincide with the patch outer bbox).
// The per-point body below is VERBATIM from sommerfeld_rout_kernel; points
// outside the active face layers were early-return threads in the original
// full-volume launch and are simply never enumerated here.
// ---------------------------------------------------------------------------
__global__ void sommerfeld_rout_compact_kernel(
    int ex0, int ex1, int ex2,
    const double* X, const double* Y, const double* Z,
    double xmin, double ymin, double zmin,
    double xmax, double ymax, double zmax,
    double dT,
    const double* chi0, const double* Lap0,
    const double* f0, double* f,
    const double SYM1, const double SYM2, const double SYM3, 
    int Symmetry,
    int precor,
    int nBnd,
    int off_ihi, int off_jlo, int off_jhi, int off_klo, int off_khi
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nBnd) return;

    int i0, j0, k0;
    if (off_khi >= 0 && t >= off_khi) {
        int u = t - off_khi; i0 = u % ex0; j0 = u / ex0; k0 = ex2 - 1;
    } else if (off_klo >= 0 && t >= off_klo) {
        int u = t - off_klo; i0 = u % ex0; j0 = u / ex0; k0 = 0;
    } else if (off_jhi >= 0 && t >= off_jhi) {
        int u = t - off_jhi; i0 = u % ex0; k0 = u / ex0; j0 = ex1 - 1;
    } else if (off_jlo >= 0 && t >= off_jlo) {
        int u = t - off_jlo; i0 = u % ex0; k0 = u / ex0; j0 = 0;
    } else if (off_ihi >= 0 && t >= off_ihi) {
        int u = t - off_ihi; j0 = u % ex1; k0 = u / ex1; i0 = ex0 - 1;
    } else {
        int u = t; j0 = u % ex1; k0 = u / ex1; i0 = 0;
    }

    if (i0 >= ex0 || j0 >= ex1 || k0 >= ex2) return;

""" + body

    # Insert the compact kernel right BEFORE the original kernel (which stays
    # in the file, unused by the new launcher, for base-build diffs).
    s = s[:m_start] + compact_kernel + "\n" + s[m_start:]

    # Rewrite the launcher.
    launch_old = """void gpu_sommerfeld_rout_launch(
    cudaStream_t &stream,
    int ex[3],
    const double* d_X, const double* d_Y, const double* d_Z,
    double xmin, double ymin, double zmin,
    double xmax, double ymax, double zmax,
    double dT, const double* d_chi0, const double* d_Lap0,
    const double* d_f0, double* d_f, const double SoA[3],
    int Symmetry, int precor
) {
    dim3 block(8, 8, 4);
    dim3 grid((ex[0] + block.x - 1) / block.x, 
              (ex[1] + block.y - 1) / block.y, 
              (ex[2] + block.z - 1) / block.z);

    sommerfeld_rout_kernel<<<grid, block, 0, stream>>>(
        ex[0], ex[1], ex[2], d_X, d_Y, d_Z, xmin, ymin, zmin, xmax, ymax, zmax,
        dT, d_chi0, d_Lap0, d_f0, d_f, SoA[0], SoA[1], SoA[2], Symmetry, precor
    );
}"""
    assert s.count(launch_old) == 1, "sommerfeld launcher anchor not unique"

    launch_new = """void gpu_sommerfeld_rout_launch(
    cudaStream_t &stream,
    int ex[3],
    const double* d_X, const double* d_Y, const double* d_Z,
    const double* h_X, const double* h_Y, const double* h_Z,
    double xmin, double ymin, double zmin,
    double xmax, double ymax, double zmax,
    double dT, const double* d_chi0, const double* d_Lap0,
    const double* d_f0, double* d_f, const double SoA[3],
    int Symmetry, int precor
) {
    // P32: evaluate the six face conditions of is_sommerfeld_boundary on the
    // host.  h_X/h_Y/h_Z are bit-identical host copies of d_X/d_Y/d_Z (each
    // uploaded once at Block construction), so these tests decide exactly the
    // same faces the device-side per-point check would decide.
    const int NO_SYMM_LOC = 0, OCTANT_LOC = 2;
    double dX = h_X[1] - h_X[0];
    double dY = h_Y[1] - h_Y[0];
    double dZ = h_Z[1] - h_Z[0];

    bool f_ihi = fabs(h_X[ex[0] - 1] - xmax) < dX;
    bool f_jhi = fabs(h_Y[ex[1] - 1] - ymax) < dY;
    bool f_khi = fabs(h_Z[ex[2] - 1] - zmax) < dZ;
    bool f_ilo = fabs(h_X[0] - xmin) < dX && !(Symmetry == OCTANT_LOC && fabs(xmin) < dX / 2.0);
    bool f_jlo = fabs(h_Y[0] - ymin) < dY && !(Symmetry == OCTANT_LOC && fabs(ymin) < dY / 2.0);
    bool f_klo = fabs(h_Z[0] - zmin) < dZ && !(Symmetry > NO_SYMM_LOC && fabs(zmin) < dZ / 2.0);

    long A_ilo = f_ilo ? (long)ex[1] * ex[2] : 0;
    long A_ihi = f_ihi ? (long)ex[1] * ex[2] : 0;
    long A_jlo = f_jlo ? (long)ex[0] * ex[2] : 0;
    long A_jhi = f_jhi ? (long)ex[0] * ex[2] : 0;
    long A_klo = f_klo ? (long)ex[0] * ex[1] : 0;
    long A_khi = f_khi ? (long)ex[0] * ex[1] : 0;
    long nBnd = A_ilo + A_ihi + A_jlo + A_jhi + A_klo + A_khi;
    if (nBnd == 0) return;  // no sommerfeld boundary on this block: skip launch

    long o = A_ilo;
    int off_ihi = A_ihi ? (int)o : -1; o += A_ihi;
    int off_jlo = A_jlo ? (int)o : -1; o += A_jlo;
    int off_jhi = A_jhi ? (int)o : -1; o += A_jhi;
    int off_klo = A_klo ? (int)o : -1; o += A_klo;
    int off_khi = A_khi ? (int)o : -1;

    dim3 block(256);
    dim3 grid((unsigned int)((nBnd + block.x - 1) / block.x));

    sommerfeld_rout_compact_kernel<<<grid, block, 0, stream>>>(
        ex[0], ex[1], ex[2], d_X, d_Y, d_Z, xmin, ymin, zmin, xmax, ymax, zmax,
        dT, d_chi0, d_Lap0, d_f0, d_f, SoA[0], SoA[1], SoA[2], Symmetry, precor,
        (int)nBnd, off_ihi, off_jlo, off_jhi, off_klo, off_khi
    );
}"""
    s = s.replace(launch_old, launch_new)
    open(p, "w").write(s)
    print("patched sommerfeld_rout_gpu.cu")

    # ---------------- 2. sommerfeld_rout.h ----------------
    p = f"{root}/src/sommerfeld_rout.h"
    s = open(p).read()
    old_decl = """void gpu_sommerfeld_rout_launch(
    cudaStream_t &stream,
    int ex[3],
    const double* d_X, const double* d_Y, const double* d_Z,
    double xmin, double ymin, double zmin,"""
    assert s.count(old_decl) == 1, "sommerfeld_rout.h anchor not unique"
    new_decl = """void gpu_sommerfeld_rout_launch(
    cudaStream_t &stream,
    int ex[3],
    const double* d_X, const double* d_Y, const double* d_Z,
    const double* h_X, const double* h_Y, const double* h_Z,
    double xmin, double ymin, double zmin,"""
    s = s.replace(old_decl, new_decl)
    open(p, "w").write(s)
    print("patched sommerfeld_rout.h")

    # ---------------- 3. bssn_step_gpu.C: two call sites ----------------
    p = f"{root}/src/bssn_step_gpu.C"
    s = open(p).read()
    if "cg->X[0], cg->X[1], cg->X[2]" in s:
        print("ALREADY_PATCHED bssn_step_gpu.C")
        sys.exit(1)
    n_sites = 0
    # generic: within each gpu_sommerfeld_rout_launch( ... ) argument list,
    # append host coordinate pointers after the d_X triple.
    idx = 0
    while True:
        idx = s.find("gpu_sommerfeld_rout_launch(", idx)
        if idx < 0:
            break
        # find the d_X triple within the next 600 chars
        window = s[idx : idx + 700]
        trip = "cg->d_X[0], cg->d_X[1], cg->d_X[2],"
        t = window.find(trip)
        assert t >= 0, f"call site at {idx} has no d_X triple"
        abs_t = idx + t
        s = s[:abs_t] + "cg->d_X[0], cg->d_X[1], cg->d_X[2], cg->X[0], cg->X[1], cg->X[2]," + s[abs_t + len(trip):]
        n_sites += 1
        idx = abs_t + len(trip) + len(" cg->X[0], cg->X[1], cg->X[2],")
    assert n_sites == 2, f"expected 2 call sites, patched {n_sites}"
    open(p, "w").write(s)
    print(f"patched bssn_step_gpu.C ({n_sites} call sites)")

    print("P32 PATCH DONE")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
