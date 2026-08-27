#!/usr/bin/env python3
# Milestone C round 2: restrict3 interior/boundary split (mirror of iter23
# prolong3 split; d_restrict3_device uses d_symmetry_bd_1b on the FINE grid).
#
#   - Creates a fresh candidate tree (cp -r ~/lab4-gpu, minus evidence/build).
#   - prolongrestrict_cell_gpu.cu: d_restrict3_device / restrict3_kernel /
#     gpu_restrict3_launch gain a trailing `int skip_interior` parameter
#     (0 = deployed behavior); boundary-mode skip block inserted after the
#     fine-index computation (interior points skipped when skip_interior=1).
#   - prolongrestrict_cell_gpu_int.cu (existing iter23 int TU, PROLONG3_INTERIOR
#     define already active): d_restrict3_device gains skip_interior + interior
#     early return (guarded by PROLONG3_INTERIOR); restrict3_kernel_int /
#     gpu_restrict3_launch_int gain skip_interior (int launch passes 0).
#   - prolongrestrict.h: gpu_restrict3_launch decl updated + gpu_restrict3_launch_int decl.
#   - Parallel_GPU.cpp (host call site, case 2): original launch passes
#     skip_interior=1; int launch appended on the same stream (disjoint writes).
#
# Interior definition (derived from d_symmetry_bd_1b semantics + restrict3
# stencil): restrict3 fine-grid taps are [fine-2, fine+3] per dim. The masked
# value equals a plain load (bit-exact) iff every tap is in [1, extf[d]] and
# non-reflective (tap >= 1):
#     if_fine in [3, extf[0]-3], jf_fine in [3, extf[1]-3], kf_fine in [3, extf[2]-3].
# The check uses the actual on-device fine-index values (identical formulas in
# both kernels) -> interior/boundary partition is exact and consistent.
import hashlib, os, re, sys, shutil

FORMAL = os.environ.get("P24_FORMAL", os.path.expanduser("~/lab4-gpu"))
EXCLUDE_DIRS = {"evidence", "build", "__pycache__"}

# --- d_restrict3_device signature (base TU; d_prolong3 already has skip) ---
DEV_SIG_IN = """    const double* llbr, const double* uubr,
    const double* SoA, int Symmetry
) {"""
DEV_SIG_OUT = """    const double* llbr, const double* uubr,
    const double* SoA, int Symmetry, int skip_interior
) {"""

# --- fine-index computation anchor (insertion point) ---
FINE3 = """    int if_fine = 2 * (i_1b + lbc[0] - 1) - 1 - lbf[0] + 1;
    int jf_fine = 2 * (j_1b + lbc[1] - 1) - 1 - lbf[1] + 1;
    int kf_fine = 2 * (k_1b + lbc[2] - 1) - 1 - lbf[2] + 1;
"""
SKIP_INTERIOR = FINE3 + """
    // boundary-only mode (skip_interior=1): interior points are computed by
    // restrict3_kernel_int in prolongrestrict_cell_gpu_int.cu; skip them here
    // (deployed behavior at skip_interior=0: condition never matches).
    if (skip_interior) {
        if (if_fine >= 3 && if_fine <= extf[0] - 3 &&
            jf_fine >= 3 && jf_fine <= extf[1] - 3 &&
            kf_fine >= 3 && kf_fine <= extf[2] - 3) return;
    }
"""
INT_EARLY = FINE3 + """#ifdef PROLONG3_INTERIOR
    // interior-only fast path: all fine-grid stencil taps (6 per dim,
    // [fine-2, fine+3]) strictly inside [1, extf[d]] and non-reflective ->
    // d_symmetry_bd_1b masks are identity -> fmisc.h pure-load path is
    // bit-exact for these points.
    if (if_fine < 3 || if_fine > extf[0] - 3 ||
        jf_fine < 3 || jf_fine > extf[1] - 3 ||
        kf_fine < 3 || kf_fine > extf[2] - 3) return;
#endif
"""

# --- restrict3_kernel signature tail (prolong3_kernel already has skip) ---
KERN_SIG_IN = """    double SoA0, double SoA1, double SoA2,
    int Symmetry
) {"""
KERN_SIG_OUT = """    double SoA0, double SoA1, double SoA2,
    int Symmetry, int skip_interior
) {"""

# --- device call in restrict3_kernel ---
DEV_CALL_IN = """    d_restrict3_device(
        i, j, k,
        arr_llbc, arr_uubc, arr_extc, d_dst_c,
        arr_llbf, arr_uubf, arr_extf, d_src_f,
        arr_llbt, arr_uubt,
        arr_SoA, Symmetry
    );"""
DEV_CALL_OUT = """    d_restrict3_device(
        i, j, k,
        arr_llbc, arr_uubc, arr_extc, d_dst_c,
        arr_llbf, arr_uubf, arr_extf, d_src_f,
        arr_llbt, arr_uubt,
        arr_SoA, Symmetry, skip_interior
    );"""

# --- gpu_restrict3_launch signature tail (gpu_prolong3_launch has skip) ---
LAUNCH_SIG_IN = """    const double* llbt, const double* uubt,
    const double* SoA, int Symmetry
) {"""
LAUNCH_SIG_OUT = """    const double* llbt, const double* uubt,
    const double* SoA, int Symmetry, int skip_interior
) {"""

# --- kernel launch tail in gpu_restrict3_launch ---
LAUNCH_CALL_IN = """        SoA[0], SoA[1], SoA[2],
        Symmetry
    );"""
LAUNCH_CALL_OUT = """        SoA[0], SoA[1], SoA[2],
        Symmetry, skip_interior
    );"""

# --- host call site (Parallel_GPU.cpp case 2) ---
HOST_CALL_IN = """                        case 2: {
                            gpu_restrict3_launch(
                                src->data->Bg->stream,
                                d_src_ptr, d_dst_ptr, // src_f, dst_c
                                dst->data->llb, dst->data->uub, dst->data->shape,        
                                src->data->Bg->bbox, src->data->Bg->bbox + dim, src->data->Bg->shape, 
                                dst->data->llb, dst->data->uub, 
                                varls->data->SoA, Symmetry
                            );
                            break;
                        }"""
HOST_CALL_OUT = """                        case 2: {
                            gpu_restrict3_launch(
                                src->data->Bg->stream,
                                d_src_ptr, d_dst_ptr, // src_f, dst_c
                                dst->data->llb, dst->data->uub, dst->data->shape,        
                                src->data->Bg->bbox, src->data->Bg->bbox + dim, src->data->Bg->shape, 
                                dst->data->llb, dst->data->uub, 
                                varls->data->SoA, Symmetry, 1
                            );
                            gpu_restrict3_launch_int(
                                src->data->Bg->stream,
                                d_src_ptr, d_dst_ptr, // src_f, dst_c
                                dst->data->llb, dst->data->uub, dst->data->shape,        
                                src->data->Bg->bbox, src->data->Bg->bbox + dim, src->data->Bg->shape, 
                                dst->data->llb, dst->data->uub, 
                                varls->data->SoA, Symmetry, 0
                            );
                            break;
                        }"""

# --- prolongrestrict.h: gpu_restrict3_launch decl + int decl ---
HDR_IN = """void gpu_restrict3_launch(
    cudaStream_t stream,
    const double* d_src_f, double* d_dst_c,
    const double* llbc, const double* uubc, const int* extc,
    const double* llbf, const double* uubf, const int* extf,
    const double* llbt, const double* uubt,
    const double* SoA, int Symmetry
);"""
HDR_OUT = """void gpu_restrict3_launch(
    cudaStream_t stream,
    const double* d_src_f, double* d_dst_c,
    const double* llbc, const double* uubc, const int* extc,
    const double* llbf, const double* uubf, const int* extf,
    const double* llbt, const double* uubt,
    const double* SoA, int Symmetry, int skip_interior
);

void gpu_restrict3_launch_int(
    cudaStream_t stream,
    const double* d_src_f, double* d_dst_c,
    const double* llbc, const double* uubc, const int* extc,
    const double* llbf, const double* uubf, const int* extf,
    const double* llbt, const double* uubt,
    const double* SoA, int Symmetry, int skip_interior
);"""


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    if len(sys.argv) != 2:
        print("usage: patch_p24_restrict3_int.py <candidate_root>"); sys.exit(2)
    cand = sys.argv[1]
    if os.path.exists(cand):
        print(f"ERROR: {cand} exists"); sys.exit(3)

    # ---- 0. hash-guard formal baseline (deployed iter22 rhs + iter23 prolong3) ----
    formal_hash = {}
    for f in ("prolongrestrict_cell_gpu.cu", "prolongrestrict_cell_gpu_int.cu",
              "prolongrestrict.h", "Parallel_GPU.cpp", "fmisc.h"):
        formal_hash[f] = sha(os.path.join(FORMAL, "src", f))
    formal_hash["CMakeLists.txt"] = sha(os.path.join(FORMAL, "CMakeLists.txt"))
    print("formal src hashes:")
    for f, h in formal_hash.items():
        print(f"  {f} {h[:16]}")
    assert formal_hash["prolongrestrict_cell_gpu.cu"].startswith("6aaaf4a6"), "prolongrestrict_cell_gpu.cu drifted"
    assert formal_hash["prolongrestrict_cell_gpu_int.cu"].startswith("4d0c3acd"), "int TU drifted"
    assert formal_hash["prolongrestrict.h"].startswith("69b82da7"), "prolongrestrict.h drifted"
    assert formal_hash["Parallel_GPU.cpp"].startswith("1f7b2617"), "Parallel_GPU.cpp drifted"
    assert formal_hash["fmisc.h"].startswith("d5c4f4a2"), "fmisc.h drifted"

    # ---- 1. copy tree ----
    shutil.copytree(FORMAL, cand, ignore=shutil.ignore_patterns(*EXCLUDE_DIRS))
    print(f"copied formal -> {cand}")

    # ---- 2. base TU: prolongrestrict_cell_gpu.cu ----
    p = os.path.join(cand, "src", "prolongrestrict_cell_gpu.cu")
    s = open(p, encoding="utf-8").read()
    assert s.count(DEV_SIG_IN) == 1, f"dev sig anchor: {s.count(DEV_SIG_IN)}"
    assert s.count(KERN_SIG_IN) == 1, f"kernel sig anchor: {s.count(KERN_SIG_IN)}"
    assert s.count(DEV_CALL_IN) == 1, f"dev call anchor: {s.count(DEV_CALL_IN)}"
    assert s.count(LAUNCH_SIG_IN) == 1, f"launch sig anchor: {s.count(LAUNCH_SIG_IN)}"
    assert s.count(LAUNCH_CALL_IN) == 1, f"launch call anchor: {s.count(LAUNCH_CALL_IN)}"
    assert s.count(FINE3) == 1, f"fine3 anchor: {s.count(FINE3)}"
    s = s.replace(DEV_SIG_IN, DEV_SIG_OUT)
    s = s.replace(KERN_SIG_IN, KERN_SIG_OUT)
    s = s.replace(DEV_CALL_IN, DEV_CALL_OUT)
    s = s.replace(LAUNCH_SIG_IN, LAUNCH_SIG_OUT)
    s = s.replace(LAUNCH_CALL_IN, LAUNCH_CALL_OUT)
    s = s.replace(FINE3, SKIP_INTERIOR)
    open(p, "w", encoding="utf-8").write(s)
    assert "int Symmetry, int skip_interior" in s
    assert "if (skip_interior) {" in s
    assert s.count("if (skip_interior) {") == 2, f"skip blocks: {s.count('if (skip_interior) {')}"  # prolong3 + restrict3
    print("patched prolongrestrict_cell_gpu.cu (restrict3 skip_interior + boundary skip)")

    # ---- 3. int TU: prolongrestrict_cell_gpu_int.cu ----
    p = os.path.join(cand, "src", "prolongrestrict_cell_gpu_int.cu")
    s = open(p, encoding="utf-8").read()
    assert s.count(DEV_SIG_IN) == 1, f"int dev sig anchor: {s.count(DEV_SIG_IN)}"
    assert s.count(KERN_SIG_IN) == 1, f"int kernel sig anchor: {s.count(KERN_SIG_IN)}"
    assert s.count(DEV_CALL_IN) == 1, f"int dev call anchor: {s.count(DEV_CALL_IN)}"
    assert s.count(LAUNCH_SIG_IN) == 1, f"int launch sig anchor: {s.count(LAUNCH_SIG_IN)}"
    assert s.count(LAUNCH_CALL_IN) == 1, f"int launch call anchor: {s.count(LAUNCH_CALL_IN)}"
    assert s.count(FINE3) == 1, f"int fine3 anchor: {s.count(FINE3)}"
    s = s.replace(DEV_SIG_IN, DEV_SIG_OUT)
    s = s.replace(KERN_SIG_IN, KERN_SIG_OUT)
    s = s.replace(DEV_CALL_IN, DEV_CALL_OUT)
    s = s.replace(LAUNCH_SIG_IN, LAUNCH_SIG_OUT)
    s = s.replace(LAUNCH_CALL_IN, LAUNCH_CALL_OUT)
    s = s.replace(FINE3, INT_EARLY)
    open(p, "w", encoding="utf-8").write(s)
    assert "if (if_fine < 3 || if_fine > extf[0] - 3" in s
    assert s.count("int skip_interior") >= 4, f"int TU skip count: {s.count('int skip_interior')}"
    print("patched prolongrestrict_cell_gpu_int.cu (restrict3 interior early return + skip sig)")

    # ---- 4. prolongrestrict.h ----
    p = os.path.join(cand, "src", "prolongrestrict.h")
    s = open(p, encoding="utf-8").read()
    assert s.count(HDR_IN) == 1, f"header launch decl: {s.count(HDR_IN)}"
    s = s.replace(HDR_IN, HDR_OUT)
    open(p, "w", encoding="utf-8").write(s)
    assert "gpu_restrict3_launch_int" in s
    print("patched prolongrestrict.h (gpu_restrict3_launch skip + _int decl)")

    # ---- 5. Parallel_GPU.cpp: host call site (case 2) ----
    p = os.path.join(cand, "src", "Parallel_GPU.cpp")
    s = open(p, encoding="utf-8").read()
    assert s.count(HOST_CALL_IN) == 1, f"host call anchor: {s.count(HOST_CALL_IN)}"
    s = s.replace(HOST_CALL_IN, HOST_CALL_OUT)
    open(p, "w", encoding="utf-8").write(s)
    assert s.count("gpu_restrict3_launch_int(") == 1
    # note: `varls->data->SoA, Symmetry, 1` also appears in the deployed
    # prolong3 boundary launch (case 3), so use >= 1 here.
    assert s.count("varls->data->SoA, Symmetry, 1") >= 1
    print("patched Parallel_GPU.cpp (restrict3 skip=1 + int launch appended)")

    # ---- 6. CMakeLists.txt: int TU already listed (iter23) ----
    p = os.path.join(cand, "CMakeLists.txt")
    s = open(p, encoding="utf-8").read()
    assert "src/prolongrestrict_cell_gpu_int.cu" in s, "int TU missing from CMake"
    print("CMakeLists.txt OK (prolongrestrict_cell_gpu_int.cu already listed)")

    # ---- 7. verification ----
    print("\n=== verification ===")
    for f in ("prolongrestrict_cell_gpu.cu", "prolongrestrict_cell_gpu_int.cu",
              "prolongrestrict.h", "Parallel_GPU.cpp", "fmisc.h", "CMakeLists.txt"):
        p = os.path.join(cand, "src", f) if f != "CMakeLists.txt" else os.path.join(cand, f)
        print(f"  {f}: formal {formal_hash[f][:16]} -> cand {sha(p)[:16]}")
    g = open(os.path.join(cand, "src", "Parallel_GPU.cpp"), encoding="utf-8").read()
    assert g.count("gpu_restrict3_launch_int(") == 1, "int launch count != 1"
    print("  1 host call site: restrict3 skip=1 + int launch verified")
    print("PATCH OK")


if __name__ == "__main__":
    main()
