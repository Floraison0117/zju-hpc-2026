#!/usr/bin/env python3
# Milestone C round 1: prolong3 interior/boundary split (mirror of rhs split,
# milestone-B-rhs-ceiling.md §3.5/§3.6 + search-memory iteration 22).
#
#   - Creates a fresh candidate tree (cp -r ~/lab4-gpu, minus evidence/build).
#   - fmisc.h: d_symmetry_bd_1b gains a PROLONG3_INTERIOR pure-load fast path
#     (dormant in the base build; active only in the int TU).
#   - Creates src/prolongrestrict_cell_gpu_int.cu = verbatim copy of
#     prolongrestrict_cell_gpu.cu with:
#       * #define PROLONG3_INTERIOR prepended
#       * interior-only early return in d_prolong3_device (guarded by the define)
#       * kernels/launches renamed (prolong3_kernel_int / gpu_prolong3_launch_int
#         / restrict3_kernel_int / gpu_restrict3_launch_int) to avoid duplicate
#         global symbols at link time.
#   - prolongrestrict_cell_gpu.cu: d_prolong3_device / prolong3_kernel /
#     gpu_prolong3_launch gain a trailing `int skip_interior` parameter
#     (0 = deployed behavior); when skip_interior=1 the kernel early-returns
#     for interior points (boundary-shell mode).
#   - prolongrestrict.h: declarations updated + gpu_prolong3_launch_int decl.
#   - Parallel_GPU.cpp (single host call site, case 3): original launch passes
#     skip_interior=1; int launch appended on the same stream (disjoint writes).
#   - CMakeLists.txt: add src/prolongrestrict_cell_gpu_int.cu to ABEGPU CUDA
#     sources.
#
# Interior definition (derived from d_symmetry_bd_1b semantics + prolong3
# stencil): prolong3 taps form a 6x6x6 coarse-index cube [cxI_d-2, cxI_d+3]
# per dim. d_symmetry_bd_1b is identity (pure load == masked value, bit-exact)
# iff every tap is in [1, extc[d]] and > 0 (no out-of-range zero, no
# reflection, factor==1.0 with IEEE x*1.0==x):
#     cxI_i in [3, extc[0]-3], cxI_j in [3, extc[1]-3], cxI_k in [3, extc[2]-3].
# The check uses the actual on-device cxI values (identical formulas in both
# kernels) -> the interior/boundary partition is exact and consistent; both
# kernels keep the interpolation formulas token-identical.
import hashlib, os, re, sys, shutil

FORMAL = os.path.expanduser("~/lab4-gpu")
EXCLUDE_DIRS = {"evidence", "build", "__pycache__"}

# --- fmisc.h d_symmetry_bd_1b pure-load fast path ---
SYM_START = """) {
\t// out-of-range stays zero, matching funcc = 0.d0 initialization
"""
SYM_PURE = """) {
#ifdef PROLONG3_INTERIOR
\t// P23 interior fast path: active only in the prolongrestrict_cell_gpu_int
\t// TU. At interior prolong3 points all 216 coarse stencil taps are strictly
\t// inside [1, extc] and non-reflective, so the range/reflection masks are
\t// identity and factor==1.0 (IEEE x*1.0==x) -> plain load is bit-exact.
\treturn f_at_1b(func, extc, i1b, j1b, k1b);
#else
\t// out-of-range stays zero, matching funcc = 0.d0 initialization
"""
SYM_END = """\treturn f_at_1b(func, extc, ii, jj, kk) * factor;
}
"""
SYM_END_PURE = """\treturn f_at_1b(func, extc, ii, jj, kk) * factor;
#endif
}
"""

# --- int TU: interior early return (inserted after cxI_k computation) ---
CXIK = "    int cxI_k = (k_1b + lbf[2] - 1) / 2 - lbc[2] + 1;\n"
INT_EARLY = """    int cxI_k = (k_1b + lbf[2] - 1) / 2 - lbc[2] + 1;
#ifdef PROLONG3_INTERIOR
    // interior-only fast path: all 216 coarse stencil taps (6x6x6) strictly
    // inside [1, extc[d]] and non-reflective -> d_symmetry_bd_1b masks are
    // identity -> fmisc.h pure-load path is bit-exact for these points.
    if (cxI_i < 3 || cxI_i > extc[0] - 3 ||
        cxI_j < 3 || cxI_j > extc[1] - 3 ||
        cxI_k < 3 || cxI_k > extc[2] - 3) return;
#endif
"""

# --- original TU: boundary-mode skip (inserted after cxI_k computation) ---
SKIP_INTERIOR = """    int cxI_k = (k_1b + lbf[2] - 1) / 2 - lbc[2] + 1;

    // boundary-only mode (skip_interior=1): interior points are computed by
    // prolong3_kernel_int in prolongrestrict_cell_gpu_int.cu; skip them here
    // (deployed behavior at skip_interior=0: condition never matches).
    if (skip_interior) {
        if (cxI_i >= 3 && cxI_i <= extc[0] - 3 &&
            cxI_j >= 3 && cxI_j <= extc[1] - 3 &&
            cxI_k >= 3 && cxI_k <= extc[2] - 3) return;
    }
"""

# --- signatures ---
DEV_SIG_IN = """    const double* llbp, const double* uubp,
    const double* SoA, int Symmetry
) {"""
DEV_SIG_OUT = """    const double* llbp, const double* uubp,
    const double* SoA, int Symmetry, int skip_interior
) {"""

KERN_SIG_IN = """    double* __restrict__ d_dst_f,
    double llbt0, double llbt1, double llbt2,
    double uubt0, double uubt1, double uubt2,
    double SoA0, double SoA1, double SoA2,
    int Symmetry
) {"""
KERN_SIG_OUT = """    double* __restrict__ d_dst_f,
    double llbt0, double llbt1, double llbt2,
    double uubt0, double uubt1, double uubt2,
    double SoA0, double SoA1, double SoA2,
    int Symmetry, int skip_interior
) {"""

DEV_CALL_IN = """        arr_llbf, arr_uubf, arr_extf, d_dst_f,
        arr_llbt, arr_uubt,
        arr_SoA, Symmetry
    );"""
DEV_CALL_OUT = """        arr_llbf, arr_uubf, arr_extf, d_dst_f,
        arr_llbt, arr_uubt,
        arr_SoA, Symmetry, skip_interior
    );"""

LAUNCH_SIG_IN = """void gpu_prolong3_launch(
    cudaStream_t stream,
    const double* d_src_c, double* d_dst_f,
    const double* llbc, const double* uubc, const int* extc,
    const double* llbf, const double* uubf, const int* extf,
    const double* llbt, const double* uubt,
    const double* SoA, int Symmetry
) {"""
LAUNCH_SIG_OUT = """void gpu_prolong3_launch(
    cudaStream_t stream,
    const double* d_src_c, double* d_dst_f,
    const double* llbc, const double* uubc, const int* extc,
    const double* llbf, const double* uubf, const int* extf,
    const double* llbt, const double* uubt,
    const double* SoA, int Symmetry, int skip_interior
) {"""

LAUNCH_CALL_IN = """        d_dst_f,
        llbt[0], llbt[1], llbt[2],
        uubt[0], uubt[1], uubt[2],
        SoA[0], SoA[1], SoA[2],
        Symmetry
    );"""
LAUNCH_CALL_OUT = """        d_dst_f,
        llbt[0], llbt[1], llbt[2],
        uubt[0], uubt[1], uubt[2],
        SoA[0], SoA[1], SoA[2],
        Symmetry, skip_interior
    );"""

# --- host call site (Parallel_GPU.cpp case 3) ---
HOST_CALL_IN = """                        case 3: {
                            gpu_prolong3_launch(
                                src->data->Bg->stream,
                                d_src_ptr, d_dst_ptr, // src_c, dst_f
                                src->data->Bg->bbox, src->data->Bg->bbox + dim, src->data->Bg->shape, 
                                dst->data->llb, dst->data->uub, dst->data->shape,        
                                dst->data->llb, dst->data->uub, 
                                varls->data->SoA, Symmetry
                            );
                            break;
                        }"""
HOST_CALL_OUT = """                        case 3: {
                            gpu_prolong3_launch(
                                src->data->Bg->stream,
                                d_src_ptr, d_dst_ptr, // src_c, dst_f
                                src->data->Bg->bbox, src->data->Bg->bbox + dim, src->data->Bg->shape, 
                                dst->data->llb, dst->data->uub, dst->data->shape,        
                                dst->data->llb, dst->data->uub, 
                                varls->data->SoA, Symmetry, 1
                            );
                            gpu_prolong3_launch_int(
                                src->data->Bg->stream,
                                d_src_ptr, d_dst_ptr, // src_c, dst_f
                                src->data->Bg->bbox, src->data->Bg->bbox + dim, src->data->Bg->shape, 
                                dst->data->llb, dst->data->uub, dst->data->shape,        
                                dst->data->llb, dst->data->uub, 
                                varls->data->SoA, Symmetry, 0
                            );
                            break;
                        }"""


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    if len(sys.argv) != 2:
        print("usage: patch_p23_prolong3_int.py <candidate_root>"); sys.exit(2)
    cand = sys.argv[1]
    if os.path.exists(cand):
        print(f"ERROR: {cand} exists"); sys.exit(3)

    # ---- 0. hash-guard formal baseline ----
    formal_hash = {f: sha(os.path.join(FORMAL, "src", f)) for f in
                   ("fmisc.h", "prolongrestrict_cell_gpu.cu", "prolongrestrict.h",
                    "Parallel_GPU.cpp")}
    formal_hash["CMakeLists.txt"] = sha(os.path.join(FORMAL, "CMakeLists.txt"))
    print("formal src hashes:")
    for f, h in formal_hash.items():
        print(f"  {f} {h[:16]}")
    assert formal_hash["fmisc.h"].startswith("56ad30e4"), "fmisc.h drifted"
    assert formal_hash["prolongrestrict_cell_gpu.cu"].startswith("a81cd33e"), "prolongrestrict_cell_gpu.cu drifted"
    assert formal_hash["prolongrestrict.h"].startswith("d5b08789"), "prolongrestrict.h drifted"
    assert formal_hash["Parallel_GPU.cpp"].startswith("7e0651b1"), "Parallel_GPU.cpp drifted"

    # ---- 1. copy tree (exclude heavy dirs) ----
    shutil.copytree(FORMAL, cand, ignore=shutil.ignore_patterns(*EXCLUDE_DIRS))
    print(f"copied formal -> {cand}")

    # ---- 2. fmisc.h: PROLONG3_INTERIOR pure-load path in d_symmetry_bd_1b ----
    p = os.path.join(cand, "src", "fmisc.h")
    s = open(p).read()
    assert s.count(SYM_START) == 1, f"d_symmetry_bd_1b start anchor: {s.count(SYM_START)}"
    assert s.count(SYM_END) == 1, f"d_symmetry_bd_1b end anchor: {s.count(SYM_END)}"
    s = s.replace(SYM_START, SYM_PURE)
    s = s.replace(SYM_END, SYM_END_PURE)
    open(p, "w").write(s)
    assert s.count("PROLONG3_INTERIOR") == 1, "PROLONG3_INTERIOR guard count != 1"
    print("patched fmisc.h (d_symmetry_bd_1b PROLONG3_INTERIOR pure-load path)")

    # ---- 3. create src/prolongrestrict_cell_gpu_int.cu ----
    src = open(os.path.join(FORMAL, "src", "prolongrestrict_cell_gpu.cu")).read()
    src = "#define PROLONG3_INTERIOR\n" + src
    assert src.count(CXIK) == 1, f"cxI_k anchor count: {src.count(CXIK)}"
    src = src.replace(CXIK, INT_EARLY)
    # same skip_interior signature patches as the base TU (original-form anchors,
    # applied before renames). The int launch passes skip_interior=0, so the
    # boundary skip block (never fires) is NOT added here; the interior
    # early-return above handles boundary points.
    assert src.count(DEV_SIG_IN) == 1 and src.count(KERN_SIG_IN) == 1
    assert src.count(DEV_CALL_IN) == 1 and src.count(LAUNCH_SIG_IN) == 1
    assert src.count(LAUNCH_CALL_IN) == 1
    src = src.replace(DEV_SIG_IN, DEV_SIG_OUT)
    src = src.replace(KERN_SIG_IN, KERN_SIG_OUT)
    src = src.replace(DEV_CALL_IN, DEV_CALL_OUT)
    src = src.replace(LAUNCH_SIG_IN, LAUNCH_SIG_OUT)
    src = src.replace(LAUNCH_CALL_IN, LAUNCH_CALL_OUT)
    # rename kernels + launches (kernel first, then launch; no substring clash)
    src = src.replace("prolong3_kernel", "prolong3_kernel_int")
    src = src.replace("gpu_prolong3_launch", "gpu_prolong3_launch_int")
    src = src.replace("restrict3_kernel", "restrict3_kernel_int")
    src = src.replace("gpu_restrict3_launch", "gpu_restrict3_launch_int")
    # __constant__ arrays must not be defined twice across TUs (-rdc): the int
    # TU declares them extern, resolved to the base TU's definitions at link.
    cprol = "__constant__ double C_PROLONG[6] = {\n    77.0 / 8192.0,    // C1\n    -693.0 / 8192.0,  // C2\n    3465.0 / 4096.0,  // C3\n    1155.0 / 4096.0,  // C4\n    -495.0 / 8192.0,  // C5\n    63.0 / 8192.0     // C6\n};"
    crest = "__constant__ double C_RESTRICT[3] = {\n    3.0 / 256.0,      // C1\n    -25.0 / 256.0,    // C2\n    75.0 / 128.0      // C3\n};"
    assert src.count(cprol) == 1 and src.count(crest) == 1, "__constant__ anchor"
    src = src.replace(cprol, "// P23: C_PROLONG defined in prolongrestrict_cell_gpu.cu\nextern __constant__ double C_PROLONG[6];")
    src = src.replace(crest, "// P23: C_RESTRICT defined in prolongrestrict_cell_gpu.cu\nextern __constant__ double C_RESTRICT[3];")
    ip = os.path.join(cand, "src", "prolongrestrict_cell_gpu_int.cu")
    open(ip, "w").write(src)
    print(f"wrote src/prolongrestrict_cell_gpu_int.cu ({len(src)} bytes)")
    assert "if (cxI_i < 3 || cxI_i > extc[0] - 3" in src
    assert "extern __constant__ double C_PROLONG[6];" in src
    assert "extern __constant__ double C_RESTRICT[3];" in src
    assert src.count("int skip_interior") >= 3, f"int TU skip_interior count: {src.count('int skip_interior')}"
    assert src.count("prolong3_kernel_int") >= 2
    assert src.count("gpu_prolong3_launch_int") == 1
    assert src.count("restrict3_kernel_int") >= 2
    assert src.count("gpu_restrict3_launch_int") == 1

    # ---- 4. prolongrestrict_cell_gpu.cu: skip_interior param + boundary skip ----
    p = os.path.join(cand, "src", "prolongrestrict_cell_gpu.cu")
    s = open(p).read()
    assert s.count(DEV_SIG_IN) == 1, f"device sig: {s.count(DEV_SIG_IN)}"
    assert s.count(KERN_SIG_IN) == 1, f"kernel sig: {s.count(KERN_SIG_IN)}"
    assert s.count(DEV_CALL_IN) == 1, f"device call: {s.count(DEV_CALL_IN)}"
    assert s.count(LAUNCH_SIG_IN) == 1, f"launch sig: {s.count(LAUNCH_SIG_IN)}"
    assert s.count(LAUNCH_CALL_IN) == 1, f"launch call: {s.count(LAUNCH_CALL_IN)}"
    assert s.count(CXIK) == 1, f"cxI_k anchor: {s.count(CXIK)}"
    s = s.replace(DEV_SIG_IN, DEV_SIG_OUT)
    s = s.replace(KERN_SIG_IN, KERN_SIG_OUT)
    s = s.replace(DEV_CALL_IN, DEV_CALL_OUT)
    s = s.replace(LAUNCH_SIG_IN, LAUNCH_SIG_OUT)
    s = s.replace(LAUNCH_CALL_IN, LAUNCH_CALL_OUT)
    s = s.replace(CXIK, SKIP_INTERIOR)
    open(p, "w").write(s)
    assert "int Symmetry, int skip_interior" in s
    assert "if (skip_interior) {" in s
    print("patched prolongrestrict_cell_gpu.cu (skip_interior param + boundary-mode skip)")

    # ---- 5. prolongrestrict.h ----
    p = os.path.join(cand, "src", "prolongrestrict.h")
    s = open(p).read()
    hdev_in = """    const double* llbp, const double* uubp,
    const double* SoA, int Symmetry
);"""
    hdev_out = """    const double* llbp, const double* uubp,
    const double* SoA, int Symmetry, int skip_interior
);"""
    assert s.count(hdev_in) == 1, f"header device decl: {s.count(hdev_in)}"
    s = s.replace(hdev_in, hdev_out)
    hlaunch_in = """void gpu_prolong3_launch(
    cudaStream_t stream,
    const double* d_src_c, double* d_dst_f,
    const double* llbc, const double* uubc, const int* extc,
    const double* llbf, const double* uubf, const int* extf,
    const double* llbt, const double* uubt,
    const double* SoA, int Symmetry
);"""
    hlaunch_out = """void gpu_prolong3_launch(
    cudaStream_t stream,
    const double* d_src_c, double* d_dst_f,
    const double* llbc, const double* uubc, const int* extc,
    const double* llbf, const double* uubf, const int* extf,
    const double* llbt, const double* uubt,
    const double* SoA, int Symmetry, int skip_interior
);

void gpu_prolong3_launch_int(
    cudaStream_t stream,
    const double* d_src_c, double* d_dst_f,
    const double* llbc, const double* uubc, const int* extc,
    const double* llbf, const double* uubf, const int* extf,
    const double* llbt, const double* uubt,
    const double* SoA, int Symmetry, int skip_interior
);"""
    assert s.count(hlaunch_in) == 1, f"header launch decl: {s.count(hlaunch_in)}"
    s = s.replace(hlaunch_in, hlaunch_out)
    open(p, "w").write(s)
    assert "gpu_prolong3_launch_int" in s
    print("patched prolongrestrict.h (skip_interior + int declaration)")

    # ---- 6. Parallel_GPU.cpp: host call site (case 3) ----
    p = os.path.join(cand, "src", "Parallel_GPU.cpp")
    s = open(p).read()
    assert s.count(HOST_CALL_IN) == 1, f"host call anchor: {s.count(HOST_CALL_IN)}"
    s = s.replace(HOST_CALL_IN, HOST_CALL_OUT)
    open(p, "w").write(s)
    assert s.count("gpu_prolong3_launch_int(") == 1
    assert s.count("varls->data->SoA, Symmetry, 1") == 1
    print("patched Parallel_GPU.cpp (skip_interior=1 + int launch appended)")

    # ---- 7. CMakeLists.txt ----
    p = os.path.join(cand, "CMakeLists.txt")
    s = open(p).read()
    old = "src/prolongrestrict_cell_gpu.cu"
    assert s.count(old) == 1
    s = s.replace(old, "src/prolongrestrict_cell_gpu.cu src/prolongrestrict_cell_gpu_int.cu")
    open(p, "w").write(s)
    print("patched CMakeLists.txt (prolongrestrict_cell_gpu_int.cu in ABEGPU_CUDA_SOURCES)")

    # ---- 8. verification ----
    print("\n=== verification ===")
    cand_hash = {f: sha(os.path.join(cand, "src", f)) if f != "CMakeLists.txt" else sha(os.path.join(cand, f)) for f in formal_hash}
    for f in ("fmisc.h", "prolongrestrict_cell_gpu.cu", "prolongrestrict.h", "Parallel_GPU.cpp"):
        print(f"  {f}: formal {formal_hash[f][:16]} -> cand {cand_hash[f][:16]}")
    print(f"  CMakeLists.txt: formal {formal_hash['CMakeLists.txt'][:16]} -> cand {cand_hash['CMakeLists.txt'][:16]}")
    ip = os.path.join(cand, "src", "prolongrestrict_cell_gpu_int.cu")
    print(f"  prolongrestrict_cell_gpu_int.cu: {sha(ip)[:16]}")
    g = open(os.path.join(cand, "src", "Parallel_GPU.cpp")).read()
    assert g.count("gpu_prolong3_launch_int(") == 1, "int launch count != 1"
    print("  1 host call site: skip_interior=1 + int launch verified")
    print("PATCH OK")


if __name__ == "__main__":
    main()
