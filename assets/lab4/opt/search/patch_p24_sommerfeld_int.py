#!/usr/bin/env python3
# Milestone C round 2: sommerfeld_rout interior/boundary split (mirror of
# iter23 prolong3 split). sommerfeld_rout_kernel is boundary-only; each
# boundary point interpolates a 6x6x6 cube of f0 via d_decide3d + d_polin3_1b.
#
#   - sommerfeld_rout_gpu.cu: sommerfeld_rout_kernel / gpu_sommerfeld_rout_launch
#     gain a trailing `int skip_interior` parameter (0 = deployed behavior);
#     boundary-mode skip block inserted after the window clamps (interior
#     interpolation points skipped when skip_interior=1).
#   - Creates src/sommerfeld_rout_gpu_int.cu = verbatim copy with
#     #define SOMMERFELD_INTERIOR + interior-only early return + renames
#     (is_sommerfeld_boundary_int / sommerfeld_rout_kernel_int /
#     sommerfeld_routbam_kernel_int / gpu_sommerfeld_rout_launch_int /
#     gpu_sommerfeld_routbam_launch_int; the non-forceinline device helper
#     must be renamed to avoid -rdc duplicate symbols). CORRECTSTEP points
#     (plain copy, handled by the boundary kernel) early-return in the int TU.
#   - sommerfeld_rout.h: gpu_sommerfeld_rout_launch decl updated +
#     gpu_sommerfeld_rout_launch_int decl.
#   - bssn_step_gpu.C (2 host call sites, lev>0): original launch passes
#     skip_interior=1; int launch appended on the same stream (disjoint writes).
#   - CMakeLists.txt: add src/sommerfeld_rout_gpu_int.cu.
#
# Interior definition (from d_decide3d semantics): the masked value equals a
# plain load iff every interpolation tap is in [1, ext] and non-reflective:
#     post-clamp cxB[m] >= 1 && cxT[m] <= ext[m] for all m.
# For the deployed equatorial config (Symmetry=1) the k-direction is the only
# reflection-capable axis; the full condition keeps it general. Identical
# formulas in both kernels -> exact partition.
import hashlib, os, re, sys, shutil

FORMAL = os.environ.get("P24_FORMAL", os.path.expanduser("~/lab4-gpu"))
EXCLUDE_DIRS = {"evidence", "build", "__pycache__"}

KERN_SIG_IN = """    int Symmetry,
    int precor
) {"""
KERN_SIG_OUT = """    int Symmetry,
    int precor,
    int skip_interior
) {"""

CLAMP_ANCHOR = """        for (int m = 0; m < 3; ++m) {
            if (cxT[m] > ext[m]) {
                cx[m] += (cxT[m] - ext[m]);
                cxB[m] -= (cxT[m] - ext[m]);
                cxT[m] = ext[m];
            }
        }

        double ya[ORDN * ORDN * ORDN];"""
INT_CHECK = """        for (int m = 0; m < 3; ++m) {
            if (cxT[m] > ext[m]) {
                cx[m] += (cxT[m] - ext[m]);
                cxB[m] -= (cxT[m] - ext[m]);
                cxT[m] = ext[m];
            }
        }

#ifdef SOMMERFELD_INTERIOR
        // interior-only fast path: all 216 interpolation taps strictly inside
        // [1, ext] and non-reflective -> d_decide3d masks are identity ->
        // fmisc.h pure-load variant is bit-exact for these points.
        if (!(cxB[0] >= 1 && cxB[1] >= 1 && cxB[2] >= 1 &&
              cxT[0] <= ext[0] && cxT[1] <= ext[1] && cxT[2] <= ext[2])) return;
#else
        // boundary-only mode (skip_interior=1): interior interpolation points
        // are computed by the int TU kernel; skip them here (deployed
        // behavior at skip_interior=0: condition never matches).
        if (skip_interior) {
            if (cxB[0] >= 1 && cxB[1] >= 1 && cxB[2] >= 1 &&
                cxT[0] <= ext[0] && cxT[1] <= ext[1] && cxT[2] <= ext[2]) return;
        }
#endif

        double ya[ORDN * ORDN * ORDN];"""

# CORRECTSTEP guard for the int TU (insert after the is_sommerfeld_boundary
# early return inside sommerfeld_rout_kernel_int). The ANCHOR uses the _int
# name because renames are applied before this replacement.
BDY_CHECK_INT_ANCHOR = """    // 仅当属于边界时才继续执行计算
    if (!is_sommerfeld_boundary_int(i, j, k, ex0, ex1, ex2, X, Y, Z, xmin, ymin, zmin, xmax, ymax, zmax, Symmetry)) {
        return;
    }
"""
BDY_CHECK_INT = BDY_CHECK_INT_ANCHOR + """
#ifdef SOMMERFELD_INTERIOR
    // P24: CORRECTSTEP is a plain copy handled by the boundary kernel; skip it
    // here (a redundant f[ex_idx] = f0[ex_idx] double-write would be harmless
    // but wasteful).
    if (precor == CORRECTSTEP) return;
#endif
"""

LAUNCH_SIG_IN = """    double dT, const double* d_chi0, const double* d_Lap0,
    const double* d_f0, double* d_f, const double SoA[3],
    int Symmetry, int precor
) {"""
LAUNCH_SIG_OUT = """    double dT, const double* d_chi0, const double* d_Lap0,
    const double* d_f0, double* d_f, const double SoA[3],
    int Symmetry, int precor, int skip_interior
) {"""

LAUNCH_CALL_IN = """    sommerfeld_rout_kernel<<<grid, block, 0, stream>>>(
        ex[0], ex[1], ex[2], d_X, d_Y, d_Z, xmin, ymin, zmin, xmax, ymax, zmax,
        dT, d_chi0, d_Lap0, d_f0, d_f, SoA[0], SoA[1], SoA[2], Symmetry, precor
    );"""
LAUNCH_CALL_OUT = """    sommerfeld_rout_kernel<<<grid, block, 0, stream>>>(
        ex[0], ex[1], ex[2], d_X, d_Y, d_Z, xmin, ymin, zmin, xmax, ymax, zmax,
        dT, d_chi0, d_Lap0, d_f0, d_f, SoA[0], SoA[1], SoA[2], Symmetry, precor,
        skip_interior
    );"""

HDR_IN = """void gpu_sommerfeld_rout_launch(
    cudaStream_t &stream,
    int ex[3],
    const double* d_X, const double* d_Y, const double* d_Z,
    double xmin, double ymin, double zmin,
    double xmax, double ymax, double zmax,
    double dT, const double* d_chi0, const double* d_Lap0,
    const double* d_f0, double* d_f, const double SoA[3],
    int Symmetry, int precor
);"""
HDR_OUT = """void gpu_sommerfeld_rout_launch(
    cudaStream_t &stream,
    int ex[3],
    const double* d_X, const double* d_Y, const double* d_Z,
    double xmin, double ymin, double zmin,
    double xmax, double ymax, double zmax,
    double dT, const double* d_chi0, const double* d_Lap0,
    const double* d_f0, double* d_f, const double SoA[3],
    int Symmetry, int precor, int skip_interior
);

void gpu_sommerfeld_rout_launch_int(
    cudaStream_t &stream,
    int ex[3],
    const double* d_X, const double* d_Y, const double* d_Z,
    double xmin, double ymin, double zmin,
    double xmax, double ymax, double zmax,
    double dT, const double* d_chi0, const double* d_Lap0,
    const double* d_f0, double* d_f, const double SoA[3],
    int Symmetry, int precor, int skip_interior
);"""

# host call sites (bssn_step_gpu.C, lev>0 fix-BD-point blocks)
CALL_RE = re.compile(
    r'(gpu_sommerfeld_rout_launch\([\s\S]*?Symmetry, (?:cor|pre)\s*\);)')
TAIL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*\);$')


def split_som_calls(s):
    def repl(m):
        call = m.group(1)
        bnd = TAIL_RE.sub(r'\1, 1);', call)
        intl = TAIL_RE.sub(r'\1, 0);', call.replace('gpu_sommerfeld_rout_launch(', 'gpu_sommerfeld_rout_launch_int(', 1))
        return bnd + "\n" + intl
    return CALL_RE.sub(repl, s)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    if len(sys.argv) != 2:
        print("usage: patch_p24_sommerfeld_int.py <candidate_root>"); sys.exit(2)
    cand = sys.argv[1]
    if os.path.exists(cand):
        print(f"ERROR: {cand} exists"); sys.exit(3)

    # ---- 0. hash-guard formal baseline ----
    formal_hash = {}
    for f in ("sommerfeld_rout_gpu.cu", "sommerfeld_rout.h", "bssn_step_gpu.C", "fmisc.h"):
        formal_hash[f] = sha(os.path.join(FORMAL, "src", f))
    formal_hash["CMakeLists.txt"] = sha(os.path.join(FORMAL, "CMakeLists.txt"))
    print("formal src hashes:")
    for f, h in formal_hash.items():
        print(f"  {f} {h[:16]}")
    assert formal_hash["sommerfeld_rout_gpu.cu"].startswith("e9ea7810"), "sommerfeld_rout_gpu.cu drifted"
    assert formal_hash["sommerfeld_rout.h"].startswith("08ba5269"), "sommerfeld_rout.h drifted"
    assert formal_hash["bssn_step_gpu.C"].startswith("2bd3b2d9"), "bssn_step_gpu.C drifted"
    assert formal_hash["fmisc.h"].startswith("d5c4f4a2"), "fmisc.h drifted"

    # ---- 1. copy tree ----
    shutil.copytree(FORMAL, cand, ignore=shutil.ignore_patterns(*EXCLUDE_DIRS))
    print(f"copied formal -> {cand}")

    # ---- 2. base TU: sommerfeld_rout_gpu.cu ----
    p = os.path.join(cand, "src", "sommerfeld_rout_gpu.cu")
    s = open(p, encoding="utf-8").read()
    assert s.count(KERN_SIG_IN) == 1, f"kernel sig anchor: {s.count(KERN_SIG_IN)}"
    assert s.count(CLAMP_ANCHOR) == 1, f"clamp anchor: {s.count(CLAMP_ANCHOR)}"
    assert s.count(LAUNCH_SIG_IN) == 1, f"launch sig anchor: {s.count(LAUNCH_SIG_IN)}"
    assert s.count(LAUNCH_CALL_IN) == 1, f"launch call anchor: {s.count(LAUNCH_CALL_IN)}"
    s = s.replace(KERN_SIG_IN, KERN_SIG_OUT)
    s = s.replace(CLAMP_ANCHOR, INT_CHECK)
    s = s.replace(LAUNCH_SIG_IN, LAUNCH_SIG_OUT)
    s = s.replace(LAUNCH_CALL_IN, LAUNCH_CALL_OUT)
    open(p, "w", encoding="utf-8").write(s)
    assert "if (skip_interior) {" in s
    assert s.count("int skip_interior") == 2, f"skip count: {s.count('int skip_interior')}"
    print("patched sommerfeld_rout_gpu.cu (skip_interior + boundary skip)")

    # ---- 3. create src/sommerfeld_rout_gpu_int.cu ----
    src = open(os.path.join(cand, "src", "sommerfeld_rout_gpu.cu"), encoding="utf-8").read()
    src = "#define SOMMERFELD_INTERIOR\n" + src
    # renames: non-forceinline helper + kernels + launches (order: routbam first)
    src = src.replace("is_sommerfeld_boundary", "is_sommerfeld_boundary_int")
    src = src.replace("gpu_sommerfeld_routbam_launch", "gpu_sommerfeld_routbam_launch_int")
    src = src.replace("sommerfeld_routbam_kernel", "sommerfeld_routbam_kernel_int")
    src = src.replace("gpu_sommerfeld_rout_launch", "gpu_sommerfeld_rout_launch_int")
    src = src.replace("sommerfeld_rout_kernel", "sommerfeld_rout_kernel_int")
    assert src.count("sommerfeld_rout_kernel_int") >= 2
    assert src.count("gpu_sommerfeld_rout_launch_int(") == 1
    assert src.count("gpu_sommerfeld_routbam_launch_int(") == 1
    assert src.count("is_sommerfeld_boundary_int") >= 3
    # CORRECTSTEP guard inside sommerfeld_rout_kernel_int (anchor uses _int names)
    assert src.count(BDY_CHECK_INT_ANCHOR) == 1, f"bdy check anchor: {src.count(BDY_CHECK_INT_ANCHOR)}"
    src = src.replace(BDY_CHECK_INT_ANCHOR, BDY_CHECK_INT)
    ip = os.path.join(cand, "src", "sommerfeld_rout_gpu_int.cu")
    open(ip, "w", encoding="utf-8").write(src)
    assert "if (precor == CORRECTSTEP) return;" in src
    print(f"wrote src/sommerfeld_rout_gpu_int.cu ({len(src)} bytes)")

    # ---- 4. sommerfeld_rout.h ----
    p = os.path.join(cand, "src", "sommerfeld_rout.h")
    s = open(p, encoding="utf-8").read()
    assert s.count(HDR_IN) == 1, f"header launch decl: {s.count(HDR_IN)}"
    s = s.replace(HDR_IN, HDR_OUT)
    open(p, "w", encoding="utf-8").write(s)
    assert "gpu_sommerfeld_rout_launch_int" in s
    print("patched sommerfeld_rout.h (launch decl + _int decl)")

    # ---- 5. bssn_step_gpu.C host call sites ----
    p = os.path.join(cand, "src", "bssn_step_gpu.C")
    s = open(p, encoding="utf-8").read()
    n0 = s.count("gpu_sommerfeld_rout_launch(")
    assert n0 == 2, f"bssn_step_gpu.C: {n0} call sites (expect 2)"
    s = split_som_calls(s)
    n1 = s.count("gpu_sommerfeld_rout_launch_int(")
    assert n1 == 2, f"int launch count: {n1}"
    open(p, "w", encoding="utf-8").write(s)
    print("patched bssn_step_gpu.C (2 call sites: skip=1 + int launch)")

    # ---- 6. CMakeLists.txt ----
    p = os.path.join(cand, "CMakeLists.txt")
    s = open(p, encoding="utf-8").read()
    old = "src/sommerfeld_rout_gpu.cu"
    assert s.count(old) == 1, f"CMake sommerfeld anchor: {s.count(old)}"
    s = s.replace(old, "src/sommerfeld_rout_gpu.cu src/sommerfeld_rout_gpu_int.cu")
    open(p, "w", encoding="utf-8").write(s)
    print("patched CMakeLists.txt (sommerfeld_rout_gpu_int.cu in ABEGPU_CUDA_SOURCES)")

    # ---- 7. verification ----
    print("\n=== verification ===")
    for f in ("sommerfeld_rout_gpu.cu", "sommerfeld_rout.h", "bssn_step_gpu.C", "fmisc.h"):
        print(f"  {f}: formal {formal_hash[f][:16]} -> cand {sha(os.path.join(cand, 'src', f))[:16]}")
    ip = os.path.join(cand, "src", "sommerfeld_rout_gpu_int.cu")
    print(f"  sommerfeld_rout_gpu_int.cu: {sha(ip)[:16]}")
    print(f"  CMakeLists.txt: formal {formal_hash['CMakeLists.txt'][:16]} -> cand {sha(os.path.join(cand, 'CMakeLists.txt'))[:16]}")
    print("PATCH OK")


if __name__ == "__main__":
    main()
