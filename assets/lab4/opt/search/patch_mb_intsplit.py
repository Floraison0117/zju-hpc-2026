#!/usr/bin/env python3
# Milestone B L1 decisive test: real interior/boundary RHS kernel split.
#   - Creates a fresh candidate tree (cp -r ~/lab4-gpu, minus evidence/build).
#   - Patches derivatives.h / lopsidediff.h / kodiss.h with RHSPROBE_INTERIOR
#     plain-load fast path (same insertion as patch_intprobe.py, bit-exact for
#     interior points).
#   - Creates src/bssn_rhs_gpu_int.cu = verbatim rhs_kernel + launch copy,
#     renamed rhs_kernel_int / gpu_compute_rhs_bssn_launch_int, with
#     #define RHSPROBE_INTERIOR before includes and an interior-only early return.
#   - bssn_rhs_gpu.cu: rhs_kernel + gpu_compute_rhs_bssn_launch gain a trailing
#     `int skip_interior` parameter (0 = deployed behavior).
#   - bssn_rhs.h: skip_interior in original declaration + int-version declaration.
#   - 5 host call sites (bssn_gpu_class.C:2843/2952/3210, bssn_step_gpu.C:92/202):
#     original launch passes skip_interior=1; int launch appended with same args.
#   - CMakeLists.txt: add src/bssn_rhs_gpu_int.cu to ABEGPU CUDA sources.
#
# Correctness-critical deviation from the milestone-B plan (documented):
#   Plan §3.5 says margin-2 interior (k>=2). Under the deployed config
#   (Symmetry=1 equatorial, z-bbox [0,320] => Z[0]=0 => kmin=-3 in
#   lopsidediff.h/kodiss.h), lopsided/kodis reach k-3 at k=2 and REFLECT
#   (fh(i,j,-1) -> f[z=0]*SoA[2]); the RHSPROBE_INTERIOR plain load would read
#   a negative index -> OOB (UB / illegal-memory-access risk, BR_ORD lesson).
#   Hence k lower margin is 3 (k in [3, ex2-3]); i/j margins stay 2
#   (imin=jmin=0 -> no negative stencil access reachable at i,j>=2).
#   Union of {interior} and {boundary shell} still covers the full domain
#   exactly once; bit-exactness argument unchanged for all covered points.
import hashlib, os, re, sys, shutil

FORMAL = os.path.expanduser("~/lab4-gpu")
EXCLUDE_DIRS = {"evidence", "build", "__pycache__"}
LAMBDA_RE = re.compile(r"const auto fh = \[&\]\(int ii, int jj, int kk\) -> double \{")
INS = """#ifdef RHSPROBE_INTERIOR
        // interior fast path: plain load, no in_range/valid/reflection masks
        // (bit-exact for interior points: masks are identity there, P2-verified)
        return f[((kk) * ex[1] + (jj)) * ex[0] + (ii)];
#endif
"""

BOUNDS_CHECK = """    // 越界检查
    if (i >= ex0 || j >= ex1 || k >= ex2) return;

    int idx = IDX3D(i, j, k, ex0, ex1, ex2);
"""

INT_EARLY = """    // 越界检查
    if (i >= ex0 || j >= ex1 || k >= ex2) return;

    // interior-only fast path: skip the boundary shell.
    // margins: i,j = 2 (imin=jmin=0 -> no reflection reachable at i,j>=2);
    // k lower = 3 (equatorial symmetry -> kmin=-3 -> lopsided/kodis reflect at
    // k<=2, plain-load fh would read OOB); k upper = 2.
    if (i < 2 || i > ex0 - 3 || j < 2 || j > ex1 - 3 || k < 3 || k > ex2 - 3) return;

    int idx = IDX3D(i, j, k, ex0, ex1, ex2);
"""

SKIP_INTERIOR = """    // 越界检查
    if (i >= ex0 || j >= ex1 || k >= ex2) return;

    // boundary-only mode (skip_interior=1): interior points are computed by
    // rhs_kernel_int in bssn_rhs_gpu_int.cu; skip them here (deployed
    // behavior when skip_interior=0: condition never matches, no code change).
    if (skip_interior) { if (i >= 2 && i < ex0 - 2 && j >= 2 && j < ex1 - 2 && k >= 3 && k < ex2 - 2) return; }

    int idx = IDX3D(i, j, k, ex0, ex1, ex2);
"""


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    if len(sys.argv) != 2:
        print("usage: patch_mb_intsplit.py <candidate_root>"); sys.exit(2)
    cand = sys.argv[1]
    if os.path.exists(cand):
        print(f"ERROR: {cand} exists"); sys.exit(3)

    # ---- 0. hash-guard formal baseline ----
    formal_hash = {f: sha(os.path.join(FORMAL, "src", f)) for f in
                   ("bssn_rhs_gpu.cu", "derivatives.h", "lopsidediff.h", "kodiss.h", "bssn_rhs.h",
                    "bssn_gpu_class.C", "bssn_step_gpu.C")}
    formal_hash["CMakeLists.txt"] = sha(os.path.join(FORMAL, "CMakeLists.txt"))
    print("formal src hashes:")
    for f, h in formal_hash.items():
        print(f"  {f} {h[:16]}")
    assert formal_hash["bssn_rhs_gpu.cu"].startswith("4a4b2aab"), "bssn_rhs_gpu.cu drifted"
    assert formal_hash["derivatives.h"].startswith("3ede4646"), "derivatives.h drifted"
    assert formal_hash["lopsidediff.h"].startswith("5dddaa75"), "lopsidediff.h drifted"
    assert formal_hash["kodiss.h"].startswith("e88f9632"), "kodiss.h drifted"

    # ---- 1. copy tree (exclude heavy dirs) ----
    shutil.copytree(FORMAL, cand, ignore=shutil.ignore_patterns(*EXCLUDE_DIRS))
    # keep a buildable tree: golden/scripts/GW250118/Input files are copied.
    print(f"copied formal -> {cand}")

    # ---- 2. header RHSPROBE_INTERIOR patch ----
    for fn in ("derivatives.h", "lopsidediff.h", "kodiss.h"):
        p = os.path.join(cand, "src", fn)
        s = open(p).read()
        if "#ifdef RHSPROBE_INTERIOR" in s:
            print(f"  {fn}: SKIP (already patched)")
        else:
            s2, n = LAMBDA_RE.subn(lambda m: m.group(0) + "\n" + INS, s)
            assert (n == 2 if fn == "derivatives.h" else n == 1), f"{fn}: {n} lambdas"
            open(p, "w").write(s2)
            print(f"  {fn}: patched {n} lambda(s)")

    # ---- 3. create src/bssn_rhs_gpu_int.cu ----
    src = open(os.path.join(FORMAL, "src", "bssn_rhs_gpu.cu")).read()
    src = "#define RHSPROBE_INTERIOR\n" + src
    src = src.replace("gpu_compute_rhs_bssn_launch", "gpu_compute_rhs_bssn_launch_int")
    src = src.replace("rhs_kernel", "rhs_kernel_int")
    n = src.count(BOUNDS_CHECK)
    assert n == 1, f"bounds-check block occurrences: {n}"
    src = src.replace(BOUNDS_CHECK, INT_EARLY)
    ip = os.path.join(cand, "src", "bssn_rhs_gpu_int.cu")
    open(ip, "w").write(src)
    print(f"wrote src/bssn_rhs_gpu_int.cu ({len(src)} bytes)")
    assert "if (i < 2 || i > ex0 - 3 || j < 2 || j > ex1 - 3 || k < 3 || k > ex2 - 3) return;" in src

    # ---- 4. bssn_rhs_gpu.cu: skip_interior param + boundary-mode early return ----
    p = os.path.join(cand, "src", "bssn_rhs_gpu.cu")
    s = open(p).read()
    k_sig = "    double* Gmx_Res, double* Gmy_Res, double* Gmz_Res,\n    int symmetry, int lev, double eps, int co\n) {"
    l_sig = "    double* d_Gmx_Res, double* d_Gmy_Res, double* d_Gmz_Res,\n    int symmetry, int lev, double eps, int co\n) {"
    assert s.count(k_sig) == 1, f"kernel sig occurrences: {s.count(k_sig)}"
    assert s.count(l_sig) == 1, f"launch sig occurrences: {s.count(l_sig)}"
    s = s.replace(k_sig, "    double* Gmx_Res, double* Gmy_Res, double* Gmz_Res,\n    int symmetry, int lev, double eps, int co, int skip_interior\n) {")
    s = s.replace(l_sig, "    double* d_Gmx_Res, double* d_Gmy_Res, double* d_Gmz_Res,\n    int symmetry, int lev, double eps, int co, int skip_interior\n) {")
    assert s.count(BOUNDS_CHECK) == 1, "bounds-check not unique in bssn_rhs_gpu.cu"
    s = s.replace(BOUNDS_CHECK, SKIP_INTERIOR)
    call_end = "        symmetry, lev, eps, co\n    );"
    assert s.count(call_end) == 1, f"kernel call end occurrences: {s.count(call_end)}"
    s = s.replace(call_end, "        symmetry, lev, eps, co, skip_interior\n    );")
    open(p, "w").write(s)
    print("patched bssn_rhs_gpu.cu (skip_interior param + boundary-mode early return)")

    # ---- 5. bssn_rhs.h ----
    p = os.path.join(cand, "src", "bssn_rhs.h")
    s = open(p).read()
    decl_re = re.compile(
        r"void gpu_compute_rhs_bssn_launch\( // launch kernel with device pointers\n(.*?)\n    int symmetry, int lev, double eps, int co\n\);",
        re.S)
    m = decl_re.search(s)
    assert m, "launch declaration not found in bssn_rhs.h"
    body = m.group(1)
    orig = (f"void gpu_compute_rhs_bssn_launch( // launch kernel with device pointers\n{body}\n"
            f"    int symmetry, int lev, double eps, int co, int skip_interior\n);")
    intdecl = (f"\nvoid gpu_compute_rhs_bssn_launch_int( // launch interior-specialized rhs kernel\n{body}\n"
               f"    int symmetry, int lev, double eps, int co\n);")
    s = s[:m.start()] + orig + intdecl + s[m.end():]
    open(p, "w").write(s)
    print("patched bssn_rhs.h (skip_interior + int declaration)")

    # ---- 6. host call sites (5) ----
    call_re = re.compile(
        r"gpu_compute_rhs_bssn_launch\(\n(.*?)\n(\s*)Symmetry, lev, ndeps, (pre|cor)\n(\s*)\);", re.S)
    for fn in ("bssn_gpu_class.C", "bssn_step_gpu.C"):
        p = os.path.join(cand, "src", fn)
        s = open(p).read()
        ms = list(call_re.finditer(s))
        assert len(ms) == (3 if fn == "bssn_gpu_class.C" else 2), f"{fn}: {len(ms)} call sites"
        out = []
        last = 0
        for m in ms:
            out.append(s[last:m.start()])
            args, ind, endarg, closeind = m.group(1), m.group(2), m.group(3), m.group(4)
            out.append(f"gpu_compute_rhs_bssn_launch(\n{args}\n{ind}Symmetry, lev, ndeps, {endarg}, 1\n{closeind});\n")
            out.append(f"{closeind}gpu_compute_rhs_bssn_launch_int(\n{args}\n{ind}Symmetry, lev, ndeps, {endarg}\n{closeind});")
            last = m.end()
        out.append(s[last:])
        open(p, "w").write("".join(out))
        print(f"patched {fn}: {len(ms)} call sites -> skip_interior=1 + int launch appended")

    # ---- 7. CMakeLists.txt ----
    p = os.path.join(cand, "CMakeLists.txt")
    s = open(p).read()
    old = "src/bssn_rhs_gpu.cu src/prolongrestrict_cell_gpu.cu"
    assert s.count(old) == 1
    s = s.replace(old, "src/bssn_rhs_gpu.cu src/bssn_rhs_gpu_int.cu src/prolongrestrict_cell_gpu.cu")
    open(p, "w").write(s)
    print("patched CMakeLists.txt (bssn_rhs_gpu_int.cu in ABEGPU_CUDA_SOURCES)")

    # ---- 8. verification ----
    print("\n=== verification ===")
    cand_hash = {f: sha(os.path.join(cand, "src", f)) if f != "CMakeLists.txt" else sha(os.path.join(cand, f)) for f in formal_hash}
    for f in ("bssn_rhs_gpu.cu", "bssn_rhs.h", "bssn_gpu_class.C", "bssn_step_gpu.C"):
        print(f"  {f}: formal {formal_hash[f][:16]} -> cand {cand_hash[f][:16]}")
    for f in ("derivatives.h", "lopsidediff.h", "kodiss.h"):
        print(f"  {f}: cand {cand_hash[f][:16]} (RHSPROBE_INTERIOR patched)")
    ip = os.path.join(cand, "src", "bssn_rhs_gpu_int.cu")
    print(f"  bssn_rhs_gpu_int.cu: {sha(ip)[:16]}")
    n_int = open(ip).read().count("rhs_kernel_int")
    print(f"  bssn_rhs_gpu_int.cu: 'rhs_kernel_int' occurrences={n_int}")
    # grep checks
    g = open(os.path.join(cand, "src", "bssn_gpu_class.C")).read() + open(os.path.join(cand, "src", "bssn_step_gpu.C")).read()
    assert g.count("gpu_compute_rhs_bssn_launch_int(") == 5, "int launch count != 5"
    assert g.count("ndeps, pre, 1") + g.count("ndeps, cor, 1") == 5, "skip_interior=1 count != 5"
    print("  5 call sites: int launch appended + skip_interior=1 verified")
    print("PATCH OK")


if __name__ == "__main__":
    main()
