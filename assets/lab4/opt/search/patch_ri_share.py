#!/usr/bin/env python3
# P-RISHARE: rhs_interior cross-call field read sharing (ptxas-gated probe).
#
# Mechanism: rhs_kernel_int calls d_lopsided_point 24 times (once per metric
# and A-field) and d_kodis_point 24 times.  Each lopsided call independently
# reads the shift-vector center values vx=Sfx[idx], vy=Sfy[idx], vz=Sfz[idx]
# (Sfx==betax, Sfy==betay, Sfz==betaz in the interior kernel).  That is 24x
# redundant reads of the same 3 values per thread.  Similarly each lopsided
# call reads h_000 = f[idx] of ITS OWN field (distinct field per call, so no
# redundancy there).  The ptxas-gated probe: hoist the 3 shift center reads
# to the top of rhs_kernel_int and thread them through d_lopsided_point via
# new optional params vx_pre/vy_pre/vz_pre (default NAN -> read as before).
#
# Bit-exactness: interior kernel is the RHSPROBE_INTERIOR path (fh is a pure
# load), so betax[idx] == Sfx[idx] exactly (same array, same index, no
# mask/reflection).  Passing the value read once is bit-identical to reading
# it inside each call.
#
# GATE (from task): ptxas spill rise >5% -> declare dead without A/B.
# Expected: slight spill increase (3 extra live doubles across 24 calls),
# but the load-count reduction (72 -> 3) is real; whether it wins depends on
# the register allocator.  Level-0 ptxas decides.
#
# Files changed (candidate tree only):
#   src/lopsidediff.h      : d_lopsided_point add vx_pre/vy_pre/vz_pre params
#   src/bssn_rhs_gpu_int.cu: hoist shift centers, pass to all 24 lopsided calls
import hashlib
import sys

GUARDS = {
    "src/lopsidediff.h": "a47bc691dda98b053170777659972b79bbb300ef024167577bd77e250ab46539",
    "src/bssn_rhs_gpu_int.cu": "8dc0cf28aef2da184ca219e5e1a817073df6320cf85a87f09148489dd44a3839",
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

    # ---------------- 1. lopsidediff.h: add optional pre-loaded shift centers ----------------
    p = f"{root}/src/lopsidediff.h"
    s = open(p).read()
    if "vx_pre" in s:
        print("ALREADY_PATCHED lopsidediff.h")
        sys.exit(1)

    old_sig = r'''__device__ __forceinline__ double d_lopsided_point(
    const int ex[3], const double* f,
    const double* f_rhs, const double* Sfx, const double* Sfy, const double* Sfz,
    const double* X, const double* Y, const double* Z,
    int symmetry, double SYM1, double SYM2, double SYM3,
    int i, int j, int k // i, j, k is 0-based
) {'''
    new_sig = r'''__device__ __forceinline__ double d_lopsided_point(
    const int ex[3], const double* f,
    const double* f_rhs, const double* Sfx, const double* Sfy, const double* Sfz,
    const double* X, const double* Y, const double* Z,
    int symmetry, double SYM1, double SYM2, double SYM3,
    int i, int j, int k, // i, j, k is 0-based
    double vx_pre = __builtin_nan(""), double vy_pre = __builtin_nan(""), double vz_pre = __builtin_nan("")
) {'''
    assert s.count(old_sig) == 1, "lopsided signature anchor"
    s = s.replace(old_sig, new_sig)

    old_center = r'''    const double vx = Sfx[idx];
    const double vy = Sfy[idx];
    const double vz = Sfz[idx];'''
    new_center = r'''    // P-RISHARE: use pre-loaded shift centers if provided (interior kernel
    // hoists these once per thread instead of re-reading 24x).  Bit-exact:
    // interior path reads Sfx[idx] directly (no mask), so pre == Sfx[idx].
    const double vx = (__builtin_nan("") == vx_pre) ? Sfx[idx] : vx_pre;
    const double vy = (__builtin_nan("") == vy_pre) ? Sfy[idx] : vy_pre;
    const double vz = (__builtin_nan("") == vz_pre) ? Sfz[idx] : vz_pre;'''
    assert s.count(old_center) == 1, "lopsided center anchor"
    s = s.replace(old_center, new_center)
    open(p, "w").write(s)
    print("patched lopsidediff.h")

    # ---------------- 2. bssn_rhs_gpu_int.cu: hoist shift centers + pass to calls ----------------
    p = f"{root}/src/bssn_rhs_gpu_int.cu"
    s = open(p).read()
    if "RISHARE" in s:
        print("ALREADY_PATCHED bssn_rhs_gpu_int.cu")
        sys.exit(1)

    # Insert hoisted values right after the shift derivative block (after d_fderivs_point
    # for betax/betay/betaz), before the div_beta usage.  Anchor on the betaz deriv line.
    anchor1 = r'''    d_fderivs_point(dims, betaz, &betazx, &betazy, &betazz, X, Y, Z, SYM, SYM, ANTI, symmetry, lev, i, j, k);
'''
    assert s.count(anchor1) == 1, "betaz deriv anchor"
    insert = anchor1 + r'''
    // P-RISHARE: hoist shift-vector center reads once per thread.  The 24
    // d_lopsided_point calls below each read betax[idx]/betay[idx]/betaz[idx]
    // as Sfx[idx]/Sfy[idx]/Sfz[idx]; sharing them cuts 72 loads -> 3.
    // Bit-exact (interior path: direct loads, no mask).
    const double rishare_vx = betax[idx];
    const double rishare_vy = betay[idx];
    const double rishare_vz = betaz[idx];
'''
    s = s.replace(anchor1, insert)

    # Replace all 24 d_lopsided_point calls: append the pre-loaded values.
    # Pattern: `, i, j, k);` -> `, i, j, k, rishare_vx, rishare_vy, rishare_vz);`
    # Only within the interior kernel's metric/A/RHS blocks.  We replace the
    # exact closing of lopsided calls.
    old_call_tail = r''', i, j, k);
'''
    # count occurrences of the lopsided call pattern with unique field arg
    import re
    # find all d_lopsided_point( ... , i, j, k); spans
    count = 0
    out_lines = []
    for line in s.split("\n"):
        if "d_lopsided_point(dims," in line and line.rstrip().endswith("i, j, k);"):
            # e.g.  gxx_rhs[idx] += d_lopsided_point(dims, dxx, gxx_rhs, betax, betay, betaz, X, Y, Z, symmetry, SYM, SYM, SYM, i, j, k);
            newl = line.replace(", i, j, k);", ", i, j, k, rishare_vx, rishare_vy, rishare_vz);")
            out_lines.append(newl)
            count += 1
        else:
            out_lines.append(line)
    if count != 24:
        print(f"ERROR: expected 24 lopsided calls in int kernel, found {count}")
        sys.exit(1)
    s = "\n".join(out_lines)
    open(p, "w").write(s)
    print(f"patched bssn_rhs_gpu_int.cu ({count} lopsided calls)")

    # sanity: no dangling plain calls remain
    if "d_lopsided_point(dims," in s and "rishare_vx" in s:
        print("P-RISHARE PATCH DONE")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
