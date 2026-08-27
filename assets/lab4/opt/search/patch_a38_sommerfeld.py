#!/usr/bin/env python3
"""Iter38 A38-2: apply d_gi_fused to sommerfeld interpolation call sites.

Stacked on A38-1 (global_interp fused-z already landed in fmisc.h). The
sommerfeld_rout/routbam kernels still use d_decide3d + d_polin3_1b with their
own ya[6^3] local array. Same mechanism (eliminate the ya materialization);
bit-exact by construction (d_gi_fused replicates the exact per-tap values and
polint order; sommerfeld's clamps already handle the reflections, and
d_gi_fused's fmin1/fmin2 logic is idempotent on those clamped ranges).

Files changed:
  src/sommerfeld_rout_gpu.cu  (2 call sites -> d_gi_fused)
"""
import hashlib, sys, re

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."

def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]

ENCF = dict(encoding="utf-8")
CU = f"{ROOT}/src/sommerfeld_rout_gpu.cu"

cur = sha(CU)
print(f"guard sommerfeld_rout_gpu.cu={cur}")
assert cur == "e33e3fed", f"sommerfeld_rout_gpu.cu hash drift {cur}"

s = open(CU, **ENCF).read()
assert s.count("d_gi_fused") == 0, "reentry guard"

old = """        double ya[ORDN * ORDN * ORDN];
        d_decide3d(ext, f0, f0, cxB, cxT, SoA, ya, ORDN, Symmetry);
        double ddy;
        double r_interp;
        double xa[ORDN];
        for(int m = 0; m < ORDN; ++ m) xa[m] = (double)m;
        
        d_polin3_1b(xa, xa, xa, ya, cx[0], cx[1], cx[2], r_interp, ddy, ORDN);"""
new = """        double ddy;
        double r_interp;
        double xa[ORDN];
        for(int m = 0; m < ORDN; ++ m) xa[m] = (double)m;

        d_gi_fused(ext, f0, cxB, cxT, SoA, cx, xa, ORDN, r_interp, ddy);"""
n = s.count(old)
print(f"call sites matched: {n}")
assert n == 2, f"expected 2 call sites, got {n}"
s = s.replace(old, new)
open(CU, "w", **ENCF).write(s)
print("sommerfeld_rout_gpu.cu patched:", sha(CU))
print("DONE")
