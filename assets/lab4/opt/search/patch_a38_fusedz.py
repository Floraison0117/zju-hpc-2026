#!/usr/bin/env python3
"""Iter38 A38-1: fused-z global interpolation (eliminate ya[6^3] local array).

Mechanism: ncu on the deployed state shows global_interp_multi_kernel (the 800
MassPAng big launches = 37.2s of the 42.97s analysis) is L2-pipe-bound at 91%
(lts__throughput) with only 31% data sectors; the dominant stall is long_scoreboard
(17.05 cyc, 72% of CPI). The ya[6^3]=1728B local array (fill + read) generates
~2.9GB of scattered LDL/STL traffic per launch (2.7x the 1.08GB global load volume).
This patch fuses d_decide3d's tap load loop with d_polin3_1b's z-polint: each
(i,j) column loads its 6 z-taps directly into yqtmp and polints immediately, so
the ya[216] materialization disappears (only yatmp[36] remains).

Bit-exact by construction: the per-tap value formula (direct vs 1-idx reflection
per dim, factor multiply order SoA[0] then SoA[1] then SoA[2]) replicates
d_decide3d exactly; the polint chains consume identical values in identical order.
sommerfeld keeps its separate d_decide3d/d_polin3_1b calls (unchanged).

Files changed:
  src/fmisc.h       (add d_gi_fused next to d_polin3_1b)
  src/fmisc_gpu.cu  (global_interp_device uses d_gi_fused)
"""
import hashlib, sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."

def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]

ENCF = dict(encoding="utf-8")

H = f"{ROOT}/src/fmisc.h"
CU = f"{ROOT}/src/fmisc_gpu.cu"

cur_h = sha(H)
cur_cu = sha(CU)
print(f"guard fmisc.h={cur_h} fmisc_gpu.cu={cur_cu}")
assert cur_h == "abec6936", f"fmisc.h hash drift {cur_h}"
assert cur_cu == "d8684f83", f"fmisc_gpu.cu hash drift {cur_cu}"

# ---- 1. fmisc.h: add d_gi_fused after d_polin3_1b ----
s = open(H, **ENCF).read()
assert "d_gi_fused" not in s, "reentry guard"
anchor = "__device__ __forceinline__ void d_polin3_1b("
assert s.count(anchor) == 1
# find the end of d_polin3_1b body (the closing brace after its last polint call)
tail_marker = "\tpolint(x1a, ymtmp, x1, y, dy, ordn);\n}"
assert s.count(tail_marker) == 1, "d_polin3_1b tail not found"
fused = r"""
// A38-1: fused (d_decide3d + d_polin3_1b). Eliminates the ya[6^3] local array
// (1728B write+read per (point,var)) by consuming each (i,j) column's 6 z-taps
// immediately. Bit-exact: per-tap formula (direct vs 1-idx reflection per dim,
// factor order SoA[0], SoA[1], SoA[2]) and polint consumption order replicate
// d_decide3d + d_polin3_1b exactly. sommerfeld keeps the original pair.
__device__ __forceinline__ void d_gi_fused(
	const int ex[3], const double* f, const int cxB[3], const int cxT[3],
	const double SoA[3], const double cx[3], const double x1a[MAX_ORDN],
	int ordn, double& y, double& dy
) {
	int fmin1[3], fmin2[3], fmax1[3], fmax2[3];
	bool gont = false;
	for (int m = 0; m < 3; ++m) {
		if (!(abs(cxB[m]) >= 0)) gont = true;
		if (!(abs(cxT[m]) >= 0)) gont = true;
		fmin1[m] = max(1, cxB[m]);
		fmax1[m] = cxT[m];
		fmin2[m] = cxB[m];
		fmax2[m] = min(0, cxT[m]);
		if ((fmin1[m] <= fmax1[m]) && (fmin1[m] < 1 || fmax1[m] > ex[m])) gont = true;
		if ((fmin2[m] <= fmax2[m]) && (1 - fmax2[m] < 1 || 1 - fmin2[m] > ex[m])) gont = true;
	}
	if (gont) { y = NAN; dy = NAN; gpu_stop(); return; }

	double yatmp[MAX_ORDN * MAX_ORDN];
	double ymtmp[MAX_ORDN];
	double yntmp[MAX_ORDN];
	double yqtmp[MAX_ORDN];

	for (int i = 0; i < ordn; ++i) {
		for (int j = 0; j < ordn; ++j) {
			for (int k = 0; k < ordn; ++k) {
				int i_abs = cxB[0] + i;
				int j_abs = cxB[1] + j;
				int k_abs = cxB[2] + k;
				bool ir = (i_abs >= fmin2[0] && i_abs <= fmax2[0]);
				bool jr = (j_abs >= fmin2[1] && j_abs <= fmax2[1]);
				bool kr = (k_abs >= fmin2[2] && k_abs <= fmax2[2]);
				int ii = ir ? 1 - i_abs : i_abs;
				int jj = jr ? 1 - j_abs : j_abs;
				int kk = kr ? 1 - k_abs : k_abs;
				double tap = f_at_1b(f, ex, ii, jj, kk);
				if (ir) tap *= SoA[0];
				if (jr) tap *= SoA[1];
				if (kr) tap *= SoA[2];
				yqtmp[k] = tap;
			}
			polint(x1a, yqtmp, cx[2], yatmp[j * ordn + i], dy, ordn);
		}
		for (int j = 0; j < ordn; ++j) yntmp[j] = yatmp[j * ordn + i];
		polint(x1a, yntmp, cx[1], ymtmp[i], dy, ordn);
	}
	polint(x1a, ymtmp, cx[0], y, dy, ordn);
}
"""
s = s.replace(tail_marker, tail_marker + fused)
open(H, "w", **ENCF).write(s)
print("fmisc.h patched:", sha(H))

# ---- 2. fmisc_gpu.cu: global_interp_device uses d_gi_fused ----
s = open(CU, **ENCF).read()
assert "d_gi_fused" not in s, "reentry guard"
old_tail = """	double ya[MAX_ORDN * MAX_ORDN * MAX_ORDN];
	if (d_decide3d(ex, f, f, cxB, cxT, SoA, ya, ORDN, symmetry)) {
#if GPU_DEBUG_PRINT
        printf("global_interp position: %f %f %f\\n", x1, y1, z1);
        printf("data range: %f %f %f %f %f %f\\n",
               X_at_1b(X, 1), X_at_1b(X, ex[0]),
               X_at_1b(Y, 1), X_at_1b(Y, ex[1]),
               X_at_1b(Z, 1), X_at_1b(Z, ex[2]));
#endif
		f_int[0] = NAN;
		gpu_stop();
		return;
	}

	double ddy = 0.0;
	d_polin3_1b(x1a, x1a, x1a, ya, cx[0], cx[1], cx[2], f_int[0], ddy, ORDN);
}"""
assert s.count(old_tail) == 1, "global_interp_device tail not found"
new_tail = """	double ddy = 0.0;
	d_gi_fused(ex, f, cxB, cxT, SoA, cx, x1a, ORDN, f_int[0], ddy);
}"""
s = s.replace(old_tail, new_tail)
open(CU, "w", **ENCF).write(s)
print("fmisc_gpu.cu patched:", sha(CU))
print("DONE")
