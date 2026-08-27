#!/usr/bin/env python3
# iter15 P6b PROBE — forceinline d_symmetry_bd_1b (+ f_at_1b) into fmisc.h
#
# Diagnostic: the deployed build uses CUDA_SEPARABLE_COMPILATION (-rdc=true),
# so d_symmetry_bd_1b (defined in fmisc_gpu.cu, declared non-inline in fmisc.h)
# is an ABI CALL from prolong3_kernel: 216 call/return round-trips per output
# point (SASS evidence from job 154433: CALL.ABS.NOINC 93 in base probe).
# Call/return fixed latency + return-stack serialization is the prime suspect
# for ncu3's "fixed-latency execution dependency (2.7 cyc, 31.3%)".
#
# This patch moves BOTH f_at_1b and d_symmetry_bd_1b into fmisc.h as
# __forceinline__ definitions (bodies VERBATIM from fmisc_gpu.cu), and removes
# the now-duplicate definitions from fmisc_gpu.cu. Prolong3 + sommerfeld both
# call d_symmetry_bd_1b -> both kernels get inlined calls. Bit-exact by
# construction (identical arithmetic, no re-association).
# PROBE ONLY: verify ptxas change; A/B requires main-agent authorization.
import sys, pathlib, hashlib

FMISC_H_OLD = """__device__ double d_symmetry_bd_1b(
	int ord, const int extc[3], const double* func,
	int i1b, int j1b, int k1b, const double SoA[3]
);
"""

FMISC_H_NEW = """__device__ __forceinline__ double f_at_1b(const double* f, const int ex[3], int i1b, int j1b, int k1b) {
	return f[((k1b - 1) * ex[1] + (j1b - 1)) * ex[0] + (i1b - 1)];
}

// P6b PROBE: moved from fmisc_gpu.cu + __forceinline__ (body verbatim) so the
// 216 call sites per prolong3 output point inline instead of ABI-calling.
__device__ __forceinline__ double d_symmetry_bd_1b(
	int ord, const int extc[3], const double* func,
	int i1b, int j1b, int k1b, const double SoA[3]
) {
	// out-of-range stays zero, matching funcc = 0.d0 initialization
	if (i1b < -ord + 1 || i1b > extc[0]) return 0.0;
	if (j1b < -ord + 1 || j1b > extc[1]) return 0.0;
	if (k1b < -ord + 1 || k1b > extc[2]) return 0.0;

	int ii = i1b, jj = j1b, kk = k1b;
	double factor = 1.0;

	// apply symmetry in x, then y, then z (same order as Fortran)
	if (ii <= 0) { ii = 1 - ii; factor *= SoA[0]; }
	if (jj <= 0) { jj = 1 - jj; factor *= SoA[1]; }
	if (kk <= 0) { kk = 1 - kk; factor *= SoA[2]; }

	if (ii < 1 || ii > extc[0]) return 0.0;
	if (jj < 1 || jj > extc[1]) return 0.0;
	if (kk < 1 || kk > extc[2]) return 0.0;

	return f_at_1b(func, extc, ii, jj, kk) * factor;
}
"""

F_AT_1B_OLD = """__device__ __forceinline__ double f_at_1b(const double* f, const int ex[3], int i1b, int j1b, int k1b) {
	return f[((k1b - 1) * ex[1] + (j1b - 1)) * ex[0] + (i1b - 1)];
}

"""

SYM_OLD = """__device__ double d_symmetry_bd_1b(
	int ord, const int extc[3], const double* func,
	int i1b, int j1b, int k1b, const double SoA[3]
) {
	// out-of-range stays zero, matching funcc = 0.d0 initialization
	if (i1b < -ord + 1 || i1b > extc[0]) return 0.0;
	if (j1b < -ord + 1 || j1b > extc[1]) return 0.0;
	if (k1b < -ord + 1 || k1b > extc[2]) return 0.0;

	int ii = i1b, jj = j1b, kk = k1b;
	double factor = 1.0;

	// apply symmetry in x, then y, then z (same order as Fortran)
	if (ii <= 0) { ii = 1 - ii; factor *= SoA[0]; }
	if (jj <= 0) { jj = 1 - jj; factor *= SoA[1]; }
	if (kk <= 0) { kk = 1 - kk; factor *= SoA[2]; }

	if (ii < 1 || ii > extc[0]) return 0.0;
	if (jj < 1 || jj > extc[1]) return 0.0;
	if (kk < 1 || kk > extc[2]) return 0.0;

	return f_at_1b(func, extc, ii, jj, kk) * factor;
}

"""

def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def main():
    cand = pathlib.Path(sys.argv[1])
    h = cand / "src/fmisc.h"
    g = cand / "src/fmisc_gpu.cu"
    sh, sg = sha(h), sha(g)
    print(f"[P6b] fmisc.h     hash: {sh}")
    print(f"[P6b] fmisc_gpu.cu hash: {sg}")
    if "P6b PROBE" in h.read_text():
        print("[P6b] ABORT: re-entrancy guard")
        sys.exit(1)
    hs = h.read_text()
    gs = g.read_text()
    if hs.count(FMISC_H_OLD) != 1:
        print(f"[P6b] ABORT: fmisc.h declaration count = {hs.count(FMISC_H_OLD)}")
        sys.exit(1)
    if gs.count(F_AT_1B_OLD) != 1 or gs.count(SYM_OLD) != 1:
        print(f"[P6b] ABORT: fmisc_gpu.cu def counts f_at={gs.count(F_AT_1B_OLD)} sym={gs.count(SYM_OLD)}")
        sys.exit(1)
    hs2 = hs.replace(FMISC_H_OLD, FMISC_H_NEW, 1)
    gs2 = gs.replace(F_AT_1B_OLD, "", 1).replace(SYM_OLD, "", 1)
    # sanity: remaining uses of d_symmetry_bd_1b in fmisc_gpu.cu (d_decide3d etc. use f_at_1b only)
    if "d_symmetry_bd_1b" in gs2:
        print("[P6b] ABORT: d_symmetry_bd_1b still referenced in fmisc_gpu.cu")
        sys.exit(1)
    if "f_at_1b" not in gs2:
        print("[P6b] WARN: f_at_1b not used in fmisc_gpu.cu anymore?")
    h.write_text(hs2)
    g.write_text(gs2)
    print(f"[P6b] APPLIED fmisc.h    -> {sha(h)}")
    print(f"[P6b] APPLIED fmisc_gpu.cu -> {sha(g)}")
    print("[P6b] DONE")

if __name__ == "__main__":
    main()
