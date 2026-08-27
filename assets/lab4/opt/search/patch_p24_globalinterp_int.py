#!/usr/bin/env python3
# Milestone C round 2: global_interp interior/boundary split (mirror of iter23
# prolong3 split). global_interp_kernel (fmisc_gpu.cu) evaluates each shell
# point via global_interp_device -> d_decide3d (masking) + d_polin3_1b (Neville).
#
#   - fmisc.h: d_decide3d gains a pure-load fast path guarded by
#     GLOBAL_INTERP_INTERIOR / SOMMERFELD_INTERIOR (dormant in base builds);
#     global_interp_device forward decl gains `bool` return + skip_interior.
#   - fmisc_gpu.cu: global_interp_device returns bool and gains skip_interior
#     (boundary-only mode: interior points return false -> kernel skips the
#     atomicAdd); global_interp_kernel / gpu_global_interp_launch gain
#     skip_interior; global_interp_amr_kernel passes skip=0 (unchanged).
#   - Creates src/fmisc_gpu_int.cu (slim TU): #define GLOBAL_INTERP_INTERIOR +
#     X_at_1b + global_interp_device_int + global_interp_kernel_int +
#     gpu_global_interp_launch_int (renamed to avoid -rdc duplicate symbols).
#   - MPatch.C / MPatch_gpu.cu (3 host call sites): original launch passes
#     skip_interior=1; int launch appended on the same stream. The atomicAdd
#     accumulators get exactly one contribution per point per block (from
#     exactly one of the two kernels) -> bit-exact by construction.
#   - CMakeLists.txt: add src/fmisc_gpu_int.cu.
#
# Interior definition (from d_decide3d semantics): the masked value equals a
# plain load iff every interpolation tap is in [1, ex] and non-reflective:
#     post-clamp cxB[m] >= 1 && cxT[m] <= ex[m]  (equiv cxI[m] in [3, ex-3]
#     for ORDN=6). Identical formulas in both kernels -> exact partition.
import hashlib, os, re, sys, shutil

FORMAL = os.environ.get("P24_FORMAL", os.path.expanduser("~/lab4-gpu"))
EXCLUDE_DIRS = {"evidence", "build", "__pycache__"}

# ---------- fmisc.h ----------
DEV_DECL_IN = """__device__ void global_interp_device(
	const int* ex, const double* X, const double* Y, const double* Z,
	const double* f, double* f_int,
	double x1, double y1, double z1,
	int ORDN, const double* SoA, int symmetry
);"""
DEV_DECL_OUT = """__device__ bool global_interp_device(
	const int* ex, const double* X, const double* Y, const double* Z,
	const double* f, double* f_int,
	double x1, double y1, double z1,
	int ORDN, const double* SoA, int symmetry, int skip_interior
);"""

DECIDE_START_IN = """	(void)fpi;
	(void)Symmetry;
	bool gont = false;"""
DECIDE_START_OUT = """#if defined(GLOBAL_INTERP_INTERIOR) || defined(SOMMERFELD_INTERIOR)
	// P24 interior fast path (fmisc_gpu_int / sommerfeld_rout_gpu_int TUs):
	// the caller's interior check guarantees every tap in [1, ex] (no
	// out-of-range, no reflection), so the range/reflection machinery reduces
	// to plain loads; values and ya layout are identical to the fmin1 loops of
	// the full path -> bit-exact.
	(void)fpi; (void)SoA; (void)Symmetry;
	auto idx = [&](int i, int j, int k) {
		return ((k - cxB[2]) * ordn + (j - cxB[1])) * ordn + (i - cxB[0]);
	};
	for (int k = cxB[2]; k <= cxT[2]; ++k)
		for (int j = cxB[1]; j <= cxT[1]; ++j)
			for (int i = cxB[0]; i <= cxT[0]; ++i)
				ya[idx(i, j, k)] = f_at_1b(f, ex, i, j, k);
	return false;
#else
	(void)fpi;
	(void)Symmetry;
	bool gont = false;"""

DECIDE_END_IN = """			for (int i = fmin2[0]; i <= fmax2[0]; ++i)
				ya[idx(i, j, k)] = f_at_1b(f, ex, 1 - i, 1 - j, 1 - k) * SoA[0] * SoA[1] * SoA[2];
		}
	}
	return false;
}"""
DECIDE_END_OUT = """			for (int i = fmin2[0]; i <= fmax2[0]; ++i)
				ya[idx(i, j, k)] = f_at_1b(f, ex, 1 - i, 1 - j, 1 - k) * SoA[0] * SoA[1] * SoA[2];
		}
	}
	return false;
#endif
}"""

GI_LAUNCH_DECL_IN = """void gpu_global_interp_launch(
	cudaStream_t stream,
    int NN, int DIM,
    double* d_XX_0, double* d_XX_1, double* d_XX_2,
    int shape_0, int shape_1, int shape_2,
    double* d_X_0, double* d_X_1, double* d_X_2,
    double* d_field,
    double llb_0, double llb_1, double llb_2,
    double uub_0, double uub_1, double uub_2,
    double DH_0, double DH_1, double DH_2,
    int ordn, double SoA_0, double SoA_1, double SoA_2,
    int Symmetry, int var_idx, int num_var,
    double* d_shellf, int* d_weight
);"""
GI_LAUNCH_DECL_OUT = """void gpu_global_interp_launch(
	cudaStream_t stream,
    int NN, int DIM,
    double* d_XX_0, double* d_XX_1, double* d_XX_2,
    int shape_0, int shape_1, int shape_2,
    double* d_X_0, double* d_X_1, double* d_X_2,
    double* d_field,
    double llb_0, double llb_1, double llb_2,
    double uub_0, double uub_1, double uub_2,
    double DH_0, double DH_1, double DH_2,
    int ordn, double SoA_0, double SoA_1, double SoA_2,
    int Symmetry, int var_idx, int num_var,
    double* d_shellf, int* d_weight, int skip_interior
);

void gpu_global_interp_launch_int(
	cudaStream_t stream,
    int NN, int DIM,
    double* d_XX_0, double* d_XX_1, double* d_XX_2,
    int shape_0, int shape_1, int shape_2,
    double* d_X_0, double* d_X_1, double* d_X_2,
    double* d_field,
    double llb_0, double llb_1, double llb_2,
    double uub_0, double uub_1, double uub_2,
    double DH_0, double DH_1, double DH_2,
    int ordn, double SoA_0, double SoA_1, double SoA_2,
    int Symmetry, int var_idx, int num_var,
    double* d_shellf, int* d_weight, int skip_interior
);"""

# ---------- fmisc_gpu.cu ----------
DEV_DEF_IN = """__device__ void global_interp_device(
	const int* ex, const double* X, const double* Y, const double* Z,
	const double* f, double* f_int,
	double x1, double y1, double z1,
	int ORDN, const double* SoA, int symmetry
) {"""
DEV_DEF_OUT = """__device__ bool global_interp_device(
	const int* ex, const double* X, const double* Y, const double* Z,
	const double* f, double* f_int,
	double x1, double y1, double z1,
	int ORDN, const double* SoA, int symmetry, int skip_interior
) {"""

CLAMP_ANCHOR = """	for (int m = 0; m < 3; ++m) {
		if (cxB[m] < cmin[m]) { cxB[m] = cmin[m]; cxT[m] = cxB[m] + ORDN - 1; }
		if (cxT[m] > cmax[m]) { cxT[m] = cmax[m]; cxB[m] = cxT[m] + 1 - ORDN; }
	}

	double cx[3];"""
INTERIOR_CHECK = """	for (int m = 0; m < 3; ++m) {
		if (cxB[m] < cmin[m]) { cxB[m] = cmin[m]; cxT[m] = cxB[m] + ORDN - 1; }
		if (cxT[m] > cmax[m]) { cxT[m] = cmax[m]; cxB[m] = cxT[m] + 1 - ORDN; }
	}

#ifdef GLOBAL_INTERP_INTERIOR
	// interior-only fast path: all taps strictly inside [1, ex] and
	// non-reflective -> d_decide3d masks identity -> pure-load variant in
	// fmisc.h is bit-exact for these points.
	if (!(cxB[0] >= 1 && cxB[1] >= 1 && cxB[2] >= 1 &&
	      cxT[0] <= ex[0] && cxT[1] <= ex[1] && cxT[2] <= ex[2])) return false;
#else
	// boundary-only mode (skip_interior=1): interior points are computed by
	// global_interp_kernel_int in fmisc_gpu_int.cu; skip them here (deployed
	// behavior at skip_interior=0: condition never matches).
	if (skip_interior) {
		if (cxB[0] >= 1 && cxB[1] >= 1 && cxB[2] >= 1 &&
		    cxT[0] <= ex[0] && cxT[1] <= ex[1] && cxT[2] <= ex[2]) return false;
	}
#endif

	double cx[3];"""

DEV_END_IN = """	double ddy = 0.0;
	d_polin3_1b(x1a, x1a, x1a, ya, cx[0], cx[1], cx[2], f_int[0], ddy, ORDN);
}"""
DEV_END_OUT = """	double ddy = 0.0;
	d_polin3_1b(x1a, x1a, x1a, ya, cx[0], cx[1], cx[2], f_int[0], ddy, ORDN);
	return true;
}"""

KERN_CALL_IN = """    double val = 0.0;
    global_interp_device(
        ex, d_X_arr[0], d_X_arr[1], d_X_arr[2],
        d_field, &val,
        px, py, pz,
        ordn, SoA_arr, Symmetry
    );"""
KERN_CALL_OUT = """    double val = 0.0;
    if (!global_interp_device(
            ex, d_X_arr[0], d_X_arr[1], d_X_arr[2],
            d_field, &val,
            px, py, pz,
            ordn, SoA_arr, Symmetry, skip_interior
    )) return;"""

KERN_SIG_IN = """    int ordn, double SoA_0, double SoA_1, double SoA_2, int Symmetry,
    int var_idx, int num_var,
    double* d_shellf, int* d_weight
) {"""
KERN_SIG_OUT = """    int ordn, double SoA_0, double SoA_1, double SoA_2, int Symmetry,
    int var_idx, int num_var,
    double* d_shellf, int* d_weight, int skip_interior
) {"""

LAUNCH_SIG_IN = """    int ordn, double SoA_0, double SoA_1, double SoA_2,
    int Symmetry, int var_idx, int num_var,
    double* d_shellf, int* d_weight
) {"""
LAUNCH_SIG_OUT = """    int ordn, double SoA_0, double SoA_1, double SoA_2,
    int Symmetry, int var_idx, int num_var,
    double* d_shellf, int* d_weight, int skip_interior
) {"""

LAUNCH_CALL_IN = """	global_interp_kernel<<<gridSize, blockSize, 0, stream>>>(
		NN, DIM,
        d_XX_0, d_XX_1, d_XX_2,
        shape_0, shape_1, shape_2, d_X_0, d_X_1, d_X_2, d_field,
        llb_0, llb_1, llb_2, uub_0, uub_1, uub_2,
        DH_0, DH_1, DH_2,
        ordn, SoA_0, SoA_1, SoA_2, Symmetry,
        var_idx, num_var, d_shellf, d_weight
	);"""
LAUNCH_CALL_OUT = """	global_interp_kernel<<<gridSize, blockSize, 0, stream>>>(
		NN, DIM,
        d_XX_0, d_XX_1, d_XX_2,
        shape_0, shape_1, shape_2, d_X_0, d_X_1, d_X_2, d_field,
        llb_0, llb_1, llb_2, uub_0, uub_1, uub_2,
        DH_0, DH_1, DH_2,
        ordn, SoA_0, SoA_1, SoA_2, Symmetry,
        var_idx, num_var, d_shellf, d_weight, skip_interior
	);"""

AMR_CALL_IN = """    double val = 0.0;
    // 调用现有的设备端插值核心
    global_interp_device(
        ex, d_X_arr[0], d_X_arr[1], d_X_arr[2],
        d_field, &val,
        px, py, pz,
        ordn, SoA_arr, Symmetry
    );"""
AMR_CALL_OUT = """    double val = 0.0;
    // 调用现有的设备端插值核心 (skip=0: 部署行为)
    global_interp_device(
        ex, d_X_arr[0], d_X_arr[1], d_X_arr[2],
        d_field, &val,
        px, py, pz,
        ordn, SoA_arr, Symmetry, 0
    );"""

CALL_RE = re.compile(
    r'(gpu_global_interp_launch\([\s\S]*?Symmetry, k, num_var, [A-Za-z_][A-Za-z0-9_]*, [A-Za-z_][A-Za-z0-9_]*\s*\);)')
TAIL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*\);$')


def split_gi_calls(s):
    def repl(m):
        call = m.group(1)
        bnd = TAIL_RE.sub(r'\1, 1);', call)
        intl = TAIL_RE.sub(r'\1, 0);', call.replace('gpu_global_interp_launch(', 'gpu_global_interp_launch_int(', 1))
        return bnd + "\n" + intl
    return CALL_RE.sub(repl, s)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    if len(sys.argv) != 2:
        print("usage: patch_p24_globalinterp_int.py <candidate_root>"); sys.exit(2)
    cand = sys.argv[1]
    if os.path.exists(cand):
        print(f"ERROR: {cand} exists"); sys.exit(3)

    # ---- 0. hash-guard formal baseline ----
    formal_hash = {}
    for f in ("fmisc.h", "fmisc_gpu.cu", "MPatch.C", "MPatch_gpu.cu", "bssn_step_gpu.C",
              "Parallel_GPU.cpp"):
        formal_hash[f] = sha(os.path.join(FORMAL, "src", f))
    formal_hash["CMakeLists.txt"] = sha(os.path.join(FORMAL, "CMakeLists.txt"))
    print("formal src hashes:")
    for f, h in formal_hash.items():
        print(f"  {f} {h[:16]}")
    assert formal_hash["fmisc.h"].startswith("d5c4f4a2"), "fmisc.h drifted"
    assert formal_hash["fmisc_gpu.cu"].startswith("c7cc9f07"), "fmisc_gpu.cu drifted"
    assert formal_hash["MPatch.C"].startswith("0dee0202"), "MPatch.C drifted"
    assert formal_hash["MPatch_gpu.cu"].startswith("8474c997"), "MPatch_gpu.cu drifted"
    assert formal_hash["bssn_step_gpu.C"].startswith("2bd3b2d9"), "bssn_step_gpu.C drifted"
    assert formal_hash["Parallel_GPU.cpp"].startswith("1f7b2617"), "Parallel_GPU.cpp drifted"

    # ---- 1. copy tree ----
    shutil.copytree(FORMAL, cand, ignore=shutil.ignore_patterns(*EXCLUDE_DIRS))
    print(f"copied formal -> {cand}")

    # ---- 2. fmisc.h ----
    p = os.path.join(cand, "src", "fmisc.h")
    s = open(p, encoding="utf-8").read()
    assert s.count(DEV_DECL_IN) == 1, f"dev decl anchor: {s.count(DEV_DECL_IN)}"
    assert s.count(DECIDE_START_IN) == 1, f"decide start anchor: {s.count(DECIDE_START_IN)}"
    assert s.count(DECIDE_END_IN) == 1, f"decide end anchor: {s.count(DECIDE_END_IN)}"
    assert s.count(GI_LAUNCH_DECL_IN) == 1, f"launch decl anchor: {s.count(GI_LAUNCH_DECL_IN)}"
    s = s.replace(DEV_DECL_IN, DEV_DECL_OUT)
    s = s.replace(DECIDE_START_IN, DECIDE_START_OUT)
    s = s.replace(DECIDE_END_IN, DECIDE_END_OUT)
    s = s.replace(GI_LAUNCH_DECL_IN, GI_LAUNCH_DECL_OUT)
    open(p, "w", encoding="utf-8").write(s)
    assert s.count("GLOBAL_INTERP_INTERIOR") >= 1, "fmisc.h guard count"
    assert "gpu_global_interp_launch_int" in s
    print("patched fmisc.h (d_decide3d pure-load guard + device/launch decls)")

    # ---- 3. fmisc_gpu.cu ----
    p = os.path.join(cand, "src", "fmisc_gpu.cu")
    s = open(p, encoding="utf-8").read()
    assert s.count(DEV_DEF_IN) == 1, f"dev def anchor: {s.count(DEV_DEF_IN)}"
    assert s.count(CLAMP_ANCHOR) == 1, f"clamp anchor: {s.count(CLAMP_ANCHOR)}"
    assert s.count(DEV_END_IN) == 1, f"dev end anchor: {s.count(DEV_END_IN)}"
    assert s.count(KERN_CALL_IN) == 1, f"kernel call anchor: {s.count(KERN_CALL_IN)}"
    assert s.count(KERN_SIG_IN) == 1, f"kernel sig anchor: {s.count(KERN_SIG_IN)}"
    assert s.count(LAUNCH_SIG_IN) == 1, f"launch sig anchor: {s.count(LAUNCH_SIG_IN)}"
    assert s.count(LAUNCH_CALL_IN) == 1, f"launch call anchor: {s.count(LAUNCH_CALL_IN)}"
    assert s.count(AMR_CALL_IN) == 1, f"amr call anchor: {s.count(AMR_CALL_IN)}"
    s = s.replace(DEV_DEF_IN, DEV_DEF_OUT)
    s = s.replace(CLAMP_ANCHOR, INTERIOR_CHECK)
    s = s.replace(DEV_END_IN, DEV_END_OUT)
    s = s.replace(KERN_CALL_IN, KERN_CALL_OUT)
    s = s.replace(KERN_SIG_IN, KERN_SIG_OUT)
    s = s.replace(LAUNCH_SIG_IN, LAUNCH_SIG_OUT)
    s = s.replace(LAUNCH_CALL_IN, LAUNCH_CALL_OUT)
    s = s.replace(AMR_CALL_IN, AMR_CALL_OUT)
    open(p, "w", encoding="utf-8").write(s)
    assert "if (!global_interp_device(" in s
    assert "skip_interior" in s
    assert "global_interp_device(\n        ex, d_X_arr[0], d_X_arr[1], d_X_arr[2]," in s
    # bool return type: the two bare `return;` in global_interp_device must be
    # `return false;` (nvcc #117-D warning otherwise).
    assert s.count("if (ORDN > MAX_ORDN) { f_int[0] = NAN; gpu_stop(); return; }") == 1
    assert s.count("\t\tf_int[0] = NAN;\n\t\tgpu_stop();\n\t\treturn;") == 1
    s = s.replace("if (ORDN > MAX_ORDN) { f_int[0] = NAN; gpu_stop(); return; }",
                  "if (ORDN > MAX_ORDN) { f_int[0] = NAN; gpu_stop(); return false; }")
    s = s.replace("\t\tf_int[0] = NAN;\n\t\tgpu_stop();\n\t\treturn;",
                  "\t\tf_int[0] = NAN;\n\t\tgpu_stop();\n\t\treturn false;")
    open(p, "w", encoding="utf-8").write(s)
    print("patched fmisc_gpu.cu (device bool + skip_interior + kernel guard + return false)")

    # ---- 4. create slim src/fmisc_gpu_int.cu ----
    # Extract: includes + MAX_ORDN + X_at_1b + global_interp_device + kernel + launch
    start = s.index("#include <cuda_runtime.h>")
    dev_end = s.index("__global__ void lowerboundset_kernel(")
    kern_start = s.index("__global__ void global_interp_kernel(")
    kern_end = s.index("__forceinline__ __device__ double warpReduceSum(")
    head = s[start:dev_end]
    kblk = s[kern_start:kern_end]
    int_src = "#define GLOBAL_INTERP_INTERIOR\n" + head + kblk
    # renames (global_interp_device defined above + used in kernel)
    int_src = int_src.replace("global_interp_device", "global_interp_device_int")
    int_src = int_src.replace("global_interp_kernel", "global_interp_kernel_int")
    int_src = int_src.replace("gpu_global_interp_launch", "gpu_global_interp_launch_int")
    ip = os.path.join(cand, "src", "fmisc_gpu_int.cu")
    open(ip, "w", encoding="utf-8").write(int_src)
    assert int_src.count("global_interp_device_int(") >= 2
    assert "global_interp_kernel_int<<<" in int_src or "global_interp_kernel_int<<<" in int_src
    assert "gpu_global_interp_launch_int(" in int_src
    print(f"wrote src/fmisc_gpu_int.cu ({len(int_src)} bytes)")

    # ---- 5. MPatch.C + MPatch_gpu.cu + Parallel_GPU.cpp host call sites ----
    for f in ("MPatch.C", "MPatch_gpu.cu", "Parallel_GPU.cpp"):
        p = os.path.join(cand, "src", f)
        s = open(p, encoding="utf-8").read()
        n0 = s.count("gpu_global_interp_launch(")
        assert n0 >= 1, f"{f}: no call site"
        s = split_gi_calls(s)
        n1 = s.count("gpu_global_interp_launch_int(")
        assert n1 == n0, f"{f}: int launch count {n1} != {n0}"
        open(p, "w", encoding="utf-8").write(s)
        print(f"patched {f} ({n0} call sites -> {n1} int launches)")

    # ---- 6. CMakeLists.txt ----
    p = os.path.join(cand, "CMakeLists.txt")
    s = open(p, encoding="utf-8").read()
    old = "src/fmisc_gpu.cu"
    assert s.count(old) == 1, f"CMake fmisc anchor: {s.count(old)}"
    s = s.replace(old, "src/fmisc_gpu.cu src/fmisc_gpu_int.cu")
    open(p, "w", encoding="utf-8").write(s)
    print("patched CMakeLists.txt (fmisc_gpu_int.cu in ABEGPU_CUDA_SOURCES)")

    # ---- 7. verification ----
    print("\n=== verification ===")
    for f in ("fmisc.h", "fmisc_gpu.cu", "MPatch.C", "MPatch_gpu.cu", "bssn_step_gpu.C",
              "Parallel_GPU.cpp"):
        print(f"  {f}: formal {formal_hash[f][:16]} -> cand {sha(os.path.join(cand, 'src', f))[:16]}")
    ip = os.path.join(cand, "src", "fmisc_gpu_int.cu")
    print(f"  fmisc_gpu_int.cu: {sha(ip)[:16]}")
    print(f"  CMakeLists.txt: formal {formal_hash['CMakeLists.txt'][:16]} -> cand {sha(os.path.join(cand, 'CMakeLists.txt'))[:16]}")
    g = open(os.path.join(cand, "src", "Parallel_GPU.cpp"), encoding="utf-8").read()
    assert g.count("gpu_global_interp_launch_int(") == 1, "Parallel_GPU.cpp int launch != 1"
    gi = open(os.path.join(cand, "src", "fmisc_gpu_int.cu"), encoding="utf-8").read()
    assert gi.count("return false;") >= 2, "int TU return false count"
    print("PATCH OK")


if __name__ == "__main__":
    main()
