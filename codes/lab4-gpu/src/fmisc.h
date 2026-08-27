
#ifndef FMISC_H
#define FMISC_H

#ifdef fortran1
#define f_interp_2 interp_2
#define f_pointcopy pointcopy
#define f_copy copy
#define f_global_interp global_interp
#define f_global_interp_ss global_interp_ss
#define f_global_interp_ss_2d global_interp_ss_2d
#define f_global_interpind global_interpind
#define f_global_interpind2d global_interpind2d
#define f_global_interpind1d global_interpind1d
#define f_l2normhelper l2normhelper
#define f_l2normhelper_sh l2normhelper_sh
#define f_l2normhelper_sh_rms l2normhelper_sh_rms
#define f_average average
#define f_average3 average3
#define f_average2 average2
#define f_average2p average2p
#define f_average2m average2m
#define f_lowerboundset lowerboundset
#define f_set_value set_value
#define f_add_value add_value
#define f_array_add array_add
#define f_array_copy array_copy
#define f_array_subtract array_subtract
#define f_fft four1
#define f_find_maximum find_maximum
#define f_polint polint
#define f_d2dump d2dump
#endif
#ifdef fortran2
#define f_interp_2 INTERP_2
#define f_pointcopy POINTCOPY
#define f_copy COPY
#define f_global_interp GLOBAL_INTERP
#define f_global_interp_ss GLOBAL_INTERP_SS
#define f_global_interp_ss_2d GLOBAL_INTERP_SS_2D
#define f_global_interpind GLOBAL_INTERPIND
#define f_global_interpind2d GLOBAL_INTERPIND2D
#define f_global_interpind1d GLOBAL_INTERPIND1D
#define f_l2normhelper L2NORMHELPER
#define f_l2normhelper_sh L2NORMHELPER_SH
#define f_l2normhelper_sh_rms L2NORMHELPER_SH_RMS
#define f_average AVERAGE
#define f_average3 AVERAGE3
#define f_average2 AVERAGE2
#define f_average2p AVERAGE2P
#define f_average2m AVERAGE2M
#define f_lowerboundset LOWERBOUNDSET
#define f_set_value SET_VALU
#define f_add_value ADD_VALUE
#define f_array_add ARRAY_ADD
#define f_array_copy ARRAY_COPY
#define f_array_subtract ARRAY_SUBTRACT
#define f_fft FOUR1
#define f_find_maximum FIND_MAXIMUM
#define f_polint POLINT
#define f_d2dump D2DUMP
#endif
#ifdef fortran3
#define f_interp_2 interp_2_
#define f_pointcopy pointcopy_
#define f_copy copy_
#define f_global_interp global_interp_
#define f_global_interp_ss global_interp_ss_
#define f_global_interp_ss_2d global_interp_ss_2d_
#define f_global_interpind global_interpind_
#define f_global_interpind2d global_interpind2d_
#define f_global_interpind1d global_interpind1d_
#define f_l2normhelper l2normhelper_
#define f_l2normhelper_sh l2normhelper_sh_
#define f_l2normhelper_sh_rms l2normhelper_sh_rms_
#define f_average average_
#define f_average3 average3_
#define f_average2 average2_
#define f_average2p average2p_
#define f_average2m average2m_
#define f_lowerboundset lowerboundset_
#define f_set_value set_value_
#define f_add_value add_value_
#define f_array_add array_add_
#define f_array_copy array_copy_
#define f_array_subtract array_subtract_
#define f_fft four1_
#define f_find_maximum find_maximum_
#define f_polint polint_
#define f_d2dump d2dump_
#endif

extern "C"
{
	void f_pointcopy(int &,
					 double *, double *, int *, double *,
					 double &, double &, double &, double &);
}

extern "C"
{
	void f_copy(int &,
				double *, double *, int *, double *,
				double *, double *, int *, double *,
				double *, double *);
}

extern "C"
{
	void f_global_interp(int *, double *, double *, double *,
						 double *, double &,
						 double &, double &, double &,
						 int &, double *, int &);
}

extern "C"
{
	void f_global_interp_ss(int *, double *, double *, double *,
							double *, double &,
							double &, double &, double &,
							int &, double *, int &, int &);
}

extern "C"
{
	void f_global_interp_ss_2d(int *, double *, double *, int &,
							   double *, double &,
							   double &, double &,
							   int &, double *, int &, int &);
}

extern "C"
{
	void f_global_interpind(int *, double *, double *, double *,
							double *, double &,
							double &, double &, double &,
							int &, double *, int &,
							int *, double *, int &);
}

extern "C"
{
	void f_global_interpind2d(int *, double *, double *, double *,
							  double *, double &,
							  double &, double &, double &,
							  int &, double *, int &,
							  int *, double *, int &);
}

extern "C"
{
	void f_global_interpind1d(int *, double *, double *, double *,
							  double *, double &,
							  double &, double &, double &,
							  int &, double *, int &,
							  int *, double *, int &, int &);
}

extern "C"
{
	void f_l2normhelper(int *, double *, double *, double *,
						double &, double &, double &,
						double &, double &, double &,
						double *, double &, int &);
}

extern "C"
{
	void f_l2normhelper_sh(int *, double *, double *, double *,
						   double &, double &, double &,
						   double &, double &, double &,
						   double *, double &, int &, int &, int &);
}

extern "C"
{
	void f_l2normhelper_sh_rms(int *, double *, double *, double *,
							   double &, double &, double &,
							   double &, double &, double &,
							   double *, double &, int &, int &, int &, int &);
}

extern "C"
{
	void f_average(int *, double *, double *, double *);
}

extern "C"
{
	void f_average3(int *, double *, double *, double *);
}

extern "C"
{
	void f_average2(int *, double *, double *, double *, double *);
}

extern "C"
{
	void f_average2p(int *, double *, double *, double *, double *);
}

extern "C"
{
	void f_average2m(int *, double *, double *, double *, double *);
}

extern "C"
{
	void f_lowerboundset(int *, double *, double &);
}

extern "C"
{
	void f_set_value(int *, double *, double &);
}
extern "C"
{
	void f_add_value(int *, double *, double &);
}
extern "C"
{
	void f_array_add(int *, double *, double *);
}
extern "C"
{
	void f_array_copy(int *, double *, double *);
}
extern "C"
{
	void f_array_subtract(int *, double *, double *);
}

extern "C"
{
	void f_fft(double *, int &, int &);
}

extern "C"
{
	void f_find_maximum(int *,
						double *, double *, double *, double *,
						double &, double *, int *, int *);
}

extern "C"
{
	void f_polint(double *, double *, double &, double &, double &, int &);
}

extern "C"
{
	void f_d2dump(int &, double *, double *, int *, double *, double *, int &, double *);
}

#ifdef USE_GPU
#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
// P8: helper bodies in this header use bare max/min/abs/fabs (CUDA device
// builtins). Host TU passes need explicit std names.
#if !defined(__CUDA_ARCH__)
using std::max;
using std::min;
#endif

__device__ void global_interp_device(
	const int* ex, const double* X, const double* Y, const double* Z,
	const double* f, double* f_int,
	double x1, double y1, double z1,
	int ORDN, const double* SoA, int symmetry
);

__device__ __forceinline__ double f_at_1b(const double* f, const int ex[3], int i1b, int j1b, int k1b) {
	return f[((k1b - 1) * ex[1] + (j1b - 1)) * ex[0] + (i1b - 1)];
}

// P6b PROBE: moved from fmisc_gpu.cu + __forceinline__ (body verbatim) so the
// 216 call sites per prolong3 output point inline instead of ABI-calling.
__device__ __forceinline__ double d_symmetry_bd_1b(
	int ord, const int extc[3], const double* func,
	int i1b, int j1b, int k1b, const double SoA[3]
) {
#ifdef PROLONG3_INTERIOR
	// P23 interior fast path: active only in the prolongrestrict_cell_gpu_int
	// TU. At interior prolong3 points all 216 coarse stencil taps are strictly
	// inside [1, extc] and non-reflective, so the range/reflection masks are
	// identity and factor==1.0 (IEEE x*1.0==x) -> plain load is bit-exact.
	return f_at_1b(func, extc, i1b, j1b, k1b);
#else
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
#endif
}

#ifndef MAX_ORDN
#define MAX_ORDN 6
#endif
#ifndef GPU_DEBUG_PRINT
#define GPU_DEBUG_PRINT 0
#endif

__device__ __forceinline__ void gpu_stop() {
#if GPU_STRICT_STOP
    asm("trap;");
#endif
}

// P8 PROBE: moved from fmisc_gpu.cu + __forceinline__ (bodies verbatim) so
// sommerfeld_rout_kernel + global_interp_device's d_decide3d/d_polin3_1b/
// polint call sites inline instead of ABI-calling (P6b pattern, iter15 -13.6%).
__device__ __forceinline__ void polint(const double* xa, const double* ya, double x, double& y, double& dy, int ordn) {
	double c[MAX_ORDN], d[MAX_ORDN], den[MAX_ORDN], ho[MAX_ORDN];
	int ns = 1;
	double dif = fabs(x - xa[0]);
	for (int m = 0; m < ordn; ++m) {
		c[m] = ya[m];
		d[m] = ya[m];
		ho[m] = xa[m] - x;
		double dift = fabs(x - xa[m]);
		if (dift < dif) { ns = m + 1; dif = dift; }
	}
	y = ya[ns - 1];
	ns = ns - 1;
	for (int m = 1; m < ordn; ++m) {
		for (int i = 0; i < ordn - m; ++i) {
			den[i] = ho[i] - ho[i + m];
			if (den[i] == 0.0) {
#if GPU_DEBUG_PRINT
                printf("failure in polint for point %f\n", x);
                printf("with input points: ");
                for (int t = 0; t < ordn; ++t) printf("%f ", xa[t]);
                printf("\n");
#endif
				y = NAN; dy = NAN; gpu_stop(); return;
			}
			den[i] = (c[i + 1] - d[i]) / den[i];
			d[i] = ho[i + m] * den[i];
			c[i] = ho[i] * den[i];
		}
		if (2 * ns < (ordn - m)) {
			dy = c[ns];
		} else {
			dy = d[ns - 1];
			ns = ns - 1;
		}
		y = y + dy;
	}
}

__device__ __forceinline__ void d_polin3_1b(
	const double* x1a, const double* x2a, const double* x3a,
	const double* ya, double x1, double x2, double x3,
	double& y, double& dy, int ordn
) {
	double yatmp[MAX_ORDN * MAX_ORDN];
	double ymtmp[MAX_ORDN];
	double yntmp[MAX_ORDN];
	double yqtmp[MAX_ORDN];

	for (int i = 0; i < ordn; ++i) {
		for (int j = 0; j < ordn; ++j) {
			for (int k = 0; k < ordn; ++k) {
				yqtmp[k] = ya[(k * ordn + j) * ordn + i];
			}
			polint(x3a, yqtmp, x3, yatmp[j * ordn + i], dy, ordn);
		}
		for (int j = 0; j < ordn; ++j) yntmp[j] = yatmp[j * ordn + i];
		polint(x2a, yntmp, x2, ymtmp[i], dy, ordn);
	}
	polint(x1a, ymtmp, x1, y, dy, ordn);
}
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


__device__ __forceinline__ bool d_decide3d(
	const int ex[3], const double* f, const double* fpi,
	const int cxB[3], const int cxT[3], const double SoA[3],
	double* ya, int ordn, int Symmetry
) {
	(void)fpi;
	(void)Symmetry;
	bool gont = false;
	int fmin1[3], fmin2[3], fmax1[3], fmax2[3];

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
	if (gont) {
#if GPU_DEBUG_PRINT
        printf("error in decide3d\n");
        printf("cxB: %d %d %d, cxT: %d %d %d, ex: %d %d %d\n",
               cxB[0], cxB[1], cxB[2], cxT[0], cxT[1], cxT[2], ex[0], ex[1], ex[2]);
        printf("fmin1: %d %d %d, fmax1: %d %d %d\n",
               fmin1[0], fmin1[1], fmin1[2], fmax1[0], fmax1[1], fmax1[2]);
        printf("fmin2: %d %d %d, fmax2: %d %d %d\n",
               fmin2[0], fmin2[1], fmin2[2], fmax2[0], fmax2[1], fmax2[2]);
#endif
		return true;
	}

	auto idx = [&](int i, int j, int k) {
		return ((k - cxB[2]) * ordn + (j - cxB[1])) * ordn + (i - cxB[0]);
	};

	for (int k = fmin1[2]; k <= fmax1[2]; ++k) {
		for (int j = fmin1[1]; j <= fmax1[1]; ++j) {
			for (int i = fmin1[0]; i <= fmax1[0]; ++i)
				ya[idx(i, j, k)] = f_at_1b(f, ex, i, j, k);
			for (int i = fmin2[0]; i <= fmax2[0]; ++i)
				ya[idx(i, j, k)] = f_at_1b(f, ex, 1 - i, j, k) * SoA[0];
		}
		for (int j = fmin2[1]; j <= fmax2[1]; ++j) {
			for (int i = fmin1[0]; i <= fmax1[0]; ++i)
				ya[idx(i, j, k)] = f_at_1b(f, ex, i, 1 - j, k) * SoA[1];
			for (int i = fmin2[0]; i <= fmax2[0]; ++i)
				ya[idx(i, j, k)] = f_at_1b(f, ex, 1 - i, 1 - j, k) * SoA[0] * SoA[1];
		}
	}

	for (int k = fmin2[2]; k <= fmax2[2]; ++k) {
		for (int j = fmin1[1]; j <= fmax1[1]; ++j) {
			for (int i = fmin1[0]; i <= fmax1[0]; ++i)
				ya[idx(i, j, k)] = f_at_1b(f, ex, i, j, 1 - k) * SoA[2];
			for (int i = fmin2[0]; i <= fmax2[0]; ++i)
				ya[idx(i, j, k)] = f_at_1b(f, ex, 1 - i, j, 1 - k) * SoA[0] * SoA[2];
		}
		for (int j = fmin2[1]; j <= fmax2[1]; ++j) {
			for (int i = fmin1[0]; i <= fmax1[0]; ++i)
				ya[idx(i, j, k)] = f_at_1b(f, ex, i, 1 - j, 1 - k) * SoA[1] * SoA[2];
			for (int i = fmin2[0]; i <= fmax2[0]; ++i)
				ya[idx(i, j, k)] = f_at_1b(f, ex, 1 - i, 1 - j, 1 - k) * SoA[0] * SoA[1] * SoA[2];
		}
	}
	return false;
}

void gpu_lowerboundset_launch(
    cudaStream_t &stream,
    int ex[3],
    double* d_chi0, double TINNY
);

void gpu_pack_launch(
	cudaStream_t stream, const double* d_src_3d, double* d_dst_1d,
	int src_nx, int src_ny, int dst_nx, int dst_ny, int dst_nz,
	int off_x, int off_y, int off_z
);

void gpu_unpack_launch(
	cudaStream_t stream, const double* d_src_1d, double* d_dst_3d,
	int dst_nx, int dst_ny, int src_nx, int src_ny, int src_nz,
	int off_x, int off_y, int off_z
);

void gpu_average_launch(cudaStream_t stream, const int ext[3], const double* d_f1, const double* d_f2, double* d_fout);
void gpu_average3_launch(cudaStream_t stream, const int ext[3], const double* d_f1, const double* d_f2, double* d_fout);
void gpu_average2_launch(cudaStream_t stream, const int ext[3], const double* d_f1, const double* d_f2, const double* d_f3, double* d_fout);
void gpu_average2p_launch(cudaStream_t stream, const int ext[3], const double* d_f1, const double* d_f2, const double* d_f3, double* d_fout);
void gpu_average2m_launch(cudaStream_t stream, const int ext[3], const double* d_f1, const double* d_f2, const double* d_f3, double* d_fout);

void gpu_global_interp_launch(
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
);

void gpu_global_interp_multi_launch(
	cudaStream_t stream,
    int NN, int DIM,
    double* d_XX_0, double* d_XX_1, double* d_XX_2,
    int shape_0, int shape_1, int shape_2,
    double* d_X_0, double* d_X_1, double* d_X_2,
    double* const* d_fields, const double* d_SoA_all,
    double llb_0, double llb_1, double llb_2,
    double uub_0, double uub_1, double uub_2,
    double DH_0, double DH_1, double DH_2,
    int ordn, int Symmetry, int num_var,
    double* d_shellf, int* d_weight
);

void gpu_l2normhelper_launch(
	cudaStream_t stream, 
	const int* ex, 
	const double* X, const double* Y, const double* Z,
	double xmin, double ymin, double zmin,
	double xmax, double ymax, double zmax,
	const double* d_f, double& f_out, int gw
);

void gpu_global_interp_amr_launch(
    cudaStream_t stream,
    int active_count, int DIM,
    int* d_active_indices,
    double* d_XX_0, double* d_XX_1, double* d_XX_2,
    int shape_0, int shape_1, int shape_2,
    double* d_X_0, double* d_X_1, double* d_X_2,
    double* d_field,
    int ordn, double SoA_0, double SoA_1, double SoA_2,
    int Symmetry, int var_idx, int num_var,
    double* d_shellf
);
#endif

#endif /* FMISC_H */
