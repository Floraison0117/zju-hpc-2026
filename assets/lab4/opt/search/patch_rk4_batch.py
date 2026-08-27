#!/usr/bin/env python3
# P-RK4: RK4 cross-variable batching (发射结构杠杆).
#
# Mechanism: rungekutta4_rout_kernel is launched once per variable in a
# while-loop over StateList (24 vars), per (block, lev, RK4 substep) =
# 912,576 launches total (24 vars x 38,024 sites), 19.76s module (3.2%).
# Each launch is a tiny elementwise kernel (16 regs, 83.6% occ,
# memory-latency-bound, stall_long_scoreboard 65.4).  All 24 variables
# share the same shape ex[0..2], same dT_lev, same iter_count.
#
# Fix: one kernel per (block, lev, substep) covering all 24 variables:
# 2-D grid (ceil(n/256), num_var); blockIdx.y = var_idx.  Each thread
# (idx, v) performs the identical elementwise operation as the per-variable
# launch's thread idx did.  Bit-exact by construction (same arithmetic,
# same arrays, per-variable disjoint).
#
# Ordering: the surrounding while-loop interleaves per-variable
#   bam(var)  [lev==0]  ->  rk4(var)  ->  rout(var)  [lev>0]
# Restructured to 3 passes preserving per-variable order:
#   pass A: all bam (lev==0) + gather pointers
#   pass B: one batched rk4 (all vars)
#   pass C: all rout (lev>0)
# Cross-variable independence (distinct sgfn buffers) => bit-exact.
#
# Files changed:
#   src/rungekutta4_rout_gpu.cu : add rungekutta4_batch_kernel + launcher
#   src/rungekutta4_rout.h      : declare launcher
#   src/bssn_step_gpu.C         : restructure both call sites (predictor
#                                 line ~167, corrector line ~316) to 3 passes
import hashlib
import sys

GUARDS = {
    "src/rungekutta4_rout_gpu.cu": "bc1d3d6c814c88005385c196e8f5ab4fe4c5bc088ec89a460437e040e488b651",
    "src/rungekutta4_rout.h": "10a05092ede81319b9fffdf2fa2a93df0bfe5306cbb3dac8932d672e77beb3b9",
    "src/bssn_step_gpu.C": "2bd3b2d9bd8ffe6c4fa3f9250ca6e08c60685bf0d60670493dbe7f58fab3c548",
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

    # ---------------- 1. rungekutta4_rout_gpu.cu: batch kernel + launcher ----------------
    p = f"{root}/src/rungekutta4_rout_gpu.cu"
    s = open(p).read()
    if "rungekutta4_batch_kernel" in s:
        print("ALREADY_PATCHED rungekutta4_rout_gpu.cu")
        sys.exit(1)

    batch = r'''#define RK4_MAXVARS 24
struct RK4Batch {
    const double* f0[RK4_MAXVARS];
    double* f1[RK4_MAXVARS];
    double* f_rhs[RK4_MAXVARS];
};

// P-RK4: one launch covering all variables (grid.y = var_idx).
// Thread (idx, v) performs the identical elementwise operation that the
// per-variable launch's thread idx performed on variable v's arrays.
// Bit-exact by construction: same loads, same arithmetic, same stores.
__global__ void rungekutta4_batch_kernel(
    int n, int num_var, RK4Batch b, double dT, int RK4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y;
    if (idx >= n || v >= num_var) return;

    const double* f0 = b.f0[v];
    double* f1 = b.f1[v];
    double* f_rhs = b.f_rhs[v];

    double f0_val = f0[idx];
    double f1_val = f1[idx];
    double f_rhs_val = f_rhs[idx];

    if (RK4 == 0) {
        f1[idx] = f0_val + 0.5 * dT * f_rhs_val;
    } else if (RK4 == 1) {
        f_rhs[idx] = f_rhs_val + 2.0 * f1_val;
        f1[idx]    = f0_val + 0.5 * dT * f1_val;
    } else if (RK4 == 2) {
        f_rhs[idx] = f_rhs_val + 2.0 * f1_val;
        f1[idx]    = f0_val + dT * f1_val;
    } else if (RK4 == 3) {
        f1[idx] = f0_val + (1.0 / 6.0) * dT * (f1_val + f_rhs_val);
    }
}

void gpu_rungekutta4_batch_launch(
    cudaStream_t &stream,
    int ex[3], double dT,
    const double** h_f0s, double** h_f1s, double** h_rhss, int num_var, int RK4
) {
    int n = ex[0] * ex[1] * ex[2];
    if (num_var <= 0 || num_var > RK4_MAXVARS) {
        // fall back: launch per variable via the original launcher
        for (int v = 0; v < num_var; ++v) {
            rungekutta4_rout_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
                n, h_f0s[v], h_f1s[v], h_rhss[v], dT, RK4
            );
        }
        return;
    }

    RK4Batch b;
    for (int v = 0; v < num_var; ++v) {
        b.f0[v] = h_f0s[v];
        b.f1[v] = h_f1s[v];
        b.f_rhs[v] = h_rhss[v];
    }

    int block = 256;
    int grid_x = (n + block - 1) / block;
    dim3 grid3(grid_x, num_var);

    rungekutta4_batch_kernel<<<grid3, block, 0, stream>>>(
        n, num_var, b, dT, RK4
    );
}

'''
    # insert before the original launcher
    anchor = "void gpu_rungekutta4_rout_launch("
    assert s.count(anchor) == 1, "rungekutta4_rout_gpu.cu anchor not unique"
    s = s.replace(anchor, batch + anchor)
    open(p, "w").write(s)
    print("patched rungekutta4_rout_gpu.cu")

    # ---------------- 2. rungekutta4_rout.h: declaration ----------------
    p = f"{root}/src/rungekutta4_rout.h"
    s = open(p).read()
    if "gpu_rungekutta4_batch_launch" in s:
        print("ALREADY_PATCHED rungekutta4_rout.h")
        sys.exit(1)
    decl = r'''void gpu_rungekutta4_batch_launch(
    cudaStream_t &stream,
    int ex[3], double dT,
    const double** h_f0s, double** h_f1s, double** h_rhss, int num_var, int RK4
);
'''
    anchor = "void gpu_rungekutta4_rout_launch("
    assert s.count(anchor) == 1, "rungekutta4_rout.h anchor not unique"
    s = s.replace(anchor, decl + anchor)
    open(p, "w").write(s)
    print("patched rungekutta4_rout.h")

    # ---------------- 3. bssn_step_gpu.C: predictor loop (site 1) ----------------
    p = f"{root}/src/bssn_step_gpu.C"
    s = open(p).read()
    if "gpu_rungekutta4_batch_launch" in s:
        print("ALREADY_PATCHED bssn_step_gpu.C")
        sys.exit(1)

    # ---- 3a. predictor site (uses varl0=StateList, varl=SynchList_pre, varlrhs=RHSList) ----
    # exact current text (tabs preserved from source)
    old1 = r'''				MyList<var> *varl0 = StateList, *varl = SynchList_pre, *varlrhs = RHSList; // we do not check the correspondence here
				while (varl0) {
					if (lev == 0) { // sommerfeld indeed
						gpu_sommerfeld_routbam_launch(
							cg->stream,
							cg->shape,
							cg->d_X[0], cg->d_X[1], cg->d_X[2],
							Pp->data->bbox[0], Pp->data->bbox[1], Pp->data->bbox[2], Pp->data->bbox[3], Pp->data->bbox[4], Pp->data->bbox[5],
							cg->d_fgfs[varlrhs->data->sgfn],
							cg->d_fgfs[varl0->data->sgfn], varl0->data->propspeed, varl0->data->SoA,
							Symmetry
						);
					}
					gpu_rungekutta4_rout_launch(
						cg->stream,
						cg->shape, dT_lev, 
						cg->d_fgfs[varl0->data->sgfn], cg->d_fgfs[varl->data->sgfn], cg->d_fgfs[varlrhs->data->sgfn], iter_count
					);
					if (lev > 0) {// fix BD point
						gpu_sommerfeld_rout_launch(
							cg->stream,
							cg->shape,
							cg->d_X[0], cg->d_X[1], cg->d_X[2],
							Pp->data->bbox[0], Pp->data->bbox[1], Pp->data->bbox[2], Pp->data->bbox[3], Pp->data->bbox[4], Pp->data->bbox[5],
							dT_lev, cg->d_fgfs[phi0->sgfn],
							cg->d_fgfs[Lap0->sgfn], cg->d_fgfs[varl0->data->sgfn], cg->d_fgfs[varl->data->sgfn], varl0->data->SoA,
							Symmetry, cor
						);
					}

					varl0 = varl0->next;
					varl = varl->next;
					varlrhs = varlrhs->next;
				}'''
    new1 = r'''				MyList<var> *varl0 = StateList, *varl = SynchList_pre, *varlrhs = RHSList; // we do not check the correspondence here
				// P-RK4: 3-pass restructure (bam all -> batched rk4 -> rout all).
				// Per-variable order bam_i -> rk4_i -> rout_i preserved; cross-
				// variable arrays are disjoint (distinct sgfn) => bit-exact.
				{
					const double* h_f0s[RK4_MAXVARS];
					double* h_f1s[RK4_MAXVARS];
					double* h_rhss[RK4_MAXVARS];
					int nv = 0;
					while (varl0) {
						if (lev == 0) { // sommerfeld indeed
							gpu_sommerfeld_routbam_launch(
								cg->stream,
								cg->shape,
								cg->d_X[0], cg->d_X[1], cg->d_X[2],
								Pp->data->bbox[0], Pp->data->bbox[1], Pp->data->bbox[2], Pp->data->bbox[3], Pp->data->bbox[4], Pp->data->bbox[5],
								cg->d_fgfs[varlrhs->data->sgfn],
								cg->d_fgfs[varl0->data->sgfn], varl0->data->propspeed, varl0->data->SoA,
								Symmetry
							);
						}
						h_f0s[nv] = cg->d_fgfs[varl0->data->sgfn];
						h_f1s[nv] = cg->d_fgfs[varl->data->sgfn];
						h_rhss[nv] = cg->d_fgfs[varlrhs->data->sgfn];
						nv++;
						varl0 = varl0->next;
						varl = varl->next;
						varlrhs = varlrhs->next;
					}
					gpu_rungekutta4_batch_launch(
						cg->stream,
						cg->shape, dT_lev,
						h_f0s, h_f1s, h_rhss, nv, iter_count
					);
				}
				if (lev > 0) {
					varl0 = StateList;
					varl = SynchList_pre;
					while (varl0) { // fix BD point
						gpu_sommerfeld_rout_launch(
							cg->stream,
							cg->shape,
							cg->d_X[0], cg->d_X[1], cg->d_X[2],
							Pp->data->bbox[0], Pp->data->bbox[1], Pp->data->bbox[2], Pp->data->bbox[3], Pp->data->bbox[4], Pp->data->bbox[5],
							dT_lev, cg->d_fgfs[phi0->sgfn],
							cg->d_fgfs[Lap0->sgfn], cg->d_fgfs[varl0->data->sgfn], cg->d_fgfs[varl->data->sgfn], varl0->data->SoA,
							Symmetry, cor
						);
						varl0 = varl0->next;
						varl = varl->next;
					}
				}'''
    assert s.count(old1) == 1, "predictor site anchor not unique"
    s = s.replace(old1, new1)

    # ---- 3b. corrector site (varl0=StateList, varl=SynchList_pre, varl1=SynchList_cor, varlrhs=RHSList) ----
    old2 = r'''					MyList<var> *varl0 = StateList, *varl = SynchList_pre, *varl1 = SynchList_cor, *varlrhs = RHSList; // we do not check the correspondence here
					while (varl0) {
						if (lev == 0) { // sommerfeld indeed
							gpu_sommerfeld_routbam_launch(
								cg->stream, cg->shape, cg->d_X[0], cg->d_X[1], cg->d_X[2],
								Pp->data->bbox[0], Pp->data->bbox[1], Pp->data->bbox[2], Pp->data->bbox[3], Pp->data->bbox[4], Pp->data->bbox[5],
								cg->d_fgfs[varl1->data->sgfn],
								cg->d_fgfs[varl->data->sgfn], varl0->data->propspeed, varl0->data->SoA,
								Symmetry
							);
						}
						gpu_rungekutta4_rout_launch(
							cg->stream, cg->shape, dT_lev, 
							cg->d_fgfs[varl0->data->sgfn], cg->d_fgfs[varl1->data->sgfn], cg->d_fgfs[varlrhs->data->sgfn], iter_count
						);
						if (lev > 0) { // fix BD point
							gpu_sommerfeld_rout_launch(
								cg->stream, cg->shape, cg->d_X[0], cg->d_X[1], cg->d_X[2],
								Pp->data->bbox[0], Pp->data->bbox[1], Pp->data->bbox[2], Pp->data->bbox[3], Pp->data->bbox[4], Pp->data->bbox[5],
								dT_lev, cg->d_fgfs[phi0->sgfn],
								cg->d_fgfs[Lap0->sgfn], cg->d_fgfs[varl0->data->sgfn], cg->d_fgfs[varl1->data->sgfn], varl0->data->SoA,
								Symmetry, cor
							);
						}

						varl0 = varl0->next;
						varl = varl->next;
						varl1 = varl1->next;
						varlrhs = varlrhs->next;
					}'''
    new2 = r'''					MyList<var> *varl0 = StateList, *varl = SynchList_pre, *varl1 = SynchList_cor, *varlrhs = RHSList; // we do not check the correspondence here
					// P-RK4: 3-pass restructure (see predictor site).
					{
						const double* h_f0s[RK4_MAXVARS];
						double* h_f1s[RK4_MAXVARS];
						double* h_rhss[RK4_MAXVARS];
						int nv = 0;
						while (varl0) {
							if (lev == 0) { // sommerfeld indeed
								gpu_sommerfeld_routbam_launch(
									cg->stream, cg->shape, cg->d_X[0], cg->d_X[1], cg->d_X[2],
									Pp->data->bbox[0], Pp->data->bbox[1], Pp->data->bbox[2], Pp->data->bbox[3], Pp->data->bbox[4], Pp->data->bbox[5],
									cg->d_fgfs[varl1->data->sgfn],
									cg->d_fgfs[varl->data->sgfn], varl0->data->propspeed, varl0->data->SoA,
									Symmetry
								);
							}
							h_f0s[nv] = cg->d_fgfs[varl0->data->sgfn];
							h_f1s[nv] = cg->d_fgfs[varl1->data->sgfn];
							h_rhss[nv] = cg->d_fgfs[varlrhs->data->sgfn];
							nv++;

							varl0 = varl0->next;
							varl = varl->next;
							varl1 = varl1->next;
							varlrhs = varlrhs->next;
						}
						gpu_rungekutta4_batch_launch(
							cg->stream, cg->shape, dT_lev,
							h_f0s, h_f1s, h_rhss, nv, iter_count
						);
					}
					if (lev > 0) {
						varl0 = StateList;
						varl1 = SynchList_cor;
						while (varl0) { // fix BD point
							gpu_sommerfeld_rout_launch(
								cg->stream, cg->shape, cg->d_X[0], cg->d_X[1], cg->d_X[2],
								Pp->data->bbox[0], Pp->data->bbox[1], Pp->data->bbox[2], Pp->data->bbox[3], Pp->data->bbox[4], Pp->data->bbox[5],
								dT_lev, cg->d_fgfs[phi0->sgfn],
								cg->d_fgfs[Lap0->sgfn], cg->d_fgfs[varl0->data->sgfn], cg->d_fgfs[varl1->data->sgfn], varl0->data->SoA,
								Symmetry, cor
							);
							varl0 = varl0->next;
							varl1 = varl1->next;
						}
					}'''
    assert s.count(old2) == 1, "corrector site anchor not unique"
    s = s.replace(old2, new2)

    open(p, "w").write(s)
    print("patched bssn_step_gpu.C")
    print("RK4 BATCH PATCH DONE")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
