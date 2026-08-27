# ABEGPU optimization context for diagnosis

## Purpose and evidence policy

This file is a diagnosis handoff for another model. It consolidates the repository's ABEGPU optimization context, with emphasis on `rhs_kernel` and `prolong3`. It does not claim that every scratch snapshot is the current formal remote source. The formal source is documented as `~/lab4-gpu/src` on `zju-hpc-lab2`; repository files under `tmp/` are preserved snapshots or experiment artifacts and must be compared against the formal source hash before any deployment.

`labs/lab4.typ` was explicitly excluded and was not used.

Measured facts, historical measurements, and proposed ideas are separated below. Some records reflect different optimization dates or node noise, so do not combine their end-to-end totals as if they came from one run. The most recent search-memory state is the best summary of what is deployed and what has already been tested.

## Executive summary

- Target: Lab4 task2, `ABEGPU`, on `zju-hpc-lab2`, documented hardware NVIDIA A100 MIG 1g.10gb, one GPU slice, 10 GiB, compute capability 8.0. The repository warns elsewhere that profiler device labels can be misleading; use the partition/report context as authority.
- OJ-equivalent workload: `MPI_processes=1`, `OMP_threads=8`, `GPU_Calculation=yes`, `Final_Evolution_Time=100.0`, and the judge forces `Analysis_Time=0.1`. There are 100 evolution steps.
- The latest optimization memory reports a deployed baseline of about **1007.28 s** with two main GPU optimizations: force-inlining four stencil helpers and branchless `fh`. A later validated `BR_ORD-fix` candidate reports **996.47 s**, `check.sh FINAL PASS`, `trajectoryRMS=0`, but is described as “待部署候选” in the current search-memory.
- A historical OJ submission was **1044.26 s / 64 points**, correct. Earlier records include 1228.53 s, 1266.73 s, and 1271.50 s from older stacks or different nodes. Use per-job A/B measurements, not cross-date totals, to judge a candidate.
- `rhs_kernel` is the main hotspot: about **69.1% of GPU kernel time**, historically **8.34–8.52 s per evolution step**, roughly **3.39 ms per launch**, 410 launches/step in one nsys breakdown.
- `prolong3_kernel` is the second major hotspot: about **16.5%**, historically **1.99–2.31 s/step**, with approximately **11,821 launches/step**. It is not simply bandwidth- or occupancy-limited; ncu points to fixed-latency execution dependencies and poor scheduler eligibility.
- `rhs_kernel` is a 1075-line monolithic point kernel. With `__launch_bounds__(256,2)` it uses 128 registers/thread and reaches about 21.5–25% occupancy. Its unconstrained natural register demand is about 250 registers/thread. It is latency-bound: L1TEX scoreboard stalls 46.6%, “No Eligible” 64.7%, eligible warps about 0.49/scheduler, L1 hit 80.91%, L2 hit 91.14%.
- The largest proven `rhs_kernel` win was moving four cross-translation-unit stencil helpers into a header and applying `__forceinline__`: **1266 → 1052 s, -13.8%**, bit-exact. Branchless `fh` added **1052 → 1007 s, -5.5%**, bit-exact.
- The main failed `rhs_kernel` families are launch-bounds tuning, 2-way splitting, moving Ricci contractions between split kernels, full/shared staging, `__ldg`/`__restrict__`, load hoisting/software-pipeline-like rewrites, and FP32. Their mechanisms and evidence are documented below.
- `prolong3` launch bounds `(256,3)` and a bit-exact rewrite of the serial Z accumulation both measured zero end-to-end benefit. The unresolved directions are deeper loop/dataflow restructuring, symmetry-boundary-call reduction, and cross-variable launch batching. These are ideas, not validated solutions.

## Source and evidence map

### Primary summaries

- `assets/lab4/kb/search-memory.md`: newest tested-lever table, deployed stack, job IDs, validation outcomes, and dead-end analysis.
- `assets/lab4/kb/hardware/task2-a100-gpu.md`: hardware, workload, current bottleneck summary, and formal remote paths.
- `assets/lab4/kb/patterns/gpu-kernel.md`: distilled optimization patterns and stopping rules.
- `README-lab4.md`, especially sections 15 and 15.8: chronological experiment/OJ record. This file mixes older and newer states; prefer search-memory for the current status.
- `tmp/lab4-abegpu-deep-progress.md`: detailed ncu and experiment log, including rhs register diagnostics and prolong3 experiments.
- `tmp/scout_rhs_brief.md`: source-level `rhs_kernel` decomposition and early opportunity assessment.
- `tmp/lab4-abegpu-goal-progress.md`: historical goal progress and lower-bound reasoning.
- `tmp/lab4-abegpu-from-abecpu-lessons.md`: later retrospective connecting ncu diagnostics to possible GPU work.
- `tmp/lab4-abegpu-combo-plan.md`: an older plan that proposed launch batching and tiling. Treat proposed items there as unverified unless search-memory records a job result.

### Local code snapshots

- `tmp/bssn_rhs_gpu_deployed.cu`: preserved deployed-style rhs snapshot. SHA-256 `4A4B2AABD2F2083D905F2ED838712D6C560F78FFFE4BF4C64D6D8CD6741DF8DF`.
- `tmp/lab4-cpu-sync/src/bssn_rhs_gpu.cu`: older/source snapshot without rhs launch bounds. SHA-256 `00704282AB1C85CC7D69F47B25FAC1CA577067D9112900C468F0AF91B5CF7463`.
- `tmp/lab4-cpu-sync/src/prolongrestrict_cell_gpu.cu`: prolong/restrict implementation snapshot. SHA-256 `1ED5BD43E1C451E80A45AE8B1FBB06E217735EF462AA924B0D88C189F3C62644`.
- `tmp/lab4-cpu-sync/src/diff_new_gpu.cu`: older out-of-line derivative helpers. SHA-256 `24D297E0335F9479F33BAEE66A030E3679FCA673D32A48006ED22AA940604278`.
- `tmp/lab4-cpu-sync/src/bssn_step_gpu.C`: host-side step orchestration snapshot. SHA-256 `24727EFB408863CFCF25E0C0088BA83FD5E4A334EB3AC2A2CE72B827766182C5`.
- `tmp/patch_prolong3_unroll.py`: patch used for the tested Z-accumulation expression rewrite.
- `tmp/bssn_rhs_gpu_v1.cu`, `tmp/bssn_rhs_gpu_tiling.cu`, `tmp/iter5/bssn_rhs_gpu.cu`: historical candidates; inspect only to understand prior attempts, never assume deployability.

The only meaningful diff between the two rhs snapshots above is the deployed snapshot adding `__launch_bounds__(256,2)` to `rhs_kernel`; comments also differ. The force-inlined derivatives used by the current remote deployed stack are not fully represented by the older `tmp/lab4-cpu-sync/src/diff_new_gpu.cu` snapshot. The current forceinline change reportedly moved four functions into `derivatives.h`.

## End-to-end and per-step model

One verified two-step nsys breakdown (job 121872, historical pre-latest stack) reports:

| Component | Launches/step | GPU-kernel share | Approx. time/step |
|---|---:|---:|---:|
| `rhs_kernel` | 410 | 69.1% | 8.34 s |
| `prolong3_kernel` | 11,821 | 16.5% | 1.99 s |
| `restrict3_kernel` | 2,407 | 5.1% | 0.62 s |
| interpolation/analysis | many | about 3.8% | 0.46 s |
| Sommerfeld | many | about 2.7% | 0.33 s |
| RK4 | many | about 1.7% | 0.20 s |
| other | — | about 1.3% | 0.16 s |

GPU kernels were reported essentially continuously busy: about 12.78 s kernel time/step versus 12.31 s wall time/step in an older profile. Host-device copies were only about 0.289 s/step. This argues against transfer overlap or generic stream work as the primary lever.

An older lower-bound argument used:

```text
TwoPuncture                    about 28 s (later OJ observations often about 34–40 s)
non-rhs evolution             3.81 s/step × 100 = 381 s
rhs                           8.52 s/step × 100 = 852 s
historical total              about 1261 s
```

It concluded that eliminating rhs entirely would still leave roughly 409 s, above a 340 s target. This lower-bound argument is useful for prioritization but is tied to the then-current non-rhs implementation and measured decomposition. It is not proof that a redesigned `prolong3`/launch structure cannot lower the non-rhs floor. It does prove that optimizing rhs alone cannot reach 340 s under those measurements.

## `rhs_kernel`: exact code shape

### Launch shape and interface

In `tmp/bssn_rhs_gpu_deployed.cu`:

```cpp
__global__ __launch_bounds__(256, 2) void rhs_kernel(
    int ex0, int ex1, int ex2, double T, double* X, double* Y, double* Z,
    // 20+ evolved input fields,
    // 20+ RHS output fields,
    // matter fields,
    // 18 Christoffel output fields,
    // 6 Ricci outputs and 7 constraint outputs,
    int symmetry, int lev, double eps, int co);
```

The wrapper launches:

```cpp
dim3 block(8, 8, 4);             // 256 threads
dim3 grid(ceil(ex[0]/8), ceil(ex[1]/8), ceil(ex[2]/4));
rhs_kernel<<<grid, block, 0, stream>>>(...);
```

Each thread owns one `(i,j,k)` point, uses Fortran/column-major layout with x contiguous, and computes essentially the full BSSN right-hand side plus constraints for that point.

### Major phases inside the monolith

The current snapshot is organized roughly as follows:

1. Load lapse/conformal factor, metric components, trace curvature, and extrinsic curvature.
2. First derivatives of the three shift fields through three `d_fderivs_point` calls.
3. First derivatives of `chi`; compute `chi_rhs`.
4. First derivatives of six metric components, creating 18 live derivative scalars.
5. Inverse metric and Christoffel symbols. Eighteen local Christoffel values are formed and written to global output arrays.
6. Lapse/trK derivatives, shift second/upwind derivatives, and gauge RHS work.
7. First derivatives of `Gamx/y/z` and second derivatives of all six metric fields.
8. Large Ricci tensor contractions. This is the “Ricci monster”: many of the metric derivatives, inverse metric values, Christoffels, and Gamma derivatives overlap in liveness.
9. `chi` and lapse second derivatives, covariant corrections, Ricci completion, and evolution RHS values.
10. Hamiltonian, momentum, and Gamma constraints. This introduces another six calls for Aij first derivatives and keeps many prior quantities live.

Call count visible in the snapshot is approximately 19 calls to `d_fderivs_point` and 11 calls to `d_fdderivs_point`, plus other helpers such as lopsided derivatives/Kreiss-Oliger logic. Historical notes describe 26 cross-TU device calls before force-inlining the hottest stencil helpers.

### Why register pressure is intrinsic

The key measured compiler experiment removed `__launch_bounds__`:

| Build | Registers/thread | Stack | Spill stores | Spill loads | Expected occupancy |
|---|---:|---:|---:|---:|---:|
| `__launch_bounds__(256,2)` | 128 | 1256 B | 692 B | 1688 B | 25%, 2 blocks/SM |
| no launch bounds | about 250 | 888 B | 104 B | 152 B | about 6%, 1 block/SM |

The compiler naturally wants nearly the architectural maximum. Forcing 128 registers permits two 256-thread blocks/SM, but spills. More aggressive bounds permit theoretical occupancy at the cost of enormous spills:

- `(256,3)`: about 80 registers and about 5,116 B spill in one post-forceinline measurement; around 10% slower.
- `(256,4)`: about 64 registers and about 6,416 B spill; around 15% slower.

This indicates that simple occupancy tuning is the wrong model. The goal is not “lower registers at any cost”; a useful rewrite must reduce the natural live set or shorten dependency chains without materializing a huge global scratch working set.

### Runtime bottleneck counters

The forceinline deployed state was measured at approximately:

- 3.39 ms/launch.
- Achieved occupancy 21.51%.
- Active warps around 3.52 of 16 per scheduler in an earlier report.
- Eligible warps about 0.49/scheduler.
- L1TEX scoreboard stall 46.6%.
- “No Eligible” 64.7%.
- L1 hit rate 80.91%, L2 hit rate 91.14%.
- Zero shared memory in the baseline rhs kernel.

Interpretation: the kernel lacks enough independent eligible work to hide dependent stencil-load latency. Cache hit rates are already high. This is not evidence that global bandwidth is saturated; it is evidence of load-to-use and/or instruction dependency latency combined with low warp residency.

## `rhs_kernel`: validated improvements

### 1. Forceinline the stencil helpers

Change: move the four hot derivative/stencil device routines into `derivatives.h` and apply `__device__ __forceinline__` so the monolithic rhs kernel no longer pays cross-translation-unit RDC device-call overhead and the compiler can schedule through the helper bodies.

Result:

- End-to-end about 1266 → 1052 s, **-13.8%**.
- `check.sh PASS`, `trajectoryRMS=0`, bit-exact.
- Achieved occupancy remained around 21.5%; the win was instruction/call-boundary and scheduling related, not occupancy related.

This is the strongest evidence that dependency scheduling across derivative calls matters.

### 2. Branchless `fh`

Change: replace the `fh` lambda's branch chain with branchless selection while preserving memory safety.

Result:

- About 1052 → 1007 s, **-5.5%**.
- Full correctness pass, bit-exact.

### 3. `BR_ORD-fix` candidate

Change: branchless 4th/2nd-order selection in `d_fderivs_point`, while preserving the original early return and clamping all `fh` indices before unconditional loads.

Result:

- 1007 → **996.47 s**, about **-1.07%** in the full 100-step run; short two-step A/B showed a larger but nonrepresentative gain.
- `check.sh FINAL PASS`, `trajectoryRMS=0`, bit-exact.
- Search-memory labels this as a validated candidate, not clearly deployed.

Safety lesson: the first un-clamped branchless version was bit-exact and 6.7% faster for two steps, then failed around step 28 with CUDA illegal memory access when refined/moving grids activated. Multiplying an out-of-range load by a validity mask does not make the load safe. Preserve early-return semantics and clamp every address used by unconditional evaluation.

## `rhs_kernel`: failed or exhausted approaches

Do not repeat these unchanged.

| Attempt | Measured outcome | Mechanism/diagnosis |
|---|---|---|
| launch bounds `(256,1)` / unconstrained | neutral or worse | near-250 natural registers collapse occupancy |
| launch bounds `(256,3)` | about +10% slower | 5 KB-class spills overwhelm occupancy benefit |
| launch bounds `(256,4)` | about +15% slower | still larger spills |
| block shape changes | no reliable benefit | does not change the intrinsic live set/dependency graph |
| rhs split v1 | about 3% slower | extra global intermediate traffic plus RDC ABI spill, reported 952 B |
| rhs split v2 after forceinline | rejected at ptxas stage | second kernel spill stores rose about 692 → 3,880 B and loads to about 4,416 B |
| Lever A: move 18 Ricci contractions into split kernel 1 and scratch them | dead | kernel 2 spill stores 3,880 → 3,896 B; live-set relief did not materialize |
| rhs inline `.cuh` experiment before final forceinline approach | neutral/negative in older log | exact candidate differs from successful four-helper forceinline; do not conflate them |
| full shared-memory staging | infeasible/negative | staging all relevant fields needs roughly 120 KB/block, beyond useful SM budget and does not solve registers |
| stage only `l_Gam` in shared memory (`GAMSMEM`) | zero | spills fell about 2,488 → 1,472 B but shared-memory latency canceled benefit |
| `__ldg` + `const __restrict__` | zero, bit-exact | read-only cache path did not reduce scoreboard stalls; caches already hit well |
| load hoisting / `swpipe2` | zero | spills grew about 4× |
| FP32/mixed precision | speed only about 3.7%, correctness failed | 2-step RMS 0.00715 > 0.001; ptxas still used 128 registers at lb2 and occupancy remained 25% |
| per-stream synchronization | corrected to zero | repeated A/B overturned an earlier apparent ~30 s gain |

### Why the 2-way split failed

The proposed split materialized intermediates in global scratch. Kernel 1 became lighter, but kernel 2 still contained the dense Ricci section. The second kernel simultaneously needs roughly:

- 18 first-derivative/contraction quantities,
- 18 Christoffel values,
- 9 Gamma derivatives,
- 6 inverse-metric values,
- plus Ricci accumulators and other scalars.

The forceinline optimization removes call/scheduling overhead but does not shrink this mathematical live set. Moving 18 contraction results earlier merely replaced recomputation/liveness with 18 scratch loads and did not reduce compiler register demand. A future split must be based on an actual liveness/SASS analysis and a cut where both sides have bounded state, not a source-code midpoint.

## `rhs_kernel`: unresolved directions worth diagnosing

These are not known wins. A new model should rank them by expected end-to-end impact, implementation risk, scratch footprint, and bit-exactness.

1. **Inspect the current formal SASS/source correlation after all deployed inlining/branchless changes.** Existing ncu counters identify scoreboard latency but not the precise source/SASS regions. Collect per-PC sampling/source counters for `d_fderivs_point`, `d_fdderivs_point`, Ricci contractions, and constraint tail separately if tooling allows.
2. **Second-derivative branchless safety rewrite.** `d_fdderivs_point` is called many times and may offer a sibling of `BR_ORD-fix`. It must use clamped addresses and retain early-return semantics. Validate through the moving-grid activation beyond step 28.
3. **Reduce repeated stencil loads across derivative calls.** The kernel repeatedly loads neighborhoods for the same fields. A useful design might compute a field's first and second derivatives together, reuse already-loaded ±1/±2 values, and consume results near production. The difficulty is avoiding an even larger live set.
4. **Recompute-on-demand / phase-local variables.** The old Lever B idea was to finish one Ricci component, release its contractions, and recompute limited quantities for the next component. This spends FLOPs to shorten liveness. It was proposed but not validated. The risk is that common Christoffels and inputs remain live, so ptxas may see little reduction.
5. **Three-way or component-wise Ricci decomposition.** Unlike the failed two-way split, isolate the Ricci monster and/or constraint tail at a liveness-minimal boundary. Before running, compile each candidate with `-Xptxas -v` and quantify registers, stack, spills, and scratch bytes/point. The whole end-to-end model must include added global traffic and launches.
6. **Warp/block cooperative neighborhood reuse for a small number of hottest fields.** Full-field shared staging is impossible. Partial staging must be justified by source-correlated load counts, reuse across threads, halo overhead for an `8×8×4` or alternative tile, and bank/layout analysis. The prior `l_Gam` staging result warns that reducing spills alone is insufficient.
7. **Register/liveness-guided source scheduling.** Move stores/consumption closer to production; open new scopes only if nvcc actually shortens liveness. Verify with ptxas and SASS, since C++ scopes/comments alone do not force allocation behavior.
8. **Kernel launch aggregation across patches only if independent.** rhs has about 410 launches/step, far fewer than prolong3, so launch aggregation has limited upside relative to solving its 8+ s compute time.

## `prolong3`: exact code shape

The local snapshot `tmp/lab4-cpu-sync/src/prolongrestrict_cell_gpu.cu` implements fifth-order tensor-product interpolation. Each output point is handled by one CUDA thread.

### Wrapper and launch

The host wrapper recomputes geometry/alignment and the valid target box for each launch, then uses a one-dimensional launch:

```cpp
int total_points = ni * nj * nk;
int block = 256;
int grid = (total_points + block - 1) / block;
prolong3_kernel<<<grid, block, 0, stream>>>(...);
```

The kernel converts the flat point index to `(i,j,k)`, constructs several thread-local 3-element arrays from scalar kernel parameters, and calls `d_prolong3_device`:

```cpp
double arr_llbc[3], arr_uubc[3], arr_llbf[3], arr_uubf[3];
double arr_llbt[3], arr_uubt[3], arr_SoA[3];
int arr_extc[3], arr_extf[3];
d_prolong3_device(i, j, k, ...);
```

This interface may induce local stack/addressable state. The reported ptxas state for prolong3 was 336 B stack and zero spill. “Zero spill” does not prove those local arrays are free; inspect SASS/local-memory counters and whether scalarizing geometry removes address computations or local loads.

### Per-point interpolation

`d_prolong3_device` does all of the following per output point:

1. Recompute coarse/fine cell dimensions `CD[3]`, `FD[3]`.
2. Recompute alignment base and integer bounds with Fortran-compatible truncation/idint behavior.
3. Reject points outside the target box, even though the host wrapper already launches only the computed valid box.
4. Compute coarse indices and parity of fine global indices.
5. Allocate `double tmp2[6][6]` and `double tmp1[6]` per thread.
6. Z interpolation: for 36 `(m,n)` pairs, make six `d_symmetry_bd_1b` calls and accumulate six weighted values. This is **216 symmetry-aware source accesses per output point**.
7. Y interpolation: six length-6 dot products over `tmp2`.
8. X interpolation: one length-6 dot product over `tmp1`.
9. Write one fine-grid double.

The coefficients live in CUDA constant memory:

```cpp
{77/8192, -693/8192, 3465/4096, 1155/4096, -495/8192, 63/8192}
```

The order and parity reversal must remain compatible with the Fortran reference. Bit-exact validation is expected for safe transformations.

### Likely structural costs visible from source

- Massive repeated geometry/index work per output point even though all threads in a launch share bounds/extents/spacing.
- 216 indirect symmetry-boundary helper calls/point. If most launched points are interior, a fast interior path could avoid per-load symmetry logic, with a separate boundary kernel/path.
- A 36-double `tmp2` plus 6-double `tmp1` per thread. Even with zero reported spill, this creates a large dependency/local-state footprint.
- Three sequential tensor-product stages. The Z accumulation was visibly serial on `val`; however, changing its expression alone did not produce measurable improvement.
- Extremely high launch count: about 11,821 launches/step in the historical breakdown, likely because the host orchestration invokes prolongation separately for many patches/variables.

## `prolong3`: profile and completed experiments

An ncu run (`ncu3`, job 144849) reported prolong3 as not compute-bound, with approximately:

- 66 registers/thread.
- Achieved occupancy about 32.2%.
- Eligible warps about 0.64/scheduler.
- “No Eligible” about 68.6%.
- Fixed-latency execution dependency as the principal stall interpretation.
- ncu estimated speedup suggestions around 31.3% from stall reduction, 28.7% from occupancy gap, and 49.4% scheduler-related. These are profiler heuristics, not additive or guaranteed speedups.

At 2.31 s/step in that later profile, a genuine 31% improvement would be roughly 0.72 s/step or 72 s/100 steps, so prolong3 is large enough to matter.

Completed experiments:

1. **`__launch_bounds__(256,3)`**: 66 registers, 336 B stack, zero spill, bit-exact. Two-step A/B was within noise. Conclusion: occupancy forcing alone did not address the fixed dependency chain.
2. **Z-loop accumulation rewrite**: replaced six sequential `val += coefficient * load` statements with a single ordered expression intended to expose independent loads while preserving operation order. Bit-exact. A later 100-step measurement confirmed **zero benefit**. Conclusion: the visible `val` chain was not the dominant unresolved stall, or the compiler already scheduled equivalently.

Do not repeat either unchanged.

## `prolong3`: high-value unresolved questions

1. **Where exactly are the dependency stalls?** Source-correlate SASS PCs to `d_symmetry_bd_1b`, local `tmp2/tmp1` traffic, integer division/modulo/indexing, and floating-point dot products. The prior source-level guess about `val +=` was wrong.
2. **Can launch-invariant geometry be precomputed once?** The host already computes the valid box. Pass precomputed `base`, `lbf/lbc`, coarse-index offsets, and parity origins instead of recomputing division/idint arrays per point. Confirm bitwise equivalence of all integer decisions.
3. **Can the valid-range check inside `d_prolong3_device` be removed safely?** The wrapper derives exactly the launch box. Prove the host and device formulas are identical for every symmetry/grid case before removing it.
4. **Can local arrays be scalarized or the tensor product reorganized?** A direct separable implementation may compute smaller tiles/stages, but any extra global intermediate array could erase gains. First use ptxas/SASS/local-memory metrics to learn whether `tmp2[36]` is registers, local stack, or optimized expressions.
5. **Interior/boundary specialization.** Launch a fast interior kernel using direct coalesced loads and a small boundary kernel using `d_symmetry_bd_1b`. This targets helper/control/index overhead without changing interpolation arithmetic. The exact interior domain must account for the 6-point stencil and symmetry semantics.
6. **Cross-variable or cross-task batching.** About 11.8k launches/step is exceptional. Batch multiple independent fields/tasks into one launch with a descriptor array or add a variable/task dimension. This can reduce host launch overhead and may improve occupancy for tiny patches. Risks: pointer indirection, divergent task dimensions, loss of per-task stream parallelism, and larger parameter/descriptor traffic.
7. **Persistent/work-queue kernel.** Build a device task list describing `(src,dst,geometry,bounds,symmetry)` and let blocks consume tasks. This is a stronger form of batching and may amortize setup. It must preserve stream/dependency ordering and cannot allow a fine patch to run before its coarse source is ready.
8. **Block-cooperative reuse.** Adjacent fine points share many coarse samples. A tile could stage a coarse brick once. Analyze parity, the 6-point halo in each dimension, symmetry boundaries, shared-memory footprint, and whether many production patches are too small/irregular for reuse.
9. **Exploit parity structure.** Fine points map to coarse cells with even/odd coefficient reversal. Computing a small group of neighboring fine points together may reuse the same 6×6×6 coarse cube and produce multiple outputs, increasing arithmetic per load. This is likely a more fundamental opportunity than merely unrolling one accumulation.

## Host scheduling and call-graph context

`bssn_step_gpu.C` is the orchestration layer for RK stages, ghost exchange, prolong/restrict, and streams. `gpu_manager.cu` maintains a stream pool. Historical profiling indicates the GPU is already nearly saturated, so generic asynchronous copies and replacing global sync with per-stream sync did not help reliably.

The unusually high prolong/restrict/RK/Sommerfeld launch count still leaves a distinct batching opportunity: saturation does not mean launch overhead is zero. Batching must be tested with same-node interleaved A/B runs and per-kernel metrics. An older plan estimated approximately 30,541 tiny prolong3/RK4/Sommerfeld launches per step, but this number is a planning estimate; the nsys table's measured component counts are more reliable.

Dependencies matter. Prolongation and restriction participate in mesh refinement/ghost filling, so arbitrary cross-patch fusion may violate ordering. Any task-list design needs explicit dependency levels or separate launches/events per refinement phase.

## Correctness and benchmark constraints

- Never modify the runner, reference, timing logic, evaluation cases, or scoring infrastructure.
- Never hard-code outputs or branch on test input values.
- Preserve the numerical method unless an explicitly authorized algorithmic rewrite remains equivalent under the lab rules.
- Fast validation is not sufficient. The original branchless-order candidate passed two steps and crashed at step 28 when new grid geometry appeared.
- Acceptance should include:
  - Level 0 compile and ptxas inspection.
  - Same-job, same-node, interleaved short A/B on identical steps/grids.
  - Full 100-step `check.sh` validation.
  - `trajectoryRMS <= 1e-3`; current accepted candidates achieve RMS 0/bit-exact.
  - Constraint maxima within the documented threshold (historically ≤2).
  - Raw profiler/timing logs retained.
- Before formal writes, compare the current remote source hash against the recorded baseline and create a recoverable snapshot. The formal package may have drifted beyond these local snapshots.
- Only the main agent should deploy a formal candidate or submit to OJ, and OJ submission requires explicit user request.

## Recommended diagnostic sequence for the next model

1. **Establish the exact current formal state.** Read-only hash/diff `~/lab4-gpu/src` against the latest recorded baseline and identify whether `BR_ORD-fix` is deployed. Do not overwrite anything.
2. **Reproduce a compact current nsys decomposition.** Confirm rhs/prolong shares and launch counts after forceinline + branchless changes. Old 69.1%/16.5% fractions may have shifted.
3. **Collect source-correlated ncu/SASS for both kernels.** For rhs, isolate the PCs behind L1TEX scoreboard stalls. For prolong3, determine whether the fixed dependency is symmetry helper logic, integer addressing, local-memory traffic, or FP dependencies.
4. **Use compiler gates before expensive runs.** For rhs candidates, reject any design whose natural registers/spills or scratch bytes/point are plainly worse. For prolong3, record registers, stack, local-memory transactions, and instruction count.
5. **Prioritize one independent variable per candidate.** The cleanest high-information candidates are:
   - prolong3 launch-invariant geometry precompute;
   - prolong3 interior/boundary specialization;
   - prolong3 cross-task batching with identical dependency phase;
   - safe `d_fdderivs_point` branchless/clamped rewrite;
   - rhs field-local derivative reuse only after source-correlated load evidence.
6. **Measure kernel time and end-to-end time.** A kernel micro-win that changes launch scheduling, scratch traffic, or synchronization can lose globally.

## Questions the next model should answer

1. In the current deployed binary, which exact SASS PCs account for rhs's 46.6% L1TEX scoreboard stalls, and which source expressions generate them?
2. Are `d_fderivs_point` and `d_fdderivs_point` fully inlined in the current formal binary, or are any RDC calls still present?
3. Can first and second derivatives of one field share loaded stencil values without increasing the peak live set beyond 128 registers or causing spills?
4. Is rhs's constraint tail required every RK substep, or can it legally be separated/called only at required analysis times without changing semantics? Verify from the real call graph and rules; do not assume.
5. For prolong3, are `tmp2`/`tmp1` held in registers, local memory, or optimized away? What are local load/store counts?
6. What fraction of prolong3 points are interior versus symmetry/boundary points, and what fraction of launches are tiny?
7. How many prolong tasks at the same dependency phase can be batched, and what is the launch-time contribution measured by CUDA API tracing?
8. Can a block produce multiple adjacent fine points from one shared coarse tile while keeping the exact coefficient/order behavior?
9. Does precomputing geometry and valid bounds on the host/device task descriptor preserve every Fortran `idint` rounding decision bit-for-bit?
10. Given the current roughly 996–1007 s state, what realistic combination of independently measured wins is required, and what target is physically plausible under the no-algorithm-change constraint?

## Chronology and status cautions

- Early baseline and first OJ records are slower than the current optimization-memory baseline.
- The apparent per-stream synchronization win was later disproved by repeated A/B and must be treated as zero.
- A short-run 6.7% branchless-order improvement was unsafe; only the clamped hybrid passed 100 steps, and its full-run gain was about 1.1%.
- The FP32 result initially looked larger when compared across different evolution steps; same-grid A/B reduced the speedup to about 3.7%, and correctness still failed.
- `prolong3` profiler “estimated speedup” values are diagnostic ceilings, not observed gains.
- Statements that all opportunities are “exhausted” apply to the specific tested transformations. They do not prove that task batching, interior specialization, dataflow redesign, or a correctly chosen liveness cut cannot work.

## Compact no-repeat checklist

Already tested with no useful benefit or worse:

- rhs launch-bounds 1/3/4-block variants.
- rhs 2-way split v1 and v2.
- rhs split Lever A contraction scratch.
- rhs `__ldg`/`__restrict__`.
- rhs full shared staging and `l_Gam` shared staging.
- rhs load hoist/swpipe2.
- mixed FP32 (also incorrect).
- generic per-stream synchronization.
- prolong3 `__launch_bounds__(256,3)`.
- prolong3 six-term Z accumulation expression rewrite.

Known good and should be preserved unless superseded by evidence:

- rhs `(8,8,4)` block with `__launch_bounds__(256,2)`.
- forceinline hot stencil helpers.
- branchless safe `fh`.
- validated `BR_ORD-fix` semantics: keep early return, clamp unconditional-load indices, branchless select.
- TwoPuncture GPU/CPU build decoupling and CPU OpenMP tuning, which reduce the non-ABEGPU front end without changing ABEGPU CUDA sources.

