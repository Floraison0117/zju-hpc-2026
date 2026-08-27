# Next Stage: Remaining Bottlenecks

Ordered by estimated performance impact / risk ratio.

## 1. Reduce Register Pressure to Improve Occupancy (HIGH impact, MEDIUM risk)

**Evidence:** Nsight Compute shows the persistent kernel is register-bound at 4 blocks/SM (25% theoretical occupancy). Achieved occupancy is only 14.3%. The ~128 registers/thread allocation limits the GPU's ability to hide DRAM latency through warp context switching.

**Suggested approach:**
- Split the persistent kernel into two sub-kernels: one for residual+apply_ab computation, another for output+state_update. TileLang's JIT may allocate fewer registers per kernel.
- Reduce per-thread fragment count: reuse `chunk_acc` across GEMM calls instead of having separate accumulators.
- Try `threads=64` or `threads=256` to change the register/shared memory tradeoff.

**Expected impact:** Could improve occupancy from 14% to 25-50%, potentially reducing L1TEX stall impact by 20-40%. All cases affected.
**Risk:** Kernel splitting adds sync points. Must maintain correctness.
**Verification:** Nsight Compute on `wide_gva_state`, comparing occupancy and L1TEX stalls before/after.

## 2. Eliminate raw_qk DRAM Traffic via Selective QK Fusion (HIGH impact, MEDIUM risk)

**Evidence:** L1TEX stall dominates at 63.5%. The largest DRAM read per block per chunk is raw_qk (16KB FP32). For `long_low_gva` with 512 chunks, this is 512 * 16KB * 64 blocks = 512 MB of raw_qk reads per kernel launch.

**Suggested approach:**
- Fuse QK GEMM into the persistent kernel for cases where Hv/Hq is small (1 or 2), using the iter4_final approach.
- For Hv/Hq >= 4, keep the current two-kernel path but optimize raw_qk layout for better cache behavior: store as BF16 instead of FP32 (halves DRAM traffic), or use column-major layout for more coalesced reads.

**Expected impact:** Could reduce kernel time by 30-50% for long sequences and 15-25% for medium sequences. Most impactful on `long_low_gva`, `batch_split_gva`, `deep_gva_state`.
**Risk:** Fused QK adds q_shared (16KB shared memory), potentially reducing occupancy further. Must benchmark to confirm net gain.
**Verification:** Compare Candidate A vs Candidate B on all 8 public cases with Nsight profiling.

## 3. Fix Uncoalesced Global Memory Accesses (MEDIUM impact, LOW risk)

**Evidence:** 33% uncoalesced global accesses (4.75M excessive sectors). This wastes ~25% of DRAM bandwidth. The raw_qk read pattern accesses `raw_qk[batch, chunk, qk_head, token, source]` where `source` varies as the innermost loop dimension, but thread mapping is over `(token, value_offset)` – the access stride doesn't match thread layout.

**Suggested approach:**
- Transpose raw_qk storage to `[B, num_chunks, Hq, 64, 64]` → `[B, num_chunks, 64, Hq, 64]` to make the `source` dimension more contiguous with thread access patterns.
- Or: use vectorized loads (float4) for raw_qk reads.
- For output writes: restructure the thread-to-output-dimension mapping.

**Expected impact:** 10-20% improvement on all cases through better DRAM utilization.
**Risk:** Low – layout changes are local to the persistent kernel and don't affect the public API.
**Verification:** Nsight Compute L2 Theoretical Sectors Global Excessive metric.

## 4. Optimize Short Case Overhead (MEDIUM impact, LOW risk)

**Evidence:** `short_tail_state` at 0.472ms still has ~0.1ms of Python overhead (tensor allocation, kernel launch, state clone). For a 1025-token sequence, the raw_qk allocation (557KB) and kernel launch overhead are proportionally large.

**Suggested approach:**
- For num_chunks <= 1 (T <= 64), use a specialized single-chunk kernel that eliminates the chunk loop.
- Pre-allocate reusable output and state buffers (torch caching allocator already does this, but explicit pre-allocation in warmup iterations may help).
- Fuse the initial state zero-fill into the persistent kernel's first iteration.

**Expected impact:** `short_tail_state` could drop from 0.472ms to 0.35-0.40ms.
**Risk:** Low – edge case optimization.
**Verification:** Repeat benchmark on `short_tail_state` with warmup=20.

## 5. Explore Kernel Specialization for Value Head Count (LOW impact, LOW risk)

**Evidence:** Different cases have vastly different Hv (4-64). For small Hv (4-8), only 16-32 blocks are launched – GPU is mostly idle. For large Hv (32-64), 128-256+ blocks are launched.

**Suggested approach:**
- For Hv <= 4: merge multiple value head blocks by processing all value_tiles for a value_head in a single block (using loops over value_tile_index internally).
- For Hv >= 32: keep current block decomposition but try `threads=256` for better SM utilization.
- These are compile-time specializations triggered by Hv, not runtime dispatch.

**Expected impact:** 5-10% on small-Hv cases (`chain_equal`, `short_tail_state`).
**Risk:** Low – TileLang jit handles specialization naturally.
**Verification:** Benchmark `chain_equal` with merged-value-tile kernel vs current.

## Summary of Priority

| Priority | Bottleneck | Expected Speedup | Risk |
|---|---|---|---|
| P0 | Register pressure / occupancy | 1.15-1.25× | Medium |
| P1 | raw_qk DRAM elimination | 1.10-1.30× | Medium |
| P2 | Uncoalesced memory access | 1.10-1.20× | Low |
| P3 | Short case overhead | 1.05-1.15× (short only) | Low |
| P4 | Kernel specialization | 1.05-1.10× | Low |

Combined geometric mean speedup potential: 1.5-2.0× across all cases.

**Immediate next step:** Run full 3×100 benchmarks on both Candidate A and Candidate B to determine which architecture to optimize further. Without empirical data, further optimization is speculative.
