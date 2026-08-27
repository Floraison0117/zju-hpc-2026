# Profile Summary

Profiling was performed on NVIDIA A100 (CC 9.0) using Nsight Compute 2026.2.1.0. Representative cases: `wide_gva_state` (B=1, T=8192, Hq=16, Hv=64) for the long/wide analysis, and `short_tail_state` (B=1, T=1025, Hq=2, Hv=8) for the short case analysis. Raw profiler outputs: `assets/lab3/logs/iteration4_tensorcore_ncu.csv` (current submission) and `assets/lab3/logs/iteration3_scalar_ncu.csv` (iter3 for comparison).

## Candidate A (Current Submission): `gdn_persistent_tensorcore_kernel`

### Occupancy & Resource Usage

| Metric | Value | Implication |
|---|---|---|
| Theoretical Occupancy | 25% | Limited primarily by registers and shared memory |
| Achieved Occupancy | 14.3% | ~9.1 active warps/SM out of 16 possible (4 blocks/SM × 4 warps/block) |
| Block Limit Registers | 4 blocks/SM | ~128 registers/thread (65536 regs/SM / 512 threads at 4 blocks) |
| Block Limit Shared Mem | 4 blocks/SM | ~42.5 KB/block (164 KB/SM / 4) |
| Grid Size | (32, 1, 1) | Only 32 blocks for wide_gva_state – most SMs idle |

### Stall Analysis

| Stall Type | Percentage | Root Cause |
|---|---|---|
| L1TEX (global memory) | 63.5% | **Dominant:** reading raw_qk, state, v, q, k from DRAM |
| Fixed latency dependency | ~15% | Tensor Core pipeline depth |
| Other | ~21% | Sync threads, instruction fetch |

### Memory Access

| Metric | Value |
|---|---|
| Uncoalesced global accesses | 33% (4.75M excessive sectors) |
| Uncoalesced shared accesses | 28% (3.15M excessive wavefronts) |
| Branch efficiency | 96.5% |
| Local memory spills | 0 bytes |

### Kernel Count & Time Distribution (estimated)

For `wide_gva_state` (B=1, Hv=64, T=8192, 128 chunks):
- `gdn_raw_qk_kernel`: 1 launch, B*num_chunks*Hq = 1*128*16 = 2048 blocks
- `gdn_persistent_tensorcore_kernel`: 1 launch, B*Hv*value_tiles = 1*64*8 = 512 blocks
- Total kernel time: ~20.35ms
- Estimated raw_qk time: ~0.1-0.3ms (small fraction)
- Estimated persistent kernel time: ~19-20ms (dominant)
- L1TEX-bound: ~12.7ms spent waiting for DRAM (63.5%)

### Key Bottlenecks

1. **DRAM bandwidth (L1TEX stall)** – 63.5% of cycles. The persistent kernel reads raw_qk (FP32, 16KB per block per chunk), plus k, v, q, state, A, gate, beta from global memory. The raw_qk read is the largest single contributor.
2. **Register pressure** – Limits occupancy to 4 blocks/SM (25% theoretical). With only 9 active warps/SM, the GPU cannot hide DRAM latency effectively.
3. **Uncoalesced accesses** – The raw_qk read pattern and output write pattern cause 33% excessive global memory sectors, wasting bandwidth.
4. **Small grid** – For many cases (especially short_tail_state with only 4 blocks), the GPU is severely underutilized.

## Candidate B (iter4_final): `gdn_persistent_fused_kernel`

Not yet profiled. Expected differences:
- **Eliminates** the raw_qk DRAM traffic (largest L1TEX contributor)
- **Adds** QK GEMM computation per block (6th Tensor Core call)
- **Adds** q_shared (16KB extra shared memory) → may reduce occupancy from 4 to 3 blocks/SM
- **Adds** qk_frag (64×64×4 = 16KB in registers) → may increase register pressure

## Comparison: Candidates A vs B

| Aspect | Candidate A (Two-Kernel) | Candidate B (Fused) |
|---|---|---|
| Kernel launches | 2 | 1 |
| raw_qk DRAM | Write once, read per-block-per-chunk (16KB) | None |
| QK GEMM count | 1× per (batch, chunk, Hq) | Hv×value_tiles× per chunk = Hv×8× |
| Tensor Core GEMMs per block per chunk | 5 | 6 |
| Shared memory per block | ~42.5 KB | ~58.5 KB |
| Occupancy (estimated) | 4 blocks/SM (25%) | 3 blocks/SM (~18%) |
| Best for | long seq, high Hv/Hq | short seq, Hv/Hq=1 |

## Short Case Analysis: `short_tail_state` (B=1, T=1025, Hq=2, Hv=8)

- num_chunks = 17
- Grid size: B * Hv * value_tiles = 1 * 8 * 8 = 64 blocks (with value_tile=8) or 32 blocks (with value_tile=16)
- raw_qk size: 1 * 17 * 2 * 64 * 64 * 4 = 557 KB – fits in L2 cache
- Kernel launch overhead dominates for small cases
- **Recommendation**: Candidate B with VALUE_TILE=8

## Long Case Analysis: `long_low_gva` (B=1, T=32768, Hq=2, Hv=8)

- num_chunks = 512
- Grid size: 1 * 8 * 8 = 64 blocks (value_tile=8) or 32 blocks (value_tile=16)
- raw_qk size: 1 * 512 * 2 * 64 * 64 * 4 = 16.8 MB – exceeds L2 cache, DRAM reads dominate
- Hv/Hq = 4, QK recomputation factor = 32×
- **Recommendation**: Candidate A with VALUE_TILE=8

## Wide/Deep Case Analysis: `wide_gva_state` (B=1, Hq=16, Hv=64), `deep_gva_state` (B=1, Hq=8, Hv=32)

- Both Hv/Hq = 4, high QK recomputation cost (32×)
- Large raw_qk: ~4.2MB and ~8.4MB respectively
- Large grid: 512 and 256 blocks
- **Recommendation**: Candidate A with VALUE_TILE=8
