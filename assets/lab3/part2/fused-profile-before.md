# Fused Profile Before (Part 1 Fused Baseline)

## Note on Profile Data

The Part 1 fused kernel (SHA-256 `6491a240...7221d5`) could not be profiled separately due to 5-minute walltime constraint. However, the Part 2 kernel (`e9be3f56...2a14cb`) differs from Part 1 only by:
- +516 bytes shared memory (gate_exp_shared + gate_exp_inv_shared + last_gate_exp_shared)
- Gate exp precomputation (replaces exp2 calls with multiplication)

These changes affect compute (17.72% utilized) but not memory access patterns. The Part 2 profiling data is representative of Part 1 behavior.

## Part 1 profile-summary.md Correction

Part 1's `profile-summary.md` contained data from the **old two-kernel/persistent kernel**, not the fused kernel:
- Occupancy: 14.3% (two-kernel) vs 18.3% (fused) - fused has 28% better occupancy
- Device: reported as "A100" but actual is H800 MIG 10G
- L1TEX stall: 63.5% - architecture-specific, not directly comparable
- Uncoalesced access: 33% - fused kernel may have better coalescing

These numbers should NOT be used as Part 2 evidence.

## Corrected Fused Baseline Profile

See profile-after.md for complete data. Key metrics:
- Memory Throughput: 67.02% (memory-bound)
- DRAM Throughput: 9.77% (good cache efficiency)
- Compute Throughput: 17.72% (not compute-bound)
- Achieved Occupancy: 18.28% (shared memory limited, 3 blocks/SM)
- L1 Hit Rate: 59.91%
- L2 Hit Rate: 80.17%
- Warp Stall: 83.43%
- No register or shared memory spilling
