# Dispatch Table

## Final Decision: Always Use Fused (Candidate B)

After benchmarking both Candidate A (two-kernel with raw_qk) and Candidate B (fused single-kernel), Candidate B is universally faster across all 8 public cases. The dispatch has been simplified to always use the fused path.

## Path Selection Logic

```python
def _select_path(B, T, Hq, Hv):
    if os.environ.get("GDN_FORCE_FUSED").strip() == "1":
        return True
    if os.environ.get("GDN_FORCE_TWO_KERNEL").strip() == "1":
        return False
    return True  # default: fused
```

Environment variables `GDN_FORCE_FUSED=1` and `GDN_FORCE_TWO_KERNEL=1` allow manual override for testing.

## VALUE_TILE Selection

Always use VALUE_TILE=16. Testing with VALUE_TILE=8 showed 39-48% regression on long/wide cases because doubling the block count doubles the QK GEMM recomputation overhead.

```python
def _pick_value_tile(Hv, num_chunks):
    if VALUE_TILE in (8, 16):
        return VALUE_TILE
    return 16
```

## Why Fused Wins (Empirical Evidence)

| Shape Type | Example Case | Two-Kernel ms | Fused ms | Fused Advantage |
|---|---|---|---|---|
| Short (B small, T < 2K) | short_tail_state | 0.474 | 0.392 | 1.21x |
| Equal Hv/Hq=1, T=8K | chain_equal | 1.920 | 1.326 | 1.45x |
| Equal Hv/Hq=1, T=2K | parallel_equal | 1.595 | 1.397 | 1.14x |
| GVA Hv/Hq=4, T=2K | parallel_gva | 1.485 | 1.417 | 1.05x |
| Long seq Hv/Hq=4 | long_low_gva | 14.899 | 10.675 | 1.40x |
| Batch Hv/Hq=4 | batch_split_gva | 9.939 | 9.908 | 1.00x |
| Wide Hv=64 | wide_gva_state | 20.222 | 19.045 | 1.06x |
| Deep T=16K | deep_gva_state | 20.117 | 20.028 | 1.00x |

The fused path's advantage is largest when the DRAM traffic of the precomputed raw_qk most exceeds the cost of recomputing QK via Tensor Core GEMM. The only cases where the advantage is marginal are those with very high Hv/Hq ratios (4) combined with large batch or very deep sequences, where QK recomputation overhead partially offsets the DRAM savings.

## Initial Shape-Aware Dispatch (Pre-Benchmark Hypothesis)

The original shape-aware dispatch was based on the hypothesis that high Hv/Hq ratios would make QK recomputation too expensive. This hypothesis was PROVEN WRONG by empirical data - the GPU's Tensor Core throughput (tens of TFLOPS) easily handles the extra GEMM, while DRAM bandwidth (2 TB/s) remains the bottleneck.

The original dispatch rules are preserved in the code history but are no longer active. They were:
- Short sequences (num_chunks ≤ 18): fused
- High GVA ratio (Hv/Hq ≥ 4): two-kernel
- Very long sequences with low GVA ratio: fused
- Default: two-kernel
