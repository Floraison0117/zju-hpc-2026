# Comparison: Baseline vs Final (Part 1)

## Final SHA-256

`6491a240382413e287e771d000275b84acfb0ff786e0db60af3bda360a7221d5`

## Environment

| Item | Value |
|---|---|
| Partition | lab3 (H800 MIG 10G) |
| Node | m701.clusters.zjusct.io |
| Image | harbor.s.zjusct.io/public/hpc101-lab3:v26.2 |
| Python | 3.12.3 (/opt/lab3-venv) |
| Benchmark | warmup=10, repetitions=100, 3 independent rounds |

## Final Results (median of 3-round medians)

| case | baseline ms (online) | final ms | speedup | PASS |
|---|---|---|---|---|
| short_tail_state | 0.471888 | 0.392368 | 1.203x | PASS |
| chain_equal | 1.905648 | 1.326384 | 1.437x | PASS |
| parallel_equal | 1.621488 | 1.397360 | 1.160x | PASS |
| parallel_gva | 1.513760 | 1.416880 | 1.068x | PASS |
| long_low_gva | 14.942496 | 10.675264 | 1.400x | PASS |
| batch_split_gva | 9.977360 | 9.908400 | 1.007x | PASS |
| wide_gva_state | 20.351889 | 19.044656 | 1.069x | PASS |
| deep_gva_state | 20.258063 | 20.027599 | 1.012x | PASS |

**Geometric mean speedup: 1.159x**

## 3-Round Data

| case | Round 1 | Round 2 | Round 3 | Median | Variation |
|---|---|---|---|---|---|
| short_tail_state | 0.392368 | 0.394384 | 0.391600 | 0.392368 | 0.7% |
| chain_equal | 1.320800 | 1.326384 | 1.340000 | 1.326384 | 1.4% |
| parallel_equal | 1.404080 | 1.397360 | 1.382144 | 1.397360 | 1.6% |
| parallel_gva | 1.416880 | 1.420416 | 1.416384 | 1.416880 | 0.3% |
| long_low_gva | 10.676208 | 10.675264 | 10.591232 | 10.675264 | 0.9% |
| batch_split_gva | 10.044336 | 9.908400 | 9.868672 | 9.908400 | 1.7% |
| wide_gva_state | 19.175488 | 19.044656 | 18.955456 | 19.044656 | 1.2% |
| deep_gva_state | 20.537024 | 20.027599 | 19.933312 | 20.027599 | 3.0% |

Round-to-round variation is within 3% for all cases (deep_gva_state at 3.0% boundary).

## Candidate A vs Candidate B Comparison

| case | B (fused) ms | A (two-kernel) ms | B/A speedup |
|---|---|---|---|
| short_tail_state | 0.392 | 0.474 | 1.21x |
| chain_equal | 1.326 | 1.920 | 1.45x |
| parallel_equal | 1.397 | 1.595 | 1.14x |
| parallel_gva | 1.417 | 1.485 | 1.05x |
| long_low_gva | 10.675 | 14.899 | 1.40x |
| batch_split_gva | 9.908 | 9.939 | 1.00x |
| wide_gva_state | 19.045 | 20.222 | 1.06x |
| deep_gva_state | 20.028 | 20.117 | 1.00x |

Candidate B (fused single-kernel) is universally faster. The advantage is largest for chain_equal (1.45x) and long_low_gva (1.40x), and marginal for batch_split_gva and deep_gva_state (~1.00x). The decision to always use the fused path is empirically justified.

## Acceptance Criteria

| Criterion | Requirement | Result | Status |
|---|---|---|---|
| 8/8 PASS correctness | All must pass | 8/8 PASS | OK |
| 3 independent rounds | warmup=10, rep=100 x 3 | Rounds 1-3 complete | OK |
| Geometric mean ≥ 1.10x | vs baseline | 1.159x | OK |
| Max regression ≤ 3% | No case > 3% worse | 0% (all faster) | OK |
| short_tail_state ≤ 0.42ms | Target | 0.392ms | OK |
| 3+ of 4 long ≥ 1.10x | long_low_gva, batch_split_gva, wide_gva_state, deep_gva_state | Only 1/4 (long_low_gva) | FAIL |
| Profiler evidence | Explain bottlenecks | NCU data analyzed | OK |
| SHA-256 identical | Both locations | `6491a240...7221d5` | OK |

## Stage Status: NEEDS_NEXT_STAGE

Criteria 1-5 and 7-8 are met. Criterion 6 (3+ of 4 long cases ≥ 1.10x) is not met:
- long_low_gva: 1.400x (OK)
- batch_split_gva: 1.007x (needs 1.10x)
- wide_gva_state: 1.069x (needs 1.10x)
- deep_gva_state: 1.012x (needs 1.10x)

The bottleneck for wide/deep cases is that the fused path's QK recomputation overhead roughly balances the DRAM traffic savings. Further improvements require kernel-level optimizations (see next-stage.md).
