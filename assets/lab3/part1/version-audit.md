# Version Audit

## Environment

| Item | Value |
|---|---|
| Hostname | `h3240101033-lab2` |
| GPU | NVIDIA A100 (CC 9.0), inferred from Nsight Compute profile |
| CUDA | 13.0 (from torch version in lab5 venv) |
| PyTorch | 2.13.0+cu130 (lab5 venv) |
| TileLang | 0.1.12 (lab5 venv) |
| Python | 3.12 / 3.13 |
| Nsight Compute | 2026.2.1.0 |

## Input Versions

| File | Size (bytes) | SHA-256 | Role |
|---|---|---|---|
| `gdn_prefill_forward.py` | 57,407 | `A05B5628...D8146174` | Current submission (baseline) |
| `tilelang_fwd.py.iter3_final` | 61,385 | `75CCF5F9...9F99F74F` | Iteration 3 (pingpong kernel) |
| `tilelang_fwd.py.iter4_final` | 14,880 | `233333A6...38DFC5A7` | Candidate B (fused single-kernel) |

## Architecture Comparison

### Candidate A: Two-Kernel with Precomputed raw_qk (current baseline)

**Kernels:**
1. `tilelang_raw_qk` – computes `Q @ K^T` via Tensor Core GEMM per `(batch, chunk, qk_head)`, writes FP32 intermediate `raw_qk` of shape `[B, num_chunks, Hq, 64, 64]` to DRAM.
2. `tilelang_persistent_tensorcore` – persistent kernel iterating over all chunks. Per chunk per `(value_head, value_tile)` block:
   - Loads k_shared, a_shared (A matrix), state_tile from DRAM
   - k@state via Tensor Core → residual
   - A@residual via Tensor Core → V_new (apply_ab)
   - Reads raw_qk from DRAM, applies gate scaling → a_shared
   - q@state via Tensor Core → output prefix
   - scores@V_new via Tensor Core → accumulate output
   - k^T@residual via Tensor Core → state update

**Key characteristics:**
- 5 Tensor Core GEMMs per chunk per block
- 1 separate QK GEMM kernel (launched once with B*num_chunks*Hq blocks)
- raw_qk FP32 DRAM: B * num_chunks * Hq * 64 * 64 * 4 bytes
- Shared memory per block: ~42.5 KB (k_shared 16KB + a_shared 8KB + state_tile 8KB + state_bf16 4KB + residual_vnew 4KB + residual_bf16 2KB + misc 0.5KB)
- Per-block DRAM raw_qk read per chunk: 64 * 64 * 4 = 16KB

### Candidate B: Single-Kernel Fused QK (iter4_final)

**Kernel:**
1. `tilelang_persistent_fused` – persistent kernel that computes QK on-the-fly. Per chunk per `(value_head, value_tile)` block:
   - Same k@state, A@residual as Candidate A
   - Instead of reading raw_qk: loads q_shared, computes `q @ k^T` via Tensor Core → qk_frag
   - Applies gate scaling to qk_frag → a_shared
   - Same q@state, scores@V_new, k^T@residual as Candidate A
   - No separate raw_qk kernel or DRAM intermediate

**Key characteristics:**
- 6 Tensor Core GEMMs per chunk per block (adds QK GEMM)
- 1 kernel launch total (persistent)
- Shared memory per block: ~58.5 KB (adds q_shared 16KB vs Candidate A)
- No raw_qk DRAM allocation or traffic
- QK GEMM recomputed Hv * value_tiles times per chunk (vs Hq times in Candidate A)
- Overhead factor: (Hv/Hq) * value_tiles redundant QK computations

### Candidate Iter3: Pingpong with Manual Scalar Loops (iter3_final, historical)

Uses `tilelang_persistent_pingpong` with async A prefetch (double-buffered) and manual serial loops instead of Tensor Core for inner computations. This version achieved ~12.9x worse performance than the current submission and is kept for historical reference only.
