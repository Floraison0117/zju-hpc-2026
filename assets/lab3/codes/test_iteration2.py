import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.support import Case, assert_close, make_inputs
from preprocessing.tilelang_cumsum import chunk_local_cumsum
from preprocessing.tilelang_kkt_solve import kkt_solve
from references.torch_gdr import ref_chunk_gated_delta_rule
from student.tilelang_fwd import CHUNK_SIZE, compute_raw_qk, gdn_prefill_forward


def reference_raw_qk(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    batch_size, num_tokens, num_heads, head_dim = q.shape
    num_chunks = (num_tokens + CHUNK_SIZE - 1) // CHUNK_SIZE
    q_pad = torch.zeros(
        batch_size,
        num_chunks * CHUNK_SIZE,
        num_heads,
        head_dim,
        dtype=q.dtype,
        device=q.device,
    )
    k_pad = torch.zeros_like(q_pad)
    q_pad[:, :num_tokens] = q
    k_pad[:, :num_tokens] = k
    q_chunks = q_pad.view(
        batch_size, num_chunks, CHUNK_SIZE, num_heads, head_dim
    )
    k_chunks = k_pad.view_as(q_chunks)
    return torch.einsum("bnthd,bnshd->bnhts", q_chunks, k_chunks).float()


def test_raw_qk() -> None:
    torch.manual_seed(7)
    q = torch.randn(1, 65, 2, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    actual = compute_raw_qk(q, k)
    expected = reference_raw_qk(q, k)
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)
    print("raw_qk: PASS")


def test_case(case: Case) -> None:
    inputs = make_inputs(case)
    expected_output, expected_state = ref_chunk_gated_delta_rule(
        inputs.q,
        inputs.k,
        inputs.v,
        inputs.g,
        inputs.beta,
        inputs.initial_state,
    )
    g_cumsum = chunk_local_cumsum(inputs.g)
    A = kkt_solve(inputs.k, g_cumsum, inputs.beta)
    actual_output, actual_state = gdn_prefill_forward(
        inputs.q,
        inputs.k,
        inputs.v,
        g_cumsum,
        inputs.beta,
        A,
        inputs.initial_state,
    )
    assert_close(f"{case.name} output", actual_output, expected_output)
    assert_close(f"{case.name} state", actual_state, expected_state)
    print(f"{case.name}: PASS")


def main() -> None:
    test_raw_qk()
    test_case(Case("one_token", 1, 1, 1, 1, True))
    test_case(Case("full_chunk_gva", 1, 64, 2, 8, False))
    test_case(Case("tail_gva_state", 1, 65, 2, 8, True))


if __name__ == "__main__":
    main()
