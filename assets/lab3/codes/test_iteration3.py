import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.support import Case, assert_close, load_cases, make_inputs
from preprocessing.tilelang_cumsum import chunk_local_cumsum
from preprocessing.tilelang_kkt_solve import kkt_solve
from references.torch_gdr import ref_chunk_gated_delta_rule
from student.tilelang_fwd import (
    CHUNK_SIZE,
    HEAD_DIM,
    compute_raw_qk,
    gdn_prefill_forward,
    persistent_forward,
)


def run_persistent(
    case: Case,
    async_a: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = make_inputs(case)
    g_cumsum = chunk_local_cumsum(inputs.g)
    A = kkt_solve(inputs.k, g_cumsum, inputs.beta)
    raw_qk = compute_raw_qk(inputs.q, inputs.k)
    state_shape = (
        case.batch_size,
        case.num_heads_v,
        HEAD_DIM,
        HEAD_DIM,
    )
    if inputs.initial_state is None:
        state = torch.zeros(state_shape, device="cuda", dtype=torch.float32)
    else:
        state = inputs.initial_state.clone()
    output = torch.empty(
        (
            case.batch_size,
            case.seqlen,
            case.num_heads_v,
            HEAD_DIM,
        ),
        device="cuda",
        dtype=torch.bfloat16,
    )
    persistent_forward(
        inputs.q,
        inputs.k,
        inputs.v,
        g_cumsum,
        inputs.beta,
        A,
        raw_qk,
        state,
        output,
        async_a=async_a,
    )
    return output, state


def test_component_case(case: Case) -> None:
    inputs = make_inputs(case)
    expected_output, expected_state = ref_chunk_gated_delta_rule(
        inputs.q,
        inputs.k,
        inputs.v,
        inputs.g,
        inputs.beta,
        inputs.initial_state,
    )
    sync_output, sync_state = run_persistent(case, async_a=0)
    async_output, async_state = run_persistent(case, async_a=1)
    assert_close(f"{case.name} sync output", sync_output, expected_output)
    assert_close(f"{case.name} sync state", sync_state, expected_state)
    assert_close(f"{case.name} async output", async_output, expected_output)
    assert_close(f"{case.name} async state", async_state, expected_state)
    assert_close(f"{case.name} async/sync output", async_output, sync_output)
    assert_close(f"{case.name} async/sync state", async_state, sync_state)
    print(f"component {case.name}: PASS")


def test_pre_update_output() -> None:
    q = torch.zeros((1, 1, 1, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    q[0, 0, 0, 0] = 1
    k[0, 0, 0, 0] = 1
    v[0, 0, 0, 0] = 3
    g_cumsum = torch.zeros((1, 1, 1), device="cuda", dtype=torch.float32)
    beta = torch.ones_like(g_cumsum)
    A = torch.zeros(
        (1, 1, 1, CHUNK_SIZE),
        device="cuda",
        dtype=torch.bfloat16,
    )
    A[0, 0, 0, 0] = 1
    initial_state = torch.zeros(
        (1, 1, HEAD_DIM, HEAD_DIM),
        device="cuda",
        dtype=torch.float32,
    )
    initial_state[0, 0, 0, 0] = 2
    expected_output = torch.zeros_like(v)
    expected_output[0, 0, 0, 0] = 3 * HEAD_DIM**-0.5

    for async_a in (0, 1):
        raw_qk = compute_raw_qk(q, k)
        state = initial_state.clone()
        output = torch.empty_like(v)
        persistent_forward(
            q,
            k,
            v,
            g_cumsum,
            beta,
            A,
            raw_qk,
            state,
            output,
            async_a=async_a,
        )
        assert_close(
            f"pre-update output async={async_a}",
            output,
            expected_output,
        )
        if state[0, 0, 0, 0].item() != 3:
            raise AssertionError("state update did not use the residual")
    print("component pre_update_output: PASS")


def test_public_case(case: Case) -> None:
    inputs = make_inputs(case)
    g_cumsum = chunk_local_cumsum(inputs.g)
    A = kkt_solve(inputs.k, g_cumsum, inputs.beta)
    expected_output, expected_state = ref_chunk_gated_delta_rule(
        inputs.q,
        inputs.k,
        inputs.v,
        inputs.g,
        inputs.beta,
        inputs.initial_state,
    )
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
    print(f"public {case.name}: PASS")


def main() -> None:
    component_cases = [
        Case("one_token_no_state", 1, 1, 1, 1, False),
        Case("full_equal_state", 1, 64, 2, 2, True),
        Case("full_to_tail_gva", 1, 65, 2, 8, False),
        Case("full_to_full_to_tail", 1, 129, 4, 16, True),
        Case("batch4_grouped", 4, 65, 2, 8, True),
    ]
    for case in component_cases:
        test_component_case(case)
    test_pre_update_output()

    cases_path = Path(__file__).resolve().parents[1] / "evaluation" / "cases.csv"
    for case in load_cases(cases_path):
        test_public_case(case)


if __name__ == "__main__":
    main()
