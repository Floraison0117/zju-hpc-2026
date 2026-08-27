#!/bin/bash
set -euo pipefail

TRSM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$TRSM_DIR"
if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "run_sme_candidate_cases.sh must run on the aarch64 SME compute node" >&2
    exit 2
fi

bash ./build_sme_candidates.sh
RUNS="${TRSM_CANDIDATE_RUNS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-38}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

for candidate in case2 case3; do
    echo "===== candidate=$candidate runs=$RUNS ====="
    numactl -N 1 "./trsm_test_${candidate}" 512 19968 "$RUNS"
    numactl -N 1 "./trsm_test_${candidate}" 2432 17024 "$RUNS"
    numactl -N 1 "./trsm_test_${candidate}" 17024 512 "$RUNS"
done
