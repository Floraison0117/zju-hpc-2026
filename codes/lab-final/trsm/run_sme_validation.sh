#!/bin/bash
set -euo pipefail

TRSM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$TRSM_DIR"
if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "run_sme_validation.sh must run on the aarch64 SME compute node" >&2
    exit 2
fi

OUT="${SME_BUILD_DIR:-.sme-validation-build}"
bash ./build_sme_microbench.sh "$OUT"

echo "===== independent in-place correctness ====="
for threads in 1 2 4 8 16 24 32 38; do
    echo "--- OMP_NUM_THREADS=$threads ---"
    OMP_NUM_THREADS="$threads" OMP_PROC_BIND=close OMP_PLACES=cores \
        "$OUT/sme_update_test"
done

echo "===== persistent/double-buffer measurement (five repeats) ====="
for repeat in 1 2 3 4 5; do
    echo "--- repeat=$repeat ---"
    OMP_NUM_THREADS=38 OMP_PROC_BIND=close OMP_PLACES=cores \
        "$OUT/sme_pipeline_bench" 16800 512 224 4
done
