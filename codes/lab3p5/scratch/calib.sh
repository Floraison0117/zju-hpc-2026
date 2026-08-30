#!/bin/bash
# scratch: clean interleaved A/B calibration of floor / tiling-read variants /
# full kernel, all in ONE binary and ONE job (cancels device drift).
# Probe codes (rowsPerChunk): 98 floor, 99 1-read, 5 direct tiling reads,
# 6 GET_TILING_DATA, 999 full kernel.
set -e
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/env.sh"
export PYTHONPATH="$ROOT:$ROOT/checker:${PYTHONPATH:-}"
if ! python3 -c "import custom_ops_lib" >/dev/null 2>&1; then bash checker/build.sh; source "$ROOT/env.sh"; fi
KERNEL_NAME=$(python3 - <<'PY'
import json,glob,os
f=sorted(glob.glob(os.path.expanduser('~/custom_opp/vendors/customize/op_impl/ai_core/tbe/kernel/ascend910b/fused_add_rms_norm/FusedAddRmsNorm_*.json')))[0]
print([e['kernelName'] for e in json.load(open(f))['kernelList']][0])
PY
)
echo "[calib] kernel: $KERNEL_NAME"

declare -A ACC
for round in 1 2 3 4 5; do
    for P in 105 7 8; do
        OUT="prof_calib"
        rm -rf "$ROOT/$OUT"
        if [ "$P" = "999" ]; then
            msprof op --kernel-name="$KERNEL_NAME" --warm-up=10 --launch-count=1 \
                --output="$ROOT/$OUT" python3 checker/test_op.py --profile 2 >/dev/null 2>&1 || true
        else
            LAB35_PROBE="$P" msprof op --kernel-name="$KERNEL_NAME" --warm-up=10 --launch-count=1 \
                --output="$ROOT/$OUT" python3 checker/test_op.py --profile 2 >/dev/null 2>&1 || true
        fi
        v=$(python3 checker/get_time.py "$ROOT/$OUT" 2>/dev/null || echo ERR)
        ACC[$P]="${ACC[$P]:-} $v"
    done
done
for P in 105 7 8; do
    echo "CODE $P samples:${ACC[$P]}"
done
