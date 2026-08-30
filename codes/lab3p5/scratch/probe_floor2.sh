#!/bin/bash
# scratch: clean floor probes — each probe profiled 5 times, medians reported.
# Uses the ladder kernel (probe codes via LAB35_PROBE) or the installed minimal
# kernel. Usage: hpc submit -p lab3p5 bash scratch/probe_floor2.sh
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
echo "[probe] kernel: $KERNEL_NAME"

run5() {  # $1 = probe code or "none"
    local vals=()
    for i in 1 2 3 4 5; do
        local OUT="prof_probe_run"
        rm -rf "$ROOT/$OUT"
        if [ "$1" = "none" ]; then
            msprof op --kernel-name="$KERNEL_NAME" --warm-up=10 --launch-count=1 \
                --output="$ROOT/$OUT" python3 checker/test_op.py --profile 2 >/dev/null 2>&1 || true
        else
            LAB35_PROBE="$1" msprof op --kernel-name="$KERNEL_NAME" --warm-up=10 --launch-count=1 \
                --output="$ROOT/$OUT" python3 checker/test_op.py --profile 2 >/dev/null 2>&1 || true
        fi
        vals+=("$(python3 checker/get_time.py "$ROOT/$OUT" 2>/dev/null || echo ERR)")
    done
    printf '%s\n' "${vals[@]}" | tr '\n' ' '
    echo ""
}

for P in none -100 -101 -105 -109; do
    echo -n "PROBE $P samples: "
    run5 "$P"
done
