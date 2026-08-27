#!/usr/bin/env bash
# level2_full_check.sh — Level 2 全量验收（唯一权威门）
# 用法：bash assets/lab4/opt/fitness/level2_full_check.sh <TASK:cpu|gpu> <CANDIDATE_DIR> [EVIDENCE_DIR]
# CPU: 40 步 + check.sh；GPU: 100 步 + check.sh（30 min 墙）。
set -euo pipefail
TASK="${1:?TASK cpu|gpu required}"
CAND="${2:?CANDIDATE_DIR required}"
EVID="${3:-./evidence/lvl2-$(date +%Y%m%d_%H%M%S)}"
HOST=$([ "$TASK" = cpu ] && echo arm || echo lab2)
FINAL=$([ "$TASK" = cpu ] && echo 40.0 || echo 100.0)
ANALYSIS=$([ "$TASK" = cpu ] && echo 1000.0 || echo 0.1)
CORES=$([ "$TASK" = cpu ] && echo "-c 60" || echo "")
PART=$([ "$TASK" = gpu ] && echo "-p lab4g10" || echo "")
mkdir -p "$EVID"

cat > "$EVID/full_job.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$CAND"
echo "=== 全量 $FINAL 步 + check.sh ==="
./build/ABE --twop-cache -F $FINAL -A $ANALYSIS 2>&1 | tee "$EVID/full_run.log"
echo "=== Program Cost ==="
grep -iE 'Program Cost|Total Evolve' "$EVID/full_run.log" | tail -3
echo "=== check.sh ==="
RESULT_DIR="$CAND/GW250118/AMSS_NCKU_output"
GOLDEN_DIR="$CAND/golden"
./check.sh "\$RESULT_DIR" "\$GOLDEN_DIR" 2>&1 | tee "$EVID/check_result.txt"
echo "LVL2_DONE"
EOF
chmod +x "$EVID/full_job.sh"
echo "Level 2 全量作业脚本：$EVID/full_job.sh"
echo "提交：hpc submit $CORES $PART -t 30m 'bash -s' < $EVID/full_job.sh"
echo "门：check.sh FINAL PASS + Trajectory RMS=0 + 约束 ≤2 + bit-exact vs baseline。"
