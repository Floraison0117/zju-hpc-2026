#!/usr/bin/env bash
# level1_short_ab.sh — Level 1 短跑 A/B（1-3 min，OFF/ON 同作业）
# 用法：bash assets/lab4/opt/fitness/level1_short_ab.sh <TASK:cpu|gpu> <CANDIDATE_DIR> [EVIDENCE_DIR]
# 生成 hpc submit 作业脚本：同节点跑 OFF（baseline）与 ON（候选），对比每步时间 + 输出字节。
set -euo pipefail
TASK="${1:?TASK cpu|gpu required}"
CAND="${2:?CANDIDATE_DIR required}"
EVID="${3:-./evidence/lvl1-$(date +%Y%m%d_%H%M%S)}"
HOST=$([ "$TASK" = cpu ] && echo arm || echo lab2)
STEPS=$([ "$TASK" = cpu ] && echo 5 || echo 2)
FINAL=$([ "$TASK" = cpu ] && echo 5 || echo 2)
CORES=$([ "$TASK" = cpu ] && echo "-c 60" || echo "")
PART=$([ "$TASK" = gpu ] && echo "-p lab4g10" || echo "")
mkdir -p "$EVID"

cat > "$EVID/ab_job.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
OUT="$EVID"
# BASELINE = 已部署提交包（~/lab4-cpu 或 ~/lab4-gpu）
BASE=$([ "$TASK" = cpu ] && echo ~/lab4-cpu || echo ~/lab4-gpu)

strip_header() { tail -n +2 "\$1"; }  # 去 '# File created on <ts>' 首行

echo "=== 构建 OFF（baseline）与 ON（candidate）==="
( cd "\$BASE"   && cmake --build build -- -j30 2>&1 | tail -1 )
( cd "$CAND"   && cmake --build build -- -j30 2>&1 | tail -1 )

echo "=== 短跑 OFF（baseline）==="
( cd "\$BASE" && ./build/ABE --twop-cache -F $FINAL -A 0.1 2>&1 | tee "\$OUT/off_run.log" ) || true
grep -E 'Computer used|step' "\$OUT/off_run.log" | tail -$STEPS > "\$OUT/off_steps.txt"
for f in bssn_BH bssn_psi4 bssn_ADMQs bssn_constraint; do
  [ -f "\$BASE/GW250118/AMSS_NCKU_output/\$f.dat" ] && strip_header "\$BASE/GW250118/AMSS_NCKU_output/\$f.dat" | sha256sum >> "\$OUT/off_hashes.txt"
done

echo "=== 短跑 ON（candidate）==="
( cd "$CAND" && ./build/ABE --twop-cache -F $FINAL -A 0.1 2>&1 | tee "\$OUT/on_run.log" ) || true
grep -E 'Computer used|step' "\$OUT/on_run.log" | tail -$STEPS > "\$OUT/on_steps.txt"
for f in bssn_BH bssn_psi4 bssn_ADMQs bssn_constraint; do
  [ -f "$CAND/GW250118/AMSS_NCKU_output/\$f.dat" ] && strip_header "$CAND/GW250118/AMSS_NCKU_output/\$f.dat" | sha256sum >> "\$OUT/on_hashes.txt"
done

echo "=== 对比 ==="
echo "--- 每步时间 ---"
paste "\$OUT/off_steps.txt" "\$OUT/on_steps.txt"
echo "--- bit-exact（OFF vs ON 哈希应一致）---"
diff "\$OUT/off_hashes.txt" "\$OUT/on_hashes.txt" && echo "BIT-EXACT: IDENTICAL" || echo "BIT-EXACT: DIFFER（须排查）"
echo "LVL1_DONE"
EOF
chmod +x "$EVID/ab_job.sh"
echo "Level 1 A/B 作业脚本：$EVID/ab_job.sh"
echo "提交：hpc submit $CORES $PART -t 10m 'bash -s' < $EVID/ab_job.sh"
