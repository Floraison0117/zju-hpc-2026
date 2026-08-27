#!/usr/bin/env bash
# snapshot.sh — 部署前快照（AGENTS.md：正式写前校验 hash + 可恢复快照）
# 用法：bash assets/lab4/opt/search/snapshot.sh <TASK:cpu|gpu> [LABEL]
set -euo pipefail
TASK="${1:?TASK cpu|gpu required}"
LABEL="${2:-snap-$(date +%Y%m%d_%H%M%S)}"
HOST=$([ "$TASK" = cpu ] && echo arm || echo lab2 )
SUBMIT=$([ "$TASK" = cpu ] && echo ~/lab4-cpu || echo ~/lab4-gpu )
EVID="assets/lab4/opt/snapshots"
mkdir -p "$EVID"

cat > "$EVID/snap_${TASK}_${LABEL}.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
SNAP="${LABEL}"
echo "=== 校验当前 formal 提交包 hash（防漂移）==="
cd $SUBMIT
sha256sum CMakeLists.txt src/*.C src/*.cu src/*.f90 src/*.h 2>/dev/null | sort > /tmp/cur_hashes.txt
echo "当前 hash 已记录到 /tmp/cur_hashes.txt"
echo
echo "=== 创建可恢复快照 ==="
cd ~
tar czf "lab4-${TASK}-snapshot-\${SNAP}.tar.gz" \$(basename $SUBMIT) 2>/dev/null || cp -r $SUBMIT "lab4-${TASK}-snapshot-\${SNAP}"
echo "快照：~/lab4-${TASK}-snapshot-\${SNAP}（可回滚）"
echo
echo "=== 基线 OJ 成绩（回滚参照）==="
echo "CPU: 408.9s / 86 分 ｜ GPU: 1044.26s / 64 分"
EOF
chmod +x "$EVID/snap_${TASK}_${LABEL}.sh"
echo "快照作业脚本：$EVID/snap_${TASK}_${LABEL}.sh"
echo "执行：bash tmp/sshz.sh $HOST 'bash -s' < $EVID/snap_${TASK}_${LABEL}.sh"
