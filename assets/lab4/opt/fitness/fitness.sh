#!/usr/bin/env bash
# fitness.sh — 计算 F = t_ref/t_candidate 并写节点元数据（KernelEvolve metadata store）
# 用法：bash assets/lab4/opt/fitness/fitness.sh <NODE_ID> <PARENT_ID> <T_REF> <T_CAND> <BITEXACT:0|1> <CHECKPASS:0|1> <LEVER_DESC> [EVIDENCE_DIR]
set -euo pipefail
NODE="${1:?NODE_ID required}"
PARENT="${2:-root}"
T_REF="${3:?T_REF (baseline 每步秒) required}"
T_CAND="${4:?T_CAND (候选每步秒) required}"
BIT="${5:?BITEXACT 0|1 required}"
CHK="${6:?CHECKPASS 0|1 required}"
LEVER="${7:?LEVER_DESC required}"
EVID="${8:-./evidence/fitness-$(date +%Y%m%d_%H%M%S)}"
TREE="assets/lab4/opt/search/tree"
mkdir -p "$TREE" "$EVID"

# F = t_ref / t_cand；正确性失败则 0
if [ "$BIT" != 1 ] || [ "$CHK" != 1 ]; then
  F=0; IS_BUGGY=1
else
  F=$(awk "BEGIN{printf \"%.4f\", $T_REF/$T_CAND}")
  IS_BUGGY=0
fi

# 节点元数据（KernelEvolve: id, parent, F, is_buggy, overview）
cat > "$TREE/$NODE.md" <<EOF
# 节点 $NODE

- id: $NODE
- parent: $PARENT
- lever: $LEVER
- t_ref: $T_REF
- t_cand: $T_CAND
- F(speedup): $F
- bit-exact: $BIT
- check.sh: $CHK
- is_buggy: $IS_BUGGY
- timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)
- evidence: $EVID

## overview

（profiling 摘要 + 优化建议回写处）

EOF

# 追加到搜索记忆账本（KernelPro search memory cross-iteration learning）
# 不直接改 search-memory.md 表格（fragile），而是写 ledger，由 agent/人工回写 markdown。
LEDGER="assets/lab4/opt/search/tree/ledger.tsv"
mkdir -p "$(dirname "$LEDGER")"
printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$NODE" "$PARENT" "$F" "$IS_BUGGY" "$BIT" "$LEVER" >> "$LEDGER"
echo "已记入账本 $LEDGER（请由 agent 回写 kb/search-memory.md 表格）"

echo "F=$F  is_buggy=$IS_BUGGY  node=$TREE/$NODE.md"
echo "verdict: $([ "$IS_BUGGY" = 0 ] && [ $(awk "BEGIN{print ($F>1.0)}") = 1 ] && echo '正收益，可进 Level 2 部署确认' || ([ "$IS_BUGGY" = 0 ] && echo '中性/负，标注死路' || echo '正确性失败，死路'))"
