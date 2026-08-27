#!/usr/bin/env bash
# search_loop.sh — 图搜索驱动（KernelEvolve `(F, π_sel, O, τ)`）
# greedy：选 F 最高已验证节点 → 应用单一变量变换 → 三级验证 → fitness → 回写 search-memory。
# 用法：bash assets/lab4/opt/search/search_loop.sh <TASK:cpu|gpu> [MAX_ITERS] [BUDGET_MIN]
set -euo pipefail
TASK="${1:?TASK cpu|gpu required}"
MAX="${2:-3}"
BUDGET="${3:-30}"
HOST=$([ "$TASK" = cpu ] && echo arm || echo lab2 )
TREE="assets/lab4/opt/search/tree"
mkdir -p "$TREE"

echo "=== Lab4 $TASK 闭环优化搜索（KernelPro + KernelEvolve 范式）==="
echo "策略：greedy（选 F 最高节点扩展）｜墙钟预算 ${BUDGET}min｜最大 $MAX 轮"
echo "硬约束：一次只改一个变量｜bit-exact｜check.sh PASS｜formal 只读"
echo

# 起始：当前已部署栈（F=1.0 baseline）
[ -f "$TREE/baseline.md" ] || cat > "$TREE/baseline.md" <<'EOF'
- id: baseline
- parent: root
- F: 1.0
- is_buggy: 0
- lever: 当前已部署栈（见 kb/search-memory.md「已部署栈」）
EOF

ITER=0
while [ "$ITER" -lt "$MAX" ]; do
  ITER=$((ITER+1))
  echo "--- 迭代 $ITER ---"
  # π_sel: 选 F 最高的非 buggy 节点
  PARENT=$(bash assets/lab4/opt/search/metadata_store.sh best 1 | head -1 | sed 's/.*tree\///;s/\.md//' || echo baseline)
  [ -z "$PARENT" ] && PARENT=baseline
  echo "选中父节点：$PARENT"

  # O（universal operator）：由 agent（LLM）在此注入候选变换
  # 人类/agent 填写本轮单一变量变换描述
  read -r -p "本轮单一变量变换 lever（描述，或输入 skip 结束）: " LEVER
  [ "$LEVER" = skip ] && break
  NODE="iter${ITER}_$(echo "$LEVER" | tr ' /' '__' | cut -c1-30)"

  # 1. Level 0 静态检查
  echo "[1/3] Level 0 静态检查..."
  bash assets/lab4/opt/fitness/level0_static.sh "$TASK" "<CANDIDATE_DIR>" "evidence/$NODE-lvl0"
  # 2. Level 1 短跑 A/B
  echo "[2/3] Level 1 短跑 A/B..."
  bash assets/lab4/opt/fitness/level1_short_ab.sh "$TASK" "<CANDIDATE_DIR>" "evidence/$NODE-lvl1"
  # 解析 A/B 结果填 T_REF/T_CAND/BIT/CHK（agent 或人工）
  echo "  → 填入 T_REF T_CAND BIT(0/1) CHK(0/1)"
  read -r -p "T_REF T_CAND BIT CHK: " T_REF T_CAND BIT CHK

  # fitness + 节点
  bash assets/lab4/opt/fitness/fitness.sh "$NODE" "$PARENT" "$T_REF" "$T_CAND" "$BIT" "$CHK" "$LEVER" "evidence/$NODE"

  # τ: 正收益且 bit-exact → Level 2 全量确认
  F=$(grep -oP '^- F\(speedup\): \K[\d.]+' "$TREE/$NODE.md")
  if awk "BEGIN{exit !($F>1.0)}" && [ "$BIT" = 1 ]; then
    echo "[3/3] Level 2 全量验收（正收益，需用户授权部署确认）..."
    echo "  bash assets/lab4/opt/fitness/level2_full_check.sh $TASK <CANDIDATE_DIR> evidence/$NODE-lvl2"
    echo "  通过后由主 agent 部署到 formal/OJ（候选不直接提交）"
  else
    echo "[3/3] 中性/负/失败 → 标注死路，回写 kb/search-memory.md"
  fi
done

echo
echo "=== 搜索树当前状态 ==="
bash assets/lab4/opt/search/metadata_store.sh list
echo
echo "最优 3："
bash assets/lab4/opt/search/metadata_store.sh best 3
