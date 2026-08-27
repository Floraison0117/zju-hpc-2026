#!/usr/bin/env bash
# run_opt.sh — Lab4 闭环优化顶层入口（KernelPro + KernelEvolve 范式）
# 用法：bash assets/lab4/opt/run_opt.sh <TASK:cpu|gpu> [STEP] [CANDIDATE_DIR]
#   STEP: diagnose | search | snapshot | full
set -euo pipefail
TASK="${1:?TASK cpu|gpu required}"
STEP="${2:-diagnose}"
CAND="${3:-<CANDIDATE_DIR>}"

case "$TASK" in
cpu) HOST=arm;  KBHW="assets/lab4/kb/hardware/task1-arm-cpu.md";  PAT="assets/lab4/kb/patterns/cpu-stencil.md" ;;
gpu) HOST=lab2; KBHW="assets/lab4/kb/hardware/task2-a100-gpu.md"; PAT="assets/lab4/kb/patterns/gpu-kernel.md" ;;
*) echo "TASK must be cpu|gpu" >&2; exit 2 ;;
esac

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Lab4 $TASK 闭环优化（KernelPro 语义反馈 + KernelEvolve 图搜索） ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "硬件约束：$KBHW"
echo "模式库：$PAT"
echo "搜索记忆：assets/lab4/kb/search-memory.md"
echo "fitness/gates：assets/lab4/kb/fitness-gates.md"
echo

case "$STEP" in
diagnose)
  echo "[Stage-1 瓶颈分类 + Stage-2 微剖析 → 语义反馈指令]"
  if [ "$TASK" = cpu ]; then
    bash assets/lab4/opt/diagnose/diag_cpu.sh "$CAND" "$CAND/build"
  else
    bash assets/lab4/opt/diagnose/diag_gpu.sh "$CAND"
  fi
  ;;
search)
  echo "[图搜索循环：greedy 选节点 → 单一变量变换 → 三级验证 → fitness → 回写]"
  bash assets/lab4/opt/search/search_loop.sh "$TASK"
  ;;
snapshot)
  echo "[部署前快照 + hash 校验]"
  bash assets/lab4/opt/search/snapshot.sh "$TASK"
  ;;
full)
  echo "[全量验收门]"
  bash assets/lab4/opt/fitness/level2_full_check.sh "$TASK" "$CAND"
  ;;
*) echo "STEP: diagnose|search|snapshot|full" >&2; exit 2 ;;
esac
