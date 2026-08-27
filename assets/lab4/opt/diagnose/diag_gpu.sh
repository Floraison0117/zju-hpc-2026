#!/usr/bin/env bash
# diag_gpu.sh — 任务二 ABEGPU 诊断（KernelPro Stage-1 roofline + Stage-2 ncu/SASS）
# 主动式：按瓶颈类跑 ncu + nvcc -Xptxas -v，翻译成自然语言指令。
#
# 用法（本地仓库根，经 sshz.sh 在 lab2 上执行）：
#   bash assets/lab4/opt/diagnose/diag_gpu.sh <CANDIDATE_DIR> [EVIDENCE_DIR]
#   CANDIDATE_DIR  远端候选源目录（如 ~/lab4-gpu-cand）
#   EVIDENCE_DIR   证据输出目录（可选，默认 ./evidence/diag-gpu-<ts>）
#
# 前提：CANDIDATE_DIR 已 compile.sh 构建；剖析用 2 步短跑（非全量）。
# 注意：ncu 需在 hpc submit -p lab4g10 作业内跑。
set -euo pipefail

CAND="${1:?CANDIDATE_DIR required}"
EVID="${2:-./evidence/diag-gpu-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$EVID"

cat > "$EVID/diag_job.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$CAND"
BIN="build/ABE"
# 2 步短跑（缓存 twop），用于 ncu profiling
RUN_ARGS="--twop-cache -F 2 -A 1000"

echo "=== Stage-1: nvcc -Xptxas -v（寄存器/spill）==="
# 重新编译带 ptxas verbose，抓 rhs_kernel 的 regs/spill
(cd build && cmake --build . -- VERBOSE=1 2>&1 | grep -iE 'ptxas|rhs_kernel|regs|spill|stack' > "$EVID/ptxas_verbose.txt" || true)

echo "=== Stage-1: ncu roofline（compute/memory bound 分类）==="
ncu --set roofline --target-processes all -k regex:rhs_kernel \\
  --csv --unit auto \\
  \$BIN \$RUN_ARGS 2> "$EVID/ncu_roofline.txt" || true

echo "=== Stage-2: ncu full（occupancy/stall/scoreboard）==="
ncu --set full --target-processes all -k regex:rhs_kernel \\
  --section LaunchStats --section Occupancy --section WarpStateStats --section SchedulerStats --section MemoryWorkloadAnalysis \\
  \$BIN \$RUN_ARGS > "$EVID/ncu_full.txt" 2>&1 || true

echo "=== Stage-2: SASS（load:FP 比，需先 cuobjdump）==="
SASS_OBJ=\$(find build -name '*.cubin' | head -1)
[ -n "\$SASS_OBJ" ] && cuobjdump -sass "\$SASS_OBJ" > "$EVID/sass.txt" 2>/dev/null || true
echo "DONE"
EOF
chmod +x "$EVID/diag_job.sh"

# 指令合成
cat > "$EVID/synthesize_directives.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
E="${1:-.}"
{
echo "# GPU 诊断指令（semantic feedback）"
echo
echo "## Stage-1 瓶颈分类"
echo "- ptxas_verbose.txt：rhs_kernel regs/spill（基线 128 regs/25% 占用率/spill 688B）"
echo "- ncu_roofline.txt：compute/memory bound"
echo
echo "## Stage-2 微剖析"
echo "- ncu_full.txt：occupancy、warp stall reasons、L1TEX scoreboard stall（基线 46.6%）、No Eligible（基线 64.7%）"
echo "- sass.txt：load:FP 比、跨 TU 调用边界"
echo
echo "## 行动建议（按 kb/patterns/gpu-kernel.md）"
echo "1. 若 latency-bound + 跨 TU 调用 + scoreboard stall 高（G1）：考虑 split-with-inline-cuh（context.md Section B）。"
echo "   **重试前必读 kb/search-memory.md §15.4**：rhs split v1 实测 -3%，inline-.cuh 单核 spill 4.3× 恶化。"
echo "2. 若 memory-bound（G2）：shared-mem stencil 仅在 inline 之后可行，且只 tile 5-6 场。"
echo "3. Level 0 → Level 1（2 步短跑 A/B）→ Level 2（100 步 + check.sh，30 min 墙）。"
echo "4. 无论成败回写 kb/search-memory.md。"
} > "$E/directives.md"
cat "$E/directives.md"
EOF
chmod +x "$EVID/synthesize_directives.sh"

echo "诊断作业脚本已生成：$EVID/diag_job.sh"
echo "合成脚本已生成：$EVID/synthesize_directives.sh"
echo
echo "下一步（lab2 lab4g10 作业内）："
echo "  hpc submit -p lab4g10 'bash -s' < $EVID/diag_job.sh"
echo "  bash $EVID/synthesize_directives.sh $EVID"
echo "证据目录：$EVID"
