#!/usr/bin/env bash
# diag_cpu.sh — 任务一 ABE CPU 诊断（KernelPro Stage-1 瓶颈分类 + Stage-2 微剖析）
# 主动式（deterministic）：按瓶颈类跑全部相关工具，把原始指标翻译成自然语言指令。
#
# 用法（在本地仓库根，经 sshz.sh 在 arm 上执行）：
#   bash assets/lab4/opt/diagnose/diag_cpu.sh <CANDIDATE_DIR> <BUILD_DIR> [EVIDENCE_DIR]
#   CANDIDATE_DIR  远端候选源目录（如 ~/lab4-cpu-cand）
#   BUILD_DIR      构建目录（如 <CANDIDATE_DIR>/build）
#   EVIDENCE_DIR   证据输出目录（可选，默认 ./evidence/diag-<ts>）
#
# 前提：BUILD_DIR 已 cmake 构建（compile.sh），TwoPuncture 缓存已生成（--twop-cache）。
# 注意：perf/ncu 需在 hpc submit 作业内跑（非登录节点）。本脚本生成作业脚本并提示提交。
set -euo pipefail

CAND="${1:?CANDIDATE_DIR required}"
BUILD="${2:?BUILD_DIR required}"
EVID="${3:-./evidence/diag-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$EVID"

cat > "$EVID/diag_job.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$CAND"
BIN="$BUILD/ABE"
# 2 步演化-only（缓存 twop），用于 profiling
RUN_ARGS="--twop-cache -F 2 -A 1000"

echo "=== Stage-1: perf stat（宏尺度：IPC/miss/branch）==="
perf stat -e cycles,instructions,cache-misses,cache-references,branch-misses,branches \
  \$BIN \$RUN_ARGS 2> "$EVID/perf_stat.txt" || true
# 解析 IPC
IPC=\$(awk '/instructions/{ins=\$1} /cycles/{cyc=\$1} END{if(cyc>0) printf "%.2f", ins/cyc}' "$EVID/perf_stat.txt")
echo "IPC=\$IPC"

echo "=== Stage-1: perf record flat（函数尺度：热点占比）==="
perf record -F 999 -g -- \$BIN \$RUN_ARGS 2>/dev/null || true
perf report --stdio --no-children -n 2>/dev/null | head -40 > "$EVID/perf_flat.txt" || true

# 瓶颈分类
CM=\$(awk '/cache-misses/{m=\$1} /cache-references/{r=\$1} END{if(r>0) printf "%.3f", m/r}' "$EVID/perf_stat.txt")
echo "cache_miss_rate=\$CM"

# 拆解 compute_rhs_bssn_ 反汇编（load:FP 比）
echo "=== Stage-2: objdump compute_rhs_bssn_（指令尺度）==="
SO=\$(find "$BUILD" -name '*.o' | xargs -r nm 2>/dev/null | awk '/compute_rhs_bssn_/&&/ T /{print \$3; exit}')
if [ -n "\$SO" ]; then
  OBJ=\$(find "$BUILD" -name "\$(basename \${SO%.T*}).o" | head -1)
  [ -n "\$OBJ" ] && objdump -d "\$OBJ" > "$EVID/objdump_rhs.txt" 2>/dev/null || true
  LOADS=\$(grep -cE '^\s+[0-9a-f]+:\s+.*(ldr|str)' "$EVID/objdump_rhs.txt" 2>/dev/null || echo 0)
  FMLA=\$(grep -cE 'fmla|fmul|fdiv' "$EVID/objdump_rhs.txt" 2>/dev/null || echo 0)
  echo "rhs loads=\$LOADS fp=\$FMLA load:fp=\$(awk "BEGIN{if(\$FMLA>0)printf \"%.2f\",\$LOADS/\$FMLA; else print \"inf\"}")"
fi

# 向量化报告
echo "=== Stage-2: fopt-info（循环尺度：向量化/missed）==="
SRC=\$(find "$CAND/src" -name 'bssn_rhs.f90' | head -1)
if [ -n "\$SRC" ]; then
  gfortran -cpp -fopt-info-vec-all -c \$SRC -J /tmp -o /dev/null 2> "$EVID/fopt_info.txt" || true
  echo "bssn_rhs vec=\$(grep -c 'vectorized' "$EVID/fopt_info.txt") missed=\$(grep -c 'missed' "$EVID/fopt_info.txt")"
fi
echo "DONE"
EOF
chmod +x "$EVID/diag_job.sh"

# 指令合成（语义反馈算子：把指标翻译成中文指令）
cat > "$EVID/synthesize_directives.sh" <<'EOF'
#!/usr/bin/env bash
# 读 Stage-1/2 原始指标，按 kb/patterns/cpu-stencil.md 合成自然语言指令。
set -euo pipefail
E="${1:-.}"
IPC=$(awk '/IPC=/{print $2}' "$E/perf_stat.txt" 2>/dev/null || echo "NA")
CM=$(awk '/cache_miss_rate=/{print $2}' "$E/perf_stat.txt" 2>/dev/null || echo "NA")
{
echo "# CPU 诊断指令（semantic feedback）"
echo
echo "## Stage-1 瓶颈分类"
echo "- IPC=$IPC（峰值~4.0；<1.8 提示 load/issue-bound）"
echo "- cache_miss_rate=$CM（<2% 非带宽受限）"
if [ "$IPC" != "NA" ] && awk "BEGIN{exit !($IPC<1.8)}"; then
  echo "- 分类：**load/issue-bound（数据依赖）**，非带宽。"
  echo "- 处方：load:FP 高是 BSSN 80+ 3D 数组 stencil 固有。**勿重试**显式循环重写/oinv（search-memory #4/#14 已证 0/负）。"
  echo "- 可探索：编译 flag 叠加（但 search-memory #20-27 已穷尽 FP-safe flag，-fno-tree-loop-distribute-patterns -ftree-loop-im 已部署）。"
  echo "- 真正未破的杠杆是 compute_rhs 算法级重写（978 行，高风险，lab 禁止未验证低精度替换，需用户授权）。"
else
  echo "- 分类：需结合 perf_flat 热点进一步判断。"
fi
echo
echo "## Stage-2 微剖析（已收集原始数据）"
echo "- perf_flat.txt：热点函数占比（compute_rhs_bssn_ 应 ~30%）"
echo "- objdump_rhs.txt：load:FP 比（应 ~3.4:1）"
echo "- fopt_info.txt：向量化/missed 计数（bssn_rhs 应 vec~194/missed~958，Ricci 已 16B NEON）"
echo
echo "## 行动建议"
echo "1. 读 assets/lab4/kb/search-memory.md 确认候选杠杆是否已测。"
echo "2. 读 assets/lab4/kb/patterns/cpu-stencil.md 取匹配模式。"
echo "3. 若新杠杆：Level 0 → Level 1 短跑 A/B →（通过）Level 2 全量 check.sh。"
echo "4. 无论成败回写 search-memory.md。"
} > "$E/directives.md"
cat "$E/directives.md"
EOF
chmod +x "$EVID/synthesize_directives.sh"

echo "诊断作业脚本已生成：$EVID/diag_job.sh"
echo "合成脚本已生成：$EVID/synthesize_directives.sh"
echo
echo "下一步（在 arm 作业内执行）："
echo "  bash tmp/sshz.sh arm 'bash -s' < $EVID/diag_job.sh   # 或提交 hpc 作业跑"
echo "  bash $EVID/synthesize_directives.sh $EVID            # 合成中文指令"
echo
echo "证据目录：$EVID（perf_stat/perf_flat/objdump_rhs/fopt_info/directives.md）"
