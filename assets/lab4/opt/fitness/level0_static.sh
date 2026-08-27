#!/usr/bin/env bash
# level0_static.sh — Level 0 静态检查（秒级，不占队列）
# 用法：bash assets/lab4/opt/fitness/level0_static.sh <TASK:cpu|gpu> <CANDIDATE_DIR> [EVIDENCE_DIR]
# 在远端（arm 或 lab2）经 sshz.sh 执行。
set -euo pipefail
TASK="${1:?TASK cpu|gpu required}"
CAND="${2:?CANDIDATE_DIR required}"
EVID="${3:-./evidence/lvl0-$(date +%Y%m%d_%H%M%S)}"
HOST=$([ "$TASK" = cpu ] && echo arm || echo lab2)
mkdir -p "$EVID"

cat > "$EVID/level0.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$CAND"
echo "=== sha256 候选关键文件 ==="
sha256sum CMakeLists.txt src/*.f90 src/*.C src/*.cu 2>/dev/null | sort > "$EVID/cand_hashes.txt"
cat "$EVID/cand_hashes.txt"

echo "=== 配置防漂移清单 ==="
grep -E 'Final_Evolution_Time|Analysis_Time|Dissipation|MPI_processes|OMP_threads|GPU_Calculation' AMSS_NCKU_Input.py || true
EXPECT_FINAL=$([ "$TASK" = cpu ] && echo 40.0 || echo 100.0)
EXPECT_MPI=$([ "$TASK" = cpu ] && echo 30 || echo 1)
EXPECT_OMP=$([ "$TASK" = cpu ] && echo 1 || echo 8)
echo "期望: Final=\$EXPECT_FINAL MPI=\$EXPECT_MPI OMP=\$EXPECT_OMP Dissipation=0.15"

echo "=== 预编译 diff（OFF vs formal，去行指令）==="
# 需要 formal 源路径
FORMAL=$([ "$TASK" = cpu ] && echo ~/HPC101/src/lab4-abe-cpu-opt || echo ~/lab4-gpu)
for f in src/bssn_rhs.f90 src/bssn_rhs_gpu.cu; do
  [ -f "\$FORMAL/\$f" ] && [ -f "\$CAND/\$f" ] && \\
    diff <(gfortran -cpp -E "\$FORMAL/\$f" 2>/dev/null | grep -v '^#') \\
         <(gfortran -cpp -E "\$CAND/\$f" 2>/dev/null | grep -v '^#') > "$EVID/diff_\$(basename \$f).txt" || true
done

echo "=== 向量化确认（CPU 任务）==="
if [ "$TASK" = cpu ]; then
  SRC=src/bssn_rhs.f90
  [ -f "\$SRC" ] && gfortran -cpp -fopt-info-vec-all -c \$SRC -J /tmp -o /dev/null 2> "$EVID/fopt_vec.txt" || true
  echo "bssn_rhs vec=\$(grep -c 'vectorized' "$EVID/fopt_vec.txt" 2>/dev/null || echo 0)"
fi
echo "=== nvcc ptxas（GPU 任务）==="
if [ "$TASK" = gpu ]; then
  grep -iE 'rhs_kernel|regs|spill' build/CMakeFiles/*.dir/*.cu.o.d 2>/dev/null | head || true
fi
echo "LVL0_DONE"
EOF
chmod +x "$EVID/level0.sh"
echo "Level 0 作业脚本：$EVID/level0.sh"
echo "执行：bash tmp/sshz.sh $HOST 'bash -s' < $EVID/level0.sh"
