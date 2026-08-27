#!/usr/bin/env bash
# Milestone C round 2 L0 probe: single-TU ptxas (base/patched/int) for the
# three interpolation kernels (restrict3 / global_interp / sommerfeld_rout).
#   GATE per kernel: int static <= 0.80*base static AND int regs <= base regs.
# No full build here (one build per A/B job, 30-min wall-clock discipline).
set -uo pipefail
BASE=/home/h3240101033/lab4-gpu
R3=$(cat ~/.p24_r3_path 2>/dev/null)
GI=$(cat ~/.p24_gi_path 2>/dev/null)
SOM=$(cat ~/.p24_som_path 2>/dev/null)
TS=$(date -u +%Y%m%d-%H%M%S)
EV="${R3:-/tmp}/evidence/p24-l0-probe-$TS"
mkdir -p "$EV"
exec > "$EV/job.log" 2>&1
echo "=== START $(date -u) ==="; hostname
nvidia-smi -L 2>/dev/null | head -2
echo "BASE=$BASE R3=$R3 GI=$GI SOM=$SOM EV=$EV"

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
NVCC=/usr/local/cuda-13.3/bin/nvcc
CU=$(ls /usr/local/cuda-13.3/bin/cuobjdump 2>/dev/null || echo /usr/local/cuda/bin/cuobjdump)
FLAGS="-arch=sm_80 -O3 -rdc=true -lineinfo -DUSE_GPU -Dfortran3 -Dnewc -DMPI_CUDA_AWARE=0"

count_sass() { # obj kernel_regex
  $CU -sass "$1" 2>/dev/null | sed -n "/Function : .*$2/,/Function :/p" | grep -cE "^\s+/\*[0-9a-f]+\*/"
}
ptxas_line() { # nvcc_log tag
  grep -A2 "Function properties for .*$2" "$1" 2>/dev/null | grep -E "$3" | head -2
}

# ============================================================
echo "=== L0 restrict3 (prolongrestrict_cell_gpu) ==="
( cd "$BASE/src" && $NVCC $FLAGS -Xptxas -v -c prolongrestrict_cell_gpu.cu     -o "$EV/r3_base.o"     > "$EV/r3_base.nvcc.log"     2>&1 ); echo "r3 base rc=$?"
( cd "$R3/src"   && $NVCC $FLAGS -Xptxas -v -c prolongrestrict_cell_gpu.cu     -o "$EV/r3_patched.o" > "$EV/r3_patched.nvcc.log"  2>&1 ); echo "r3 patched rc=$?"
( cd "$R3/src"   && $NVCC $FLAGS -Xptxas -v -c prolongrestrict_cell_gpu_int.cu -o "$EV/r3_int.o"     > "$EV/r3_int.nvcc.log"      2>&1 ); echo "r3 int rc=$?"
for tag in base patched int; do
  echo "--- r3 $tag restrict3_kernel ptxas ---"
  [ "$tag" = int ] && FN='restrict3_kernel_int' || FN='restrict3_kernel[^_]'
  ptxas_line "$EV/r3_$tag.nvcc.log" "$FN" "bytes stack frame|Used [0-9]+ registers|spill stores"
done
R3_BASE=$(count_sass "$EV/r3_base.o" 'restrict3_kernel[^_]')
R3_INT=$(count_sass "$EV/r3_int.o" 'restrict3_kernel_int')
R3_PAT=$(count_sass "$EV/r3_patched.o" 'restrict3_kernel[^_]')
R3_BASE_R=$(grep -A2 "Function properties for .*restrict3_kernel[^_]" "$EV/r3_base.nvcc.log" | grep -oE "Used [0-9]+ registers" | head -1 | grep -oE "[0-9]+")
R3_INT_R=$(grep -A2 "Function properties for .*restrict3_kernel_int" "$EV/r3_int.nvcc.log" | grep -oE "Used [0-9]+ registers" | head -1 | grep -oE "[0-9]+")
echo "r3 static: base=$R3_BASE patched=$R3_PAT int=$R3_INT | regs base=$R3_BASE_R int=$R3_INT_R"

# ============================================================
echo "=== L0 global_interp (fmisc_gpu) ==="
( cd "$BASE/src" && $NVCC $FLAGS -Xptxas -v -c fmisc_gpu.cu     -o "$EV/gi_base.o"     > "$EV/gi_base.nvcc.log"     2>&1 ); echo "gi base rc=$?"
( cd "$GI/src"   && $NVCC $FLAGS -Xptxas -v -c fmisc_gpu.cu     -o "$EV/gi_patched.o" > "$EV/gi_patched.nvcc.log"  2>&1 ); echo "gi patched rc=$?"
( cd "$GI/src"   && $NVCC $FLAGS -Xptxas -v -c fmisc_gpu_int.cu -o "$EV/gi_int.o"     > "$EV/gi_int.nvcc.log"      2>&1 ); echo "gi int rc=$?"
for tag in base patched int; do
  echo "--- gi $tag global_interp_kernel ptxas ---"
  [ "$tag" = int ] && FN='global_interp_kernel_int' || FN='global_interp_kernel[^_]'
  ptxas_line "$EV/gi_$tag.nvcc.log" "$FN" "bytes stack frame|Used [0-9]+ registers|spill stores"
done
GI_BASE=$(count_sass "$EV/gi_base.o" 'global_interp_kernel[^_]')
GI_INT=$(count_sass "$EV/gi_int.o" 'global_interp_kernel_int')
GI_PAT=$(count_sass "$EV/gi_patched.o" 'global_interp_kernel[^_]')
GI_BASE_R=$(grep -A2 "Function properties for .*global_interp_kernel[^_]" "$EV/gi_base.nvcc.log" | grep -oE "Used [0-9]+ registers" | head -1 | grep -oE "[0-9]+")
GI_INT_R=$(grep -A2 "Function properties for .*global_interp_kernel_int" "$EV/gi_int.nvcc.log" | grep -oE "Used [0-9]+ registers" | head -1 | grep -oE "[0-9]+")
echo "gi static: base=$GI_BASE patched=$GI_PAT int=$GI_INT | regs base=$GI_BASE_R int=$GI_INT_R"

# ============================================================
echo "=== L0 sommerfeld_rout (sommerfeld_rout_gpu) ==="
( cd "$BASE/src" && $NVCC $FLAGS -Xptxas -v -c sommerfeld_rout_gpu.cu     -o "$EV/som_base.o"     > "$EV/som_base.nvcc.log"     2>&1 ); echo "som base rc=$?"
( cd "$SOM/src"  && $NVCC $FLAGS -Xptxas -v -c sommerfeld_rout_gpu.cu     -o "$EV/som_patched.o" > "$EV/som_patched.nvcc.log"  2>&1 ); echo "som patched rc=$?"
( cd "$SOM/src"  && $NVCC $FLAGS -Xptxas -v -c sommerfeld_rout_gpu_int.cu -o "$EV/som_int.o"     > "$EV/som_int.nvcc.log"      2>&1 ); echo "som int rc=$?"
for tag in base patched int; do
  echo "--- som $tag sommerfeld_rout_kernel ptxas ---"
  [ "$tag" = int ] && FN='sommerfeld_rout_kernel_int' || FN='sommerfeld_rout_kernel[^_]'
  ptxas_line "$EV/som_$tag.nvcc.log" "$FN" "bytes stack frame|Used [0-9]+ registers|spill stores"
done
SOM_BASE=$(count_sass "$EV/som_base.o" 'sommerfeld_rout_kernel[^_]')
SOM_INT=$(count_sass "$EV/som_int.o" 'sommerfeld_rout_kernel_int')
SOM_PAT=$(count_sass "$EV/som_patched.o" 'sommerfeld_rout_kernel[^_]')
SOM_BASE_R=$(grep -A2 "Function properties for .*sommerfeld_rout_kernel[^_]" "$EV/som_base.nvcc.log" | grep -oE "Used [0-9]+ registers" | head -1 | grep -oE "[0-9]+")
SOM_INT_R=$(grep -A2 "Function properties for .*sommerfeld_rout_kernel_int" "$EV/som_int.nvcc.log" | grep -oE "Used [0-9]+ registers" | head -1 | grep -oE "[0-9]+")
echo "som static: base=$SOM_BASE patched=$SOM_PAT int=$SOM_INT | regs base=$SOM_BASE_R int=$SOM_INT_R"

# ============================================================
echo "=== GATE (int static <= 0.80*base AND int regs <= base regs) ==="
gates() {
  local name=$1 bc=$2 ic=$3 br=$4 ir=$5
  local thresh=$((bc * 4 / 5))
  if [ -n "$bc" ] && [ -n "$ic" ] && [ "$ic" -le "$thresh" ] && [ -n "$br" ] && [ -n "$ir" ] && [ "$ir" -le "$br" ]; then
    echo "$name: GATE PASS (int $ic <= $thresh static, regs $ir <= $br)"
  else
    echo "$name: GATE FAIL (base=$bc int=$ic thresh=$thresh, regs base=$br int=$ir)"
  fi
}
gates restrict3    "$R3_BASE" "$R3_INT" "$R3_BASE_R" "$R3_INT_R"
gates global_interp "$GI_BASE" "$GI_INT" "$GI_BASE_R" "$GI_INT_R"
gates sommerfeld   "$SOM_BASE" "$SOM_INT" "$SOM_BASE_R" "$SOM_INT_R"

echo "=== DONE $(date -u) ==="
