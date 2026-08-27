#!/usr/bin/env bash
set -euo pipefail

source_dir="$PWD"
scratch_dir="$(mktemp -d)"
trap 'rm -rf "$scratch_dir"' EXIT

cp -r "$source_dir" "$scratch_dir/lab3"
cd "$scratch_dir/lab3"

cases=(
  short_tail_state
  chain_equal
  parallel_equal
  parallel_gva
  long_low_gva
  batch_split_gva
  wide_gva_state
)
case_args=()
for case_name in "${cases[@]}"; do
  case_args+=(--case "$case_name")
done

cp "$source_dir/student/tilelang_fwd.py.iter1_final" student/tilelang_fwd.py
python3 run.py \
  "${case_args[@]}" \
  --warmup 10 \
  --repetitions 30 \
  --output-format csv \
  2>&1 | tee "$source_dir/iteration1_final_ab_hybrid.csv"

cp "$source_dir/student/tilelang_fwd.py" student/tilelang_fwd.py
python3 run.py \
  "${case_args[@]}" \
  --warmup 10 \
  --repetitions 30 \
  --output-format csv \
  2>&1 | tee "$source_dir/iteration2_ab_hybrid.csv"
