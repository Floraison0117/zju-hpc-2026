#!/usr/bin/env bash
set -euo pipefail

source_dir="$PWD"
scratch_dir="$(mktemp -d)"
trap 'rm -rf "$scratch_dir"' EXIT

cp -r "$source_dir" "$scratch_dir/lab3"
cd "$scratch_dir/lab3"

cp "$source_dir/student/tilelang_fwd.py.iter1_final" student/tilelang_fwd.py
python3 run.py \
  --case deep_gva_state \
  --warmup 10 \
  --repetitions 30 \
  --output-format csv \
  2>&1 | tee "$source_dir/iteration1_deep_ab.csv"

cp "$source_dir/student/tilelang_fwd.py" student/tilelang_fwd.py
python3 run.py \
  --case deep_gva_state \
  --warmup 10 \
  --repetitions 30 \
  --output-format csv \
  2>&1 | tee "$source_dir/iteration2_deep_ab.csv"
