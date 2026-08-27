#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 group1|group2|deep" >&2
  exit 2
fi

group="$1"
case "$group" in
  group1)
    cases=(
      short_tail_state
      long_low_gva
      batch_split_gva
      chain_equal
    )
    ;;
  group2)
    cases=(
      parallel_equal
      parallel_gva
      wide_gva_state
    )
    ;;
  deep)
    cases=(deep_gva_state)
    ;;
  *)
    echo "unknown group: $group" >&2
    exit 2
    ;;
esac

source_dir="$PWD"
scratch_dir="$(mktemp -d)"
trap 'rm -rf "$scratch_dir"' EXIT

cp -r "$source_dir" "$scratch_dir/lab3"
cd "$scratch_dir/lab3"

case_args=()
for case_name in "${cases[@]}"; do
  case_args+=(--case "$case_name")
done

cp "$source_dir/student/tilelang_fwd.py.iter2_final" student/tilelang_fwd.py
python3 run.py \
  "${case_args[@]}" \
  --warmup 10 \
  --repetitions 30 \
  --output-format csv \
  2>&1 | tee "$source_dir/iteration2_iter3_${group}.csv"

cp "$source_dir/student/tilelang_fwd.py.iter3_final" student/tilelang_fwd.py
python3 run.py \
  "${case_args[@]}" \
  --warmup 10 \
  --repetitions 30 \
  --output-format csv \
  2>&1 | tee "$source_dir/iteration3_${group}.csv"
