#!/usr/bin/env bash
set -euo pipefail

source_dir="$PWD"
scratch_dir="$(mktemp -d)"
trap 'rm -rf "$scratch_dir"' EXIT

cp -r "$source_dir" "$scratch_dir/lab3"
cd "$scratch_dir/lab3"

for value_tile in 8 16; do
  cp "$source_dir/student/tilelang_fwd.py" student/tilelang_fwd.py
  GDN_VALUE_TILE="$value_tile" python3 run.py \
    --case chain_equal \
    --case wide_gva_state \
    --warmup 10 \
    --repetitions 20 \
    --output-format csv \
    2>&1 | tee "$source_dir/iteration2_value_tile_${value_tile}.csv"
done

python3 - "$source_dir" <<'PY'
import csv
import math
import sys
from pathlib import Path

source_dir = Path(sys.argv[1])
results = {}
for value_tile in (8, 16):
    path = source_dir / f"iteration2_value_tile_{value_tile}.csv"
    timings = []
    with path.open(encoding="utf-8") as file:
        for row in csv.DictReader(
            line for line in file if not line.startswith("2026-")
        ):
            if row.get("correctness") == "PASS":
                timings.append(float(row["median_ms"]))
    results[value_tile] = math.prod(timings) ** (1 / len(timings))

ratio = results[8] / results[16]
selected = 16 if ratio >= 0.99 else 8
summary = (
    f"VALUE_TILE=8 geometric mean: {results[8]:.6f} ms\n"
    f"VALUE_TILE=16 geometric mean: {results[16]:.6f} ms\n"
    f"ratio 8/16: {ratio:.6f}\n"
    f"selected: {selected}\n"
)
print(summary, end="")
(source_dir / "iteration2_value_tile.txt").write_text(summary, encoding="utf-8")
PY
