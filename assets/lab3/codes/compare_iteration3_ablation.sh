#!/usr/bin/env bash
set -euo pipefail

source_dir="$PWD"
scratch_dir="$(mktemp -d)"
trap 'rm -rf "$scratch_dir"' EXIT

cp -r "$source_dir" "$scratch_dir/lab3"
cd "$scratch_dir/lab3"
cp "$source_dir/student/tilelang_fwd.py.iter3_final" student/tilelang_fwd.py

for async_a in 0 1; do
  GDN_ITER3_ASYNC_A="$async_a" python3 run.py \
    --case chain_equal \
    --case wide_gva_state \
    --warmup 10 \
    --repetitions 20 \
    --output-format csv \
    2>&1 | tee "$source_dir/iteration3_ablation_async${async_a}.csv"
done

python3 - "$source_dir" <<'PY'
import csv
import math
import sys
from pathlib import Path

source_dir = Path(sys.argv[1])
timings = {}
for async_a in (0, 1):
    path = source_dir / f"iteration3_ablation_async{async_a}.csv"
    with path.open(encoding="utf-8") as file:
        rows = csv.DictReader(
            line for line in file
            if line.startswith("case,") or not line.startswith("2026-")
        )
        timings[async_a] = {
            row["case"]: float(row["median_ms"])
            for row in rows
            if row.get("correctness") == "PASS"
        }

case_names = ("chain_equal", "wide_gva_state")
if any(case not in timings[mode] for mode in (0, 1) for case in case_names):
    raise SystemExit("missing PASS timing in an ablation CSV")

speedups = {
    case: timings[0][case] / timings[1][case]
    for case in case_names
}
geomean = math.prod(speedups.values()) ** (1 / len(speedups))
max_regression = max(timings[1][case] / timings[0][case] - 1 for case in case_names)
selected = int(geomean >= 1.01 and max_regression <= 0.02)

lines = [
    "case,persistent_only_ms,async_a_ms,speedup",
    *(
        f"{case},{timings[0][case]:.6f},{timings[1][case]:.6f},"
        f"{speedups[case]:.6f}"
        for case in case_names
    ),
    f"geometric_mean_speedup,{geomean:.6f}",
    f"maximum_regression,{max_regression:.6%}",
    f"selected_GDN_ITER3_ASYNC_A,{selected}",
]
summary = "\n".join(lines) + "\n"
print(summary, end="")
(source_dir / "iteration3_ablation_summary.txt").write_text(
    summary,
    encoding="utf-8",
)
PY
