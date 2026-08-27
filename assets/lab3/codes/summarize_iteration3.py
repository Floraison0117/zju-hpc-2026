import csv
import math
import sys
from pathlib import Path


def read_timings(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8") as file:
        rows = csv.DictReader(
            line for line in file
            if line.startswith("case,") or not line.startswith("2026-")
        )
        timings = {
            row["case"]: float(row["median_ms"])
            for row in rows
            if row.get("correctness") == "PASS" and row.get("median_ms")
        }
    if not timings:
        raise ValueError(f"{path}: no PASS timing rows")
    return timings


def main(argv: list[str]) -> None:
    if len(argv) != 6:
        raise SystemExit(
            "usage: summarize_iteration3.py "
            "ITER2_GROUP1 ITER3_GROUP1 ITER2_GROUP2 ITER3_GROUP2 "
            "ITER2_DEEP ITER3_DEEP"
        )

    combined: list[tuple[str, float, float]] = []
    for index in range(0, len(argv), 2):
        iteration2 = read_timings(Path(argv[index]))
        iteration3 = read_timings(Path(argv[index + 1]))
        if iteration2.keys() != iteration3.keys():
            missing_2 = sorted(iteration3.keys() - iteration2.keys())
            missing_3 = sorted(iteration2.keys() - iteration3.keys())
            raise ValueError(
                f"CSV case mismatch: missing iter2={missing_2}, "
                f"missing iter3={missing_3}"
            )
        combined.extend(
            (case, iteration2[case], iteration3[case])
            for case in iteration2
        )

    speedups = [iteration2 / iteration3 for _, iteration2, iteration3 in combined]
    faster = sum(speedup > 1 for speedup in speedups)
    regressions = [
        case
        for (case, _, _), speedup in zip(combined, speedups, strict=True)
        if speedup < 1 / 1.02
    ]
    geomean = math.prod(speedups) ** (1 / len(speedups))

    print("case,iteration2_ms,iteration3_ms,speedup,status")
    for (case, iteration2, iteration3), speedup in zip(
        combined,
        speedups,
        strict=True,
    ):
        if speedup > 1:
            status = "faster"
        elif speedup < 1 / 1.02:
            status = "regression_gt_2pct"
        else:
            status = "within_2pct"
        print(
            f"{case},{iteration2:.6f},{iteration3:.6f},"
            f"{speedup:.6f},{status}"
        )
    print()
    print(f"faster_cases: {faster}/{len(combined)}")
    print(f"geometric_mean_speedup: {geomean:.6f}x")
    print(
        "regressions_over_2pct: "
        + (", ".join(regressions) if regressions else "none")
    )
    accepted = faster >= 6 and not regressions
    print(f"iteration3_acceptance: {'PASS' if accepted else 'FAIL'}")


if __name__ == "__main__":
    main(sys.argv[1:])
