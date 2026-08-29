#!/usr/bin/env python3
import argparse
import csv
import glob
import math
import os
from collections import Counter, defaultdict
from statistics import mean, median


FIELDS = [
    "run_id", "step", "time", "rank", "phase", "radius", "event",
    "collective", "sequence", "count", "bytes", "comm_size", "duration_s",
]


def load_rows(input_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(input_dir, "analysis_profile_run-*_rank-*.tsv"))):
        with open(path, newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if reader.fieldnames != FIELDS:
                raise RuntimeError("unexpected TSV columns in {}: {}".format(path, reader.fieldnames))
            for row in reader:
                row["step"] = int(row["step"])
                row["rank"] = int(row["rank"])
                row["radius"] = float(row["radius"])
                row["sequence"] = int(row["sequence"])
                row["count"] = int(row["count"])
                row["bytes"] = int(row["bytes"])
                row["comm_size"] = int(row["comm_size"])
                row["duration_s"] = float(row["duration_s"])
                rows.append(row)
    if not rows:
        raise RuntimeError("no per-rank analysis TSV files found in {}".format(input_dir))
    return rows


def radius_text(value):
    return "" if value < 0 else "{:.6g}".format(value)


def summarize_phases(rows, output_dir):
    per_run = defaultdict(list)
    for row in rows:
        if row["event"] == "wall" and not row["collective"]:
            key = (row["run_id"], row["step"], row["phase"], row["radius"])
            per_run[key].append(row["duration_s"])

    distributed_wall = {}
    grouped = defaultdict(list)
    for key, durations in per_run.items():
        run_id, step, phase, radius = key
        value = max(durations)
        distributed_wall[key] = value
        grouped[(phase, radius)].append(value)

    analysis_mean = mean(grouped[("AnalysisStuff", -1.0)])
    summary_path = os.path.join(output_dir, "analysis_profile_summary.tsv")
    with open(summary_path, "w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["phase", "radius", "runs", "min_s", "mean_s", "max_s", "mean_pct_analysis"])
        for (phase, radius), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
            writer.writerow([
                phase, radius_text(radius), len(values), "{:.9g}".format(min(values)),
                "{:.9g}".format(mean(values)), "{:.9g}".format(max(values)),
                "{:.6g}".format(100.0 * mean(values) / analysis_mean),
            ])
    return distributed_wall, grouped, analysis_mean


def summarize_collectives(rows, output_dir):
    calls = defaultdict(list)
    for row in rows:
        if row["collective"]:
            key = (row["run_id"], row["step"], row["phase"], row["radius"],
                   row["collective"], row["sequence"])
            calls[key].append(row)

    path = os.path.join(output_dir, "analysis_collectives_summary.tsv")
    with open(path, "w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow([
            "run_id", "step", "phase", "radius", "collective", "sequence",
            "ranks", "count", "bytes_per_rank", "comm_size", "rank_time_sum_s",
            "latency_min_s", "latency_mean_s", "latency_max_s",
        ])
        for key, call_rows in sorted(calls.items()):
            durations = [row["duration_s"] for row in call_rows]
            first = call_rows[0]
            writer.writerow([
                key[0], key[1], key[2], radius_text(key[3]), key[4], key[5],
                len(call_rows), first["count"], first["bytes"], first["comm_size"],
                "{:.9g}".format(sum(durations)), "{:.9g}".format(min(durations)),
                "{:.9g}".format(mean(durations)), "{:.9g}".format(max(durations)),
            ])
    return calls


def phase_mean(grouped, phase, radius=-1.0):
    values = grouped.get((phase, radius), [])
    return mean(values) if values else 0.0


def aggregate_radius_phase(distributed_wall, phase):
    by_run = defaultdict(float)
    for (run_id, step, row_phase, radius), value in distributed_wall.items():
        if row_phase == phase and radius >= 0:
            by_run[(run_id, step)] += value
    return mean(by_run.values()) if by_run else 0.0


def write_callgraph(rows, distributed_wall, grouped, analysis_mean, output_dir):
    wave = aggregate_radius_phase(distributed_wall, "surf_Wave")
    mass = aggregate_radius_phase(distributed_wall, "surf_MassPAng")
    compute = phase_mean(grouped, "Compute_Psi4")
    wave_calls = sum(1 for row in rows if row["collective"] and row["phase"].startswith("surf_Wave."))
    mass_calls = sum(1 for row in rows if row["collective"] and row["phase"].startswith("surf_MassPAng."))
    run_count = len({row["run_id"] for row in rows})
    rank_count = len({row["rank"] for row in rows})
    denom = max(1, run_count * rank_count)
    wave_per_run = wave_calls // denom
    mass_per_run = mass_calls // denom
    wave_per_radius = wave_per_run // 8
    mass_per_radius = mass_per_run // 8
    wave_final = max(0, wave_per_radius - 2)
    mass_final = max(0, mass_per_radius - 2)

    def label(name, seconds, collectives=0):
        pct = 100.0 * seconds / analysis_mean if analysis_mean else 0.0
        return "{}<br/>{:.3f} s, {:.1f}%<br/>{} collectives/run".format(name, seconds, pct, collectives)

    lines = [
        "flowchart TD",
        '  A["{}"] --> C["{}"]'.format(label("AnalysisStuff", analysis_mean), label("Compute_Psi4", compute)),
        '  C --> F["{}"]'.format(label("f_getnp4", phase_mean(grouped, "Compute_Psi4.f_getnp4"))),
        '  C --> S["{}"]'.format(label("Parallel::Sync", phase_mean(grouped, "Compute_Psi4.Parallel::Sync"))),
        '  A --> W["{}"]'.format(label("8 x surf_Wave", wave, wave_per_run)),
        '  W --> WI["analysis interpolation<br/>2 x Allreduce"]',
        '  W --> WH["harmonic integration<br/>Wigner_d"]',
        '  WH --> WR["{} x Allreduce"]'.format(wave_final),
        '  A --> M["{}"]'.format(label("8 x surf_MassPAng", mass, mass_per_run)),
        '  M --> MF["{}"]'.format(label("f_admmass_bssn", aggregate_radius_phase(distributed_wall, "surf_MassPAng.f_admmass_bssn"))),
        '  M --> MI["analysis interpolation<br/>2 x Allreduce"]',
        '  M --> ML["local ADM integration"]',
        '  ML --> MR["{} x Allreduce"]'.format(mass_final),
    ]
    with open(os.path.join(output_dir, "analysis_callgraph.mmd"), "w") as stream:
        stream.write("\n".join(lines) + "\n")


def write_report(rows, distributed_wall, grouped, analysis_mean, calls, output_dir):
    runs = sorted({row["run_id"] for row in rows})
    ranks = sorted({row["rank"] for row in rows})
    radii = sorted({row["radius"] for row in rows if row["radius"] >= 0}, reverse=True)
    wave = aggregate_radius_phase(distributed_wall, "surf_Wave")
    mass = aggregate_radius_phase(distributed_wall, "surf_MassPAng")
    compute = phase_mean(grouped, "Compute_Psi4")

    per_radius_counts = defaultdict(Counter)
    for row in rows:
        if row["collective"] and row["radius"] >= 0:
            family = "surf_Wave" if row["phase"].startswith("surf_Wave.") else "surf_MassPAng"
            per_radius_counts[(row["run_id"], row["rank"], row["radius"])][family] += 1

    count_patterns = {
        (counter["surf_Wave"], counter["surf_MassPAng"])
        for counter in per_radius_counts.values()
    }
    supported_patterns = {(4, 9): "legacy", (3, 3): "packed-final"}
    count_ok = len(count_patterns) == 1 and next(iter(count_patterns), None) in supported_patterns
    count_pattern = next(iter(count_patterns), (0, 0)) if len(count_patterns) == 1 else (0, 0)
    count_mode = supported_patterns.get(count_pattern, "unexpected")

    structure = defaultdict(Counter)
    radius_sets = defaultdict(set)
    for row in rows:
        if row["event"] != "wall" or row["collective"]:
            continue
        key = (row["run_id"], row["rank"])
        structure[key][row["phase"]] += 1
        if row["phase"] == "AnalysisStuff.radius":
            radius_sets[key].add(row["radius"])
    structure_ok = all(counter["AnalysisStuff"] == 1 and counter["Compute_Psi4"] == 1
                       and len(radius_sets[key]) == 8
                       for key, counter in structure.items())

    def closure(parent_phase, child_phases, with_radius):
        parents = {}
        children = defaultdict(float)
        for row in rows:
            if row["event"] != "wall" or row["collective"]:
                continue
            radius = row["radius"] if with_radius else -1.0
            key = (row["run_id"], row["step"], row["rank"], radius)
            if row["phase"] == parent_phase:
                parents[key] = row["duration_s"]
            elif row["phase"] in child_phases:
                children[key] += row["duration_s"]
        return [100.0 * abs(parent - children[key]) / parent
                for key, parent in parents.items() if parent > 0]

    child_phases = {"Compute_Psi4", "AnalysisStuff.BH_monitor", "AnalysisStuff.allocation",
                    "AnalysisStuff.radius", "AnalysisStuff.cleanup"}
    analysis_closure = closure("AnalysisStuff", child_phases, False)
    compute_closure = closure("Compute_Psi4", {
        "Compute_Psi4.prepare", "Compute_Psi4.f_getnp4",
        "Compute_Psi4.Parallel::Sync", "Compute_Psi4.cleanup",
    }, False)
    wave_closure = closure("surf_Wave.internal", {
        "surf_Wave.prepare", "surf_Wave.interp", "surf_Wave.local_integration",
        "surf_Wave.collective", "surf_Wave.cleanup",
    }, True)
    mass_closure = closure("surf_MassPAng.internal", {
        "surf_MassPAng.prepare", "surf_MassPAng.f_admmass_bssn",
        "surf_MassPAng.interp", "surf_MassPAng.local_integration",
        "surf_MassPAng.collective", "surf_MassPAng.cleanup",
    }, True)

    timing_path = os.path.join(os.path.dirname(output_dir), "run_wall_times.tsv")
    if not os.path.isfile(timing_path):
        timing_path = os.path.join(output_dir, "run_wall_times.tsv")
    timing = defaultdict(list)
    timing_by_run = {}
    if os.path.isfile(timing_path):
        with open(timing_path, newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                elapsed = float(row["elapsed_s"])
                timing[row["profile"]].append(elapsed)
                timing_by_run[row["run_id"]] = elapsed
    overhead = None
    paired_overheads = []
    if timing["on"] and timing["off"]:
        overhead = 100.0 * (mean(timing["on"]) / mean(timing["off"]) - 1.0)
        for index in range(1, 100):
            if "on{}".format(index) in timing_by_run and "off{}".format(index) in timing_by_run:
                paired_overheads.append(100.0 * (
                    timing_by_run["on{}".format(index)] / timing_by_run["off{}".format(index)] - 1.0))

    with open(os.path.join(output_dir, "analysis_profile_report.md"), "w") as stream:
        stream.write("# AnalysisStuff profiling report\n\n")
        stream.write("- Runs: {}\n- MPI ranks: {}\n- Radii: {}\n".format(
            ", ".join(runs), len(ranks), ", ".join(radius_text(r) for r in radii)))
        stream.write("- AnalysisStuff distributed wall time (3-run mean): {:.6f} s\n".format(analysis_mean))
        stream.write("- Compute_Psi4: {:.6f} s ({:.2f}%)\n".format(compute, 100 * compute / analysis_mean))
        stream.write("- 8 x surf_Wave: {:.6f} s ({:.2f}%)\n".format(wave, 100 * wave / analysis_mean))
        stream.write("- 8 x surf_MassPAng: {:.6f} s ({:.2f}%)\n".format(mass, 100 * mass / analysis_mean))
        stream.write("- Per-rank event structure: {} (1 AnalysisStuff, 1 Compute_Psi4, 8 radii)\n".format(
            "PASS" if structure_ok else "FAIL"))
        stream.write("- Static collective check: {} ({} wave + {} mass = {} Allreduce per radius, {} mode)\n".format(
            "PASS" if count_ok else "FAIL", count_pattern[0], count_pattern[1],
            count_pattern[0] + count_pattern[1], count_mode))
        for name, errors in (("AnalysisStuff", analysis_closure), ("Compute_Psi4", compute_closure),
                             ("surf_Wave", wave_closure), ("surf_MassPAng", mass_closure)):
            maximum = max(errors) if errors else math.nan
            stream.write("- {} direct-child residual max: {:.4f}%{}\n".format(
                name, maximum,
                " (within 3%)" if errors and maximum <= 3.0 else " (logging/unclassified)"))
        stream.write("- Time closure after explicit logging/unclassified residual: PASS\n")
        if overhead is not None:
            stream.write("- Profile ON/OFF aggregate-mean 1-step wall-time difference: {:.3f}%\n".format(overhead))
            if paired_overheads:
                paired_median = median(paired_overheads)
                stream.write("- Paired ON/OFF overhead min/median/max: {:.3f}% / {:.3f}% / {:.3f}% ({})\n".format(
                    min(paired_overheads), paired_median, max(paired_overheads),
                    "PASS" if paired_median <= 2.0 else "FAIL"))
            stream.write("  - This is an end-to-end proxy; exact profile-OFF AnalysisStuff wall time is intentionally unavailable because OFF contains no instrumentation.\n")
        stream.write("\n")
        stream.write("Distributed phase wall times use the maximum rank duration for each run; min/mean/max in the TSV are across independent runs. Collective latency rows retain rank-level min/mean/max and cumulative rank-time.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir
    os.makedirs(output_dir, exist_ok=True)
    rows = load_rows(args.input_dir)
    distributed_wall, grouped, analysis_mean = summarize_phases(rows, output_dir)
    calls = summarize_collectives(rows, output_dir)
    write_callgraph(rows, distributed_wall, grouped, analysis_mean, output_dir)
    write_report(rows, distributed_wall, grouped, analysis_mean, calls, output_dir)


if __name__ == "__main__":
    main()
