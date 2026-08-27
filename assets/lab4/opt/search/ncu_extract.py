import csv, sys, glob, os

EV = os.path.expanduser("~/lab4-gpu/evidence/")
d = sorted(glob.glob(EV + "ncu-p313233-*"))[-1]

for K in ["rhs_kernel_facepure", "rhs_kernel_facez", "rhs_kernel_int", "restrict3_kernel", "rungekutta4_rout_kernel"]:
    rep = os.path.join(d, K + ".ncu-rep")
    if not os.path.exists(rep):
        print("MISSING", K); continue
    print("##########", K)
    import subprocess
    out = subprocess.run(["ncu", "--import", rep, "--page", "raw", "--csv"],
                         capture_output=True, text=True).stdout
    rows = list(csv.DictReader(out.splitlines()))
    gs = set()
    for r in rows:
        if r.get("launch__grid_size"): gs.add(r["launch__grid_size"])
    for g in sorted(gs):
        r = [x for x in rows if x.get("launch__grid_size") == g][0]
        print("  grid", g, "| regs", r.get("launch__registers_per_thread"),
              "| stack", r.get("launch__stack_size"), "| dur_ms", r.get("gpu__time_duration.sum"),
              "| waves", r.get("launch__waves_per_multiprocessor"))
        for m in ["smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
                  "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
                  "lts__throughput.avg.pct_of_peak_sustained_elapsed",
                  "l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed",
                  "smsp__issue_active.avg.per_cycle_active",
                  "sm__warps_active.avg.pct_of_peak_sustained_active",
                  "l1tex__t_sector_hit_rate.pct", "lts__t_sector_hit_rate.pct",
                  "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"]:
            if m in r: print("    ", m, "=", r[m])
