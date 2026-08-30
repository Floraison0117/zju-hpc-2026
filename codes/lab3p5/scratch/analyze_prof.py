#!/usr/bin/env python3
"""Analyze msprof op PipeUtilization.csv for the lab3p5 kernel.

Usage: python3 scratch/analyze_prof.py [prof_dir]   (default: newest op_prof_*)
Prints Task Duration (if findable) and per-block pipe time stats.
"""
import csv
import glob
import os
import statistics as st
import sys

root = os.path.expanduser("~/lab3p5")
if len(sys.argv) > 1:
    base = sys.argv[1]
else:
    cands = sorted(glob.glob(os.path.join(root, "op_prof_*")))
    if not cands:
        sys.exit("no op_prof_* dirs")
    base = cands[-1]
if not os.path.isabs(base):
    base = os.path.join(root, base)

pipe_files = sorted(glob.glob(os.path.join(base, "*", "PipeUtilization.csv")))
if not pipe_files:
    pipe_files = sorted(glob.glob(os.path.join(base, "PipeUtilization.csv")))
if not pipe_files:
    sys.exit(f"no PipeUtilization.csv under {base}")
pf = pipe_files[0]
print(f"[analyze] {pf}")

# Task duration lives in the msprof output summary next to the csv dir.
for info in sorted(glob.glob(os.path.join(base, "*", "OpBasicInfo.csv"))):
    with open(info) as fh:
        for r in csv.DictReader(fh):
            keys = {k.lower(): v for k, v in r.items() if k}
            dur = keys.get("task duration(us)") or keys.get("duration(us)")
            if dur and dur not in ("NA", ""):
                print(f"[analyze] OpBasicInfo: {os.path.basename(info)} task_duration={dur} us")

rows = []
with open(pf, newline="") as fh:
    for r in csv.DictReader(fh):
        r = {k.replace("(us)", "").replace("(GB/s)", ""): v for k, v in r.items() if k}
        if r.get("aiv_time") and r["aiv_time"] not in ("NA", ""):
            rows.append(r)
if not rows:
    sys.exit("no aiv rows found")

def col(k):
    out = []
    for r in rows:
        v = r.get(k, "")
        if v and v not in ("NA", ""):
            try:
                out.append(float(v))
            except ValueError:
                pass
    return out

print(f"[analyze] blocks with aiv_time: {len(rows)}")
for k in ["aiv_time", "aiv_vec_time", "aiv_scalar_time",
          "aiv_mte2_time", "aiv_mte3_time", "aiv_total_cycles"]:
    v = col(k)
    if not v:
        continue
    print(f"  {k:16s} n={len(v):3d} min={min(v):8.3f} med={st.median(v):8.3f} max={max(v):8.3f} sum={sum(v):9.3f}")

sv = sorted(rows, key=lambda r: -float(r["aiv_time"]))
print("  slowest blocks:")
for r in sv[:5]:
    try:
        print("    blk {bid:>3s}: aiv={a:6.3f} vec={v:6.3f} sca={s:6.3f} mte2={m:6.3f} mte3={o:6.3f}".format(
            bid=r.get("block_id", "?"), a=float(r["aiv_time"]),
            v=float(r.get("aiv_vec_time") or 0), s=float(r.get("aiv_scalar_time") or 0),
            m=float(r.get("aiv_mte2_time") or 0), o=float(r.get("aiv_mte3_time") or 0)))
    except (KeyError, ValueError):
        pass
