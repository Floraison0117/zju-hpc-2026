#!/usr/bin/env python3
"""Iter38 ledger builder: nsys sqlite -> module ledger + blocks/launch + 3-window.

Usage: ledger.py <reprofile.sqlite> <out_prefix>
Outputs <prefix>ledger.txt, <prefix>ledger2.txt (subcategories), <prefix>windows.txt
"""
import sqlite3, sys, statistics, os

DB = sys.argv[1]
PRE = sys.argv[2]

con = sqlite3.connect(DB)
cur = con.cursor()
# nsys stores kernel names as StringIds (ints) in CUPTI_ACTIVITY_KIND_KERNEL
has_strings = cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='StringIds'").fetchone()
sid = {}
if has_strings:
    for r in cur.execute("SELECT id, value FROM StringIds"):
        sid[r[0]] = r[1]

def resolve_name(v):
    if isinstance(v, str):
        return v
    return sid.get(v, str(v))

cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='CUPTI_ACTIVITY_KIND_KERNEL'")
if not cur.fetchone():
    print("no CUPTI_ACTIVITY_KIND_KERNEL table"); sys.exit(2)
cols = [r[1] for r in cur.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)").fetchall()]
cname = "shortName" if "shortName" in cols else "demangledName"
sel = f"start, end, {cname}, gridX, gridY, gridZ"
rows = [(start, end, resolve_name(name), gx, gy, gz) for start, end, name, gx, gy, gz in cur.execute(f"SELECT {sel} FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchall()]

kern = {}          # name -> (calls, total_ns, [grids])
for start, end, name, gx, gy, gz in rows:
    k = kern.setdefault(name, [0, 0, []])
    k[0] += 1
    k[1] += (end - start)
    k[2].append(gx * gy * gz)

def med(v):
    return statistics.median(v) if v else 0

kt = sum(v[1] for v in kern.values())
if not rows:
    print("no kernel rows"); sys.exit(1)
t0 = min(r[0] for r in rows); t1 = max(r[1] for r in rows)
span = t1 - t0

order = sorted(kern.items(), key=lambda kv: -kv[1][1])
out = []
out.append(f"kernels={len(rows)} name_col={cname}")
out.append("")
out.append(f"=== ALL KERNELS (kernel-time {kt*1e-9:.1f}s, span {span*1e-9:.1f}s, host-gap {(span-kt)*1e-9:.1f}s = {(span-kt)/span*100:.1f}%) ===")
out.append("kernel                                           calls   total_s   avg_ms  share%  blocks/launch min/med/max")
for name, (calls, tot, grids) in order:
    out.append(f"{name:<48} {calls:>9} {tot*1e-9:>9.2f} {tot*1e-6/calls:>8.3f} {tot/kt*100:>6.2f}   {min(grids)}/{med(grids):.0f}/{max(grids)}")
open(PRE + "ledger.txt", "w").write("\n".join(out) + "\n")

# module subcategories
MODS = [
    ("rhs_boundary", ["rhs_kernel"]),
    ("rhs_interior", ["rhs_kernel_int"]),
    ("rhs_face", ["rhs_kernel_facepure", "rhs_kernel_facez"]),
    ("prolong3_boundary", ["prolong3_kernel", "prolong3_multi_kernel"]),
    ("prolong3_interior", ["prolong3_kernel_int", "prolong3_multi_kernel_int"]),
    ("analysis", ["global_interp_kernel", "global_interp_multi_kernel", "global_interp_amr_kernel",
                  "surf_Wave_kernel", "surf_MassPAng_kernel", "average_kernel", "average2_kernel",
                  "l2normhelper_kernel", "admmass_bssn_kernel", "getnp4_kernel", "scale_normals_kernel",
                  "normalize_shellf_kernel"]),
    ("sommerfeld", ["sommerfeld_rout_kernel", "sommerfeld_routbam_kernel", "sommerfeld_rout_compact_kernel"]),
    ("restrict3", ["restrict3_kernel", "restrict3_multi_kernel"]),
    ("rk4", ["rungekutta4_rout_kernel", "rungekutta4_batch_kernel"]),
    ("ghost", ["gpu_pack_kernel", "gpu_unpack_kernel"]),
    ("enforce", ["enforce_ga_kernel"]),
    ("misc", ["lowerboundset_kernel"]),
]
out2 = []
out2.append("=== SUBCATEGORIES (rhs/prolong interior vs boundary, KO fused in rhs) ===")
out2.append("subcategory                 calls   total_s  share%")
for label, names in MODS:
    calls = sum(kern.get(n, [0])[0] for n in names)
    tot = sum(kern.get(n, [0, 0])[1] for n in names)
    out2.append(f"{label:<28} {calls:>9} {tot*1e-9:>9.2f} {tot/kt*100:>6.2f}")
open(PRE + "ledger2.txt", "w").write("\n".join(out2) + "\n")

# three windows (equal wall thirds)
W = 3
win = [dict() for _ in range(W)]
win_k = [dict() for _ in range(W)]
for start, end, name, gx, gy, gz in rows:
    mid = (start + end) / 2
    w = min(W - 1, int((mid - t0) / span * W))
    d = win[w].setdefault(name, [0, 0, []])
    d[0] += 1; d[1] += (end - start); d[2].append(gx * gy * gz)
outw = []
outw.append(f"=== THREE WINDOWS (span {span*1e-9:.1f}s, {span*1e-9/W:.1f}s each; share% of window kernel time) ===")
keynames = [k for k, _ in order if isinstance(k, str) and k in kern and kern[k][1] > kt * 0.004]
hdr = "kernel".ljust(44) + "".join(f"W{i+1}" .rjust(10) for i in range(W))
outw.append(hdr)
for name in keynames:
    line = name.ljust(44)
    for i in range(W):
        d = win[i].get(name, [0, 0, []])
        wkt = sum(v[1] for v in win[i].values())
        line += f"{(d[1]/wkt*100 if wkt else 0):>9.2f}%"
    outw.append(line)
outw.append("")
outw.append("blocks/launch per window (min/med/max) for key kernels:")
for name in keynames:
    line = name.ljust(44)
    for i in range(W):
        d = win[i].get(name, [0, 0, []])
        if d[2]:
            line += f" {min(d[2])}/{med(d[2]):.0f}/{max(d[2])}"
        else:
            line += " " + "-".rjust(14)
    outw.append(line)
open(PRE + "windows.txt", "w").write("\n".join(outw) + "\n")
print("ledger written to", PRE)
