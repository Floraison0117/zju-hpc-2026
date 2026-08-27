#!/usr/bin/env python3
"""
R5 L0 probe: Ricci Step-4 segment ablation on the thin interior kernel.

Diagnostic: WHERE is the register peak on the thin kernel? Replace the entire
Step-4 block (6 Ricci correction monsters, lines 505-577 in deployed file:
`// Rxx Correction` ... just before `// Step 6`) with 6 trivial statements that
read ALL the same input variables (l_gxx..l_gzz, dGamxx..dGamzz, Gamxa/y/a,
gupxx..gupzz, l_Gamxxx..l_Gamzzz, gxxx..gzzz) so the input live set is
unchanged, but the deep expression trees (transient arithmetic) are removed.

Interpretation:
- natural regs drops significantly below 255 -> Ricci Step-4 expression depth
  was a peak driver -> per-component recompute (P3/Lever B retest, candidate #3)
  is worth pursuing on the thin kernel.
- natural regs stays 255 -> peak is elsewhere (fdderivs 61-fh segment, or the
  input live set itself) -> Ricci recompute is dead again (iter9 conclusion).

Usage: python3 patch_r5_ricci_ablate.py <src_dir> <dst_dir>
"""
import sys, os, hashlib

SRC = os.path.join(sys.argv[1], "bssn_rhs_gpu_int.cu")
DST = os.path.join(sys.argv[2], "bssn_rhs_gpu_int_ablate.cu")

with open(SRC) as f:
    src = f.read()

h0 = hashlib.sha256(src.encode()).hexdigest()
EXPECT = "8dc0cf28aef2da184ca219e5e1a817073df6320cf85a87f09148489dd44a3839"
if h0 != EXPECT:
    print(f"HASH_GUARD_FAIL src={h0}", file=sys.stderr)
    sys.exit(2)

start_marker = "    // Rxx Correction\n"
end_marker = "    // Step 6: Chi 二阶导数与 Ricci 修正\n"

s = src.find(start_marker)
e = src.find(end_marker)
assert s >= 0, "Rxx marker not found"
assert e >= 0, "Step 6 marker not found"
assert e > s, "marker order wrong"

# input live set of the whole Step-4 block (all variables referenced there)
inputs = (
    "l_gxx + l_gxy + l_gxz + l_gyy + l_gyz + l_gzz + "
    "dGamxx + dGamxy + dGamxz + dGamyx + dGamyy + dGamyz + dGamzx + dGamzy + dGamzz + "
    "Gamxa + Gamya + Gamza + "
    "gupxx + gupxy + gupxz + gupyy + gupyz + gupzz + "
    "l_Gamxxx + l_Gamxxy + l_Gamxxz + l_Gamxyy + l_Gamxyz + l_Gamxzz + "
    "l_Gamyxx + l_Gamyxy + l_Gamyxz + l_Gamyyy + l_Gamyyz + l_Gamyzz + "
    "l_Gamzxx + l_Gamzxy + l_Gamzxz + l_Gamzyy + l_Gamzyz + l_Gamzzz + "
    "gxxx + gxxy + gxxz + gxyx + gxyy + gxyz + gxzx + gxzy + gxzz + "
    "gyyx + gyyy + gyyz + gyzx + gyzy + gyzz + gzzx + gzzy + gzzz"
)

ablated = "    // R5 ablation: Ricci Step-4 expression depth removed (input live set preserved)\n"
for v in ["l_Rxx", "l_Ryy", "l_Rzz", "l_Rxy", "l_Rxz", "l_Ryz"]:
    ablated += f"    {v} = {v} + ({inputs});\n"

src = src[:s] + ablated + src[e:]

os.makedirs(sys.argv[2], exist_ok=True)
with open(DST, "w") as f:
    f.write(src)
print(f"OK wrote {DST}")
print(f"sha256(new)={hashlib.sha256(src.encode()).hexdigest()}")
