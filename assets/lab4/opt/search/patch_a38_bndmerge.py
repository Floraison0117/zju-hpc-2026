#!/usr/bin/env python3
"""Iter38 A38-BND: merge rhs boundary region launches (7 -> 3).

Mechanism: ncu on the deployed state shows the rhs face kernels are latency-bound
with only 0.18-0.57 waves (grid 5-16 blocks/launch). The boundary work is split
into 7 tiny sequential launches (4x facepure R0-R3, 2x facez R4-R5, 1x R6).
This patch merges R0-R3 into ONE facepure launch and R4-R5 into ONE facez launch
(1-D union grid + cumulative region selector). Per-point code path and arithmetic
are byte-identical (bnd_map_region per region unchanged) -> bit-exact by
construction (rhs writes distinct points, no cross-thread interaction).

Files changed:
  src/bssn_rhs_gpu_face.cu   (rhs_kernel_facepure: region -> region_base/end)
  src/bssn_rhs_gpu_facez.cu  (rhs_kernel_facez: region -> region_base/end)
  src/bssn_rhs_gpu.cu        (gpu_compute_rhs_bssn_launch: 6 launches -> 2)
"""
import hashlib, re, sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."

def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]

ENCF = dict(encoding="utf-8")

FACE = f"{ROOT}/src/bssn_rhs_gpu_face.cu"
FACEZ = f"{ROOT}/src/bssn_rhs_gpu_facez.cu"
RHS = f"{ROOT}/src/bssn_rhs_gpu.cu"

cur_face = sha(FACE)
cur_facez = sha(FACEZ)
cur_rhs = sha(RHS)
print(f"guard face={cur_face} facez={cur_facez} rhs={cur_rhs}")
assert cur_face == "3d6dcc0a", f"face.cu hash drift {cur_face}"
assert cur_facez == "d909fef8", f"facez.cu hash drift {cur_facez}"
assert cur_rhs == "9baee005", f"rhs.cu hash drift {cur_rhs}"

# ---- 1. face.cu kernel ----
s = open(FACE, **ENCF).read()
assert s.count("int symmetry, int lev, double eps, int co, int skip_interior, int region") == 1
s = s.replace(
    "int symmetry, int lev, double eps, int co, int skip_interior, int region",
    "int symmetry, int lev, double eps, int co, int skip_interior, int region_base, int region_end",
)
old_map = """    // Iter26b face kernel: 1-D grid over a single boundary region.
    int i, j, k;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= bnd_region_size(region, ex0, ex1, ex2)) return;
    bnd_map_region(n, region, ex0, ex1, ex2, &i, &j, &k);
    (void)skip_interior;"""
new_map = """    // Iter26b face kernel: 1-D grid over a single boundary region.
    // A38: merged enumeration over regions [region_base, region_end).
    int i, j, k;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int region = region_base;
    while (region < region_end) {
        int sz = bnd_region_size(region, ex0, ex1, ex2);
        if (n < sz) break;
        n -= sz;
        ++region;
    }
    if (region >= region_end) return;
    bnd_map_region(n, region, ex0, ex1, ex2, &i, &j, &k);
    (void)skip_interior;"""
assert s.count(old_map) == 1, "face.cu mapping anchor not found"
s = s.replace(old_map, new_map)
open(FACE, "w", **ENCF).write(s)
print("face.cu patched:", sha(FACE))

# ---- 2. facez.cu kernel: same ----
s = open(FACEZ, **ENCF).read()
assert s.count("int symmetry, int lev, double eps, int co, int skip_interior, int region") == 1
s = s.replace(
    "int symmetry, int lev, double eps, int co, int skip_interior, int region",
    "int symmetry, int lev, double eps, int co, int skip_interior, int region_base, int region_end",
)
assert s.count(old_map) == 1, "facez.cu mapping anchor not found"
s = s.replace(old_map, new_map)
open(FACEZ, "w", **ENCF).write(s)
print("facez.cu patched:", sha(FACEZ))

# ---- 3. rhs.cu host dispatch ----
s = open(RHS, **ENCF).read()
# 3a. extern kernel declarations must match the new signature too
nd = s.count("int symmetry, int lev, double eps, int co, int skip_interior, int region")
assert nd == 2, f"extern decl count {nd}"
s = s.replace(
    "int symmetry, int lev, double eps, int co, int skip_interior, int region",
    "int symmetry, int lev, double eps, int co, int skip_interior, int region_base, int region_end",
)
start_anchor = "    // 1. Kernel 1: Derivatives & Connection Coefficients (per-region launch)"
end_anchor = "    if (bnd_region_size(6, ex[0], ex[1], ex[2]) > 0) {"
i0 = s.index(start_anchor)
i1 = s.index(end_anchor)
span = s[i0:i1]
assert span.count("rhs_kernel_facepure<<<") == 4, f"facepure launches {span.count('rhs_kernel_facepure<<<')}"
assert span.count("rhs_kernel_facez<<<") == 2, f"facez launches {span.count('rhs_kernel_facez<<<')}"
assert "A38" not in span

m = re.search(r"rhs_kernel_facepure<<<g1, block, 0, stream>>>\(\n(.*?)symmetry, lev, eps, co, 1, 0\n\s*\);", span, re.S)
assert m, "cannot extract facepure args"
args = m.group(1).rstrip("\n")
# args end with "d_Gmz_Res," (trailing comma already present)
TAIL04 = "\n            symmetry, lev, eps, co, 1, 0, 4\n        );\n"
TAIL46 = "\n            symmetry, lev, eps, co, 1, 4, 6\n        );\n"
merged = (start_anchor + "\n" +
    "    // A38: merged launches (bit-exact: per-region point computation unchanged).\n" +
    "    {\n" +
    "        int s04 = 0;\n" +
    "        for (int r = 0; r < 4; ++r) s04 += bnd_region_size(r, ex[0], ex[1], ex[2]);\n" +
    "        if (s04 > 0) {\n" +
    "            dim3 gf((s04 + 255) / 256);\n" +
    "            rhs_kernel_facepure<<<gf, block, 0, stream>>>(\n" + args + TAIL04 +
    "        }\n" +
    "        int s46 = 0;\n" +
    "        for (int r = 4; r < 6; ++r) s46 += bnd_region_size(r, ex[0], ex[1], ex[2]);\n" +
    "        if (s46 > 0) {\n" +
    "            dim3 gz((s46 + 255) / 256);\n" +
    "            rhs_kernel_facez<<<gz, block, 0, stream>>>(\n" + args + TAIL46 +
    "        }\n" +
    "    }\n")
s = s[:i0] + merged + s[i1:]
open(RHS, "w", **ENCF).write(s)
print("rhs.cu patched:", sha(RHS))
print("DONE")
