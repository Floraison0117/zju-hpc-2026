#!/usr/bin/env python3
# iter16 P7a — restrict3 Z-restriction column-fused unroll (bit-exact)
#
# Mechanism: restrict3_kernel is pinned at 128 regs by -maxrregcount=128 +
# __launch_bounds__(256,2), currently 128B spill (tmp2[6][6] local round-trip,
# 36 STL+36 LDL per thread), 2 blocks/SM (25% occupancy floor).
# P7a: per n-column, unroll Z-restriction into 6 explicit d_zrestr6 chains
# (m=0..5, 6 independent loads each) and fuse the Y-restriction immediately
# (identical left-assoc pairing tmp2[0]+tmp2[5] -> z0+z5, ...). tmp2[6][6]
# round-trip eliminated. Registers bounded per column (6 z-values + Y acc die
# at each column end) so occupancy cannot drop (already at floor); the
# 128-reg cap turns any overflow into spill, not occupancy loss.
# Association order preserved verbatim -> bit-exact by construction
# (5 invariants, mirroring P6a: helper verbatim, Y pairing (0,5)(1,4)(2,3),
# n-mapping if_fine-2+n, m-mapping jf_fine-2+m, tmp2 gone from restrict3 only).
#
# Single variable: prolongrestrict_cell_gpu.cu, d_restrict3_device section 3.
# Scope guard: d_prolong3_device untouched.
import sys, pathlib, hashlib, re

FORMAL_HASH = "a81cd33ebaabb8f569847ef26e241cc4f2e8ca4ff6df0beeb410bdacbb19b020"

def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

OLD_BLOCK = """    // --- 3. Restriction ---
    double tmp2[6][6];
    double tmp1[6];

    // Z-Direction Restriction
    for (int m = 0; m < 6; m++) {
        for (int n = 0; n < 6; n++) {
            int cur_jf = jf_fine - 2 + m;
            int cur_if = if_fine - 2 + n;
            
            double val = 0.0;
            // Ord=2 passed to symmetry_bd as per Fortran restrict3
            // Indices: -2, -1, 0, 1, 2, 3 relative to fine center
            val += C_RESTRICT[0] * (
                d_symmetry_bd_1b(2, extf, funf, cur_if, cur_jf, kf_fine - 2, SoA) + 
                d_symmetry_bd_1b(2, extf, funf, cur_if, cur_jf, kf_fine + 3, SoA)
            );
            val += C_RESTRICT[1] * (
                d_symmetry_bd_1b(2, extf, funf, cur_if, cur_jf, kf_fine - 1, SoA) + 
                d_symmetry_bd_1b(2, extf, funf, cur_if, cur_jf, kf_fine + 2, SoA)
            );
            val += C_RESTRICT[2] * (
                d_symmetry_bd_1b(2, extf, funf, cur_if, cur_jf, kf_fine    , SoA) + 
                d_symmetry_bd_1b(2, extf, funf, cur_if, cur_jf, kf_fine + 1, SoA)
            );
            
            tmp2[m][n] = val;
        }
    }

    // Y-Direction Restriction
    for (int n = 0; n < 6; n++) {
        double val = 0.0;
        val += C_RESTRICT[0] * (tmp2[0][n] + tmp2[5][n]);
        val += C_RESTRICT[1] * (tmp2[1][n] + tmp2[4][n]);
        val += C_RESTRICT[2] * (tmp2[2][n] + tmp2[3][n]);
        tmp1[n] = val;
    }
"""

HELPER = """// Z-direction 3-group restriction for one (m,n) point (P7a helper).
// Body is VERBATIM the original d_restrict3_device inner val chain:
// same C_RESTRICT index order, same kf_fine offsets, same left-assoc val +=
// chain => bit-exact identical tmp2[m][n] values. __forceinline__ so the
// 36 call sites unroll into 36 explicit chains with 6 independent loads each.
__device__ __forceinline__ double d_zrestr6(
    const int* extf, const double* funf, const double SoA[3],
    int cur_if, int cur_jf, int kf_fine
) {
    double val = 0.0;
    val += C_RESTRICT[0] * (
        d_symmetry_bd_1b(2, extf, funf, cur_if, cur_jf, kf_fine - 2, SoA) + 
        d_symmetry_bd_1b(2, extf, funf, cur_if, cur_jf, kf_fine + 3, SoA)
    );
    val += C_RESTRICT[1] * (
        d_symmetry_bd_1b(2, extf, funf, cur_if, cur_jf, kf_fine - 1, SoA) + 
        d_symmetry_bd_1b(2, extf, funf, cur_if, cur_jf, kf_fine + 2, SoA)
    );
    val += C_RESTRICT[2] * (
        d_symmetry_bd_1b(2, extf, funf, cur_if, cur_jf, kf_fine    , SoA) + 
        d_symmetry_bd_1b(2, extf, funf, cur_if, cur_jf, kf_fine + 1, SoA)
    );
    return val;
}

"""

def build_new_body():
    canon_j = {0: "jf_fine - 2", 1: "jf_fine - 1", 2: "jf_fine", 3: "jf_fine + 1", 4: "jf_fine + 2", 5: "jf_fine + 3"}
    L = []
    L.append("    // --- 3. Restriction ---")
    L.append("    // P7a: Z-restriction unrolled per n-column into 6 explicit d_zrestr6")
    L.append("    // chains (m = 0..5, 6 independent loads each); Y-restriction fused")
    L.append("    // immediately per column (identical left-assoc pairing as the original")
    L.append("    // intermediate-array read order). The 6x6 intermediate local round-trip")
    L.append("    // eliminated; registers bounded per column. Association order preserved")
    L.append("    // verbatim -> bit-exact.")
    L.append("    double tmp1[6];")
    for n in range(6):
        L.append("")
        L.append(f"    {{  // n = {n}: cur_if = if_fine - 2 + {n}")
        L.append(f"        int cur_if = if_fine - 2 + {n};")
        for m in range(6):
            L.append(f"        double z{m} = d_zrestr6(extf, funf, SoA, cur_if, {canon_j[m]}, kf_fine);")
        L.append("        double val = 0.0;")
        L.append("        val += C_RESTRICT[0] * (z0 + z5);")
        L.append("        val += C_RESTRICT[1] * (z1 + z4);")
        L.append("        val += C_RESTRICT[2] * (z2 + z3);")
        L.append(f"        tmp1[{n}] = val;")
        L.append("    }")
    return "\n".join(L) + "\n"

NEW_BODY = build_new_body()


def main():
    cand = pathlib.Path(sys.argv[1])
    f = cand / "src/prolongrestrict_cell_gpu.cu"
    s = f.read_text()
    h = sha(f)
    print(f"[P7] candidate file hash: {h}")
    if h != FORMAL_HASH:
        print(f"[P7] ABORT: hash != formal {FORMAL_HASH}")
        sys.exit(1)
    if "d_zrestr6" in s:
        print("[P7] ABORT: re-entrancy guard (d_zrestr6 already present)")
        sys.exit(1)
    if s.count(OLD_BLOCK) != 1:
        print(f"[P7] ABORT: OLD_BLOCK occurrence count = {s.count(OLD_BLOCK)} (want 1)")
        sys.exit(1)

    # ---- invariant 1: helper body must be verbatim the original inner val chain ----
    # (extracted from OLD_BLOCK itself: the restrict3 section-3 text, count==1)
    inner = re.search(r"            double val = 0\.0;\n(.*?)\n\s*\n            tmp2\[m\]\[n\] = val;", OLD_BLOCK, re.S)
    if not inner:
        print("[P7] ABORT: cannot extract original inner block from OLD_BLOCK")
        sys.exit(1)
    inner_lines = [l.strip() for l in inner.group(1).splitlines() if "val += C_RESTRICT" in l]
    helper_lines = [l.strip() for l in HELPER.splitlines() if "val += C_RESTRICT" in l]
    if inner_lines != helper_lines:
        print("[P7] ABORT: helper body != original inner block (verbatim mismatch)")
        for a, b in zip(inner_lines, helper_lines):
            if a != b:
                print(f"  ORIG: {a!r}\n  HELPER: {b!r}")
        sys.exit(1)
    print(f"[P7] OK helper verbatim match ({len(helper_lines)} C_RESTRICT lines)")

    # ---- invariant 2: Y pairing (0,5)(1,4)(2,3) with C_RESTRICT[0..2] in order ----
    o_y = re.findall(r"val \+= C_RESTRICT\[(\d)\] \* \(tmp2\[(\d)\]\[n\] \+ tmp2\[(\d)\]\[n\]\);", OLD_BLOCK)
    n_y = re.findall(r"val \+= C_RESTRICT\[(\d)\] \* \(z(\d) \+ z(\d)\);", NEW_BODY)
    exp = [("0", "0", "5"), ("1", "1", "4"), ("2", "2", "3")]
    if o_y != exp:
        print(f"[P7] ABORT: orig Y pairing unexpected {o_y}")
        sys.exit(1)
    if n_y != exp * 6:
        print(f"[P7] ABORT: new Y pairing unexpected {n_y}")
        sys.exit(1)
    print("[P7] OK Y pairing preserved (z0+z5 / z1+z4 / z2+z3)")

    # ---- invariant 3: 36 chains + per-column m-mapping (z<m> <-> jf_fine-2+m) ----
    calls = re.findall(r"d_zrestr6\(extf, funf, SoA, cur_if, (.*?), kf_fine\)", NEW_BODY)
    calls = [c.strip() for c in calls]
    if len(calls) != 36:
        print(f"[P7] ABORT: d_zrestr6 call count = {len(calls)} (want 36)")
        sys.exit(1)
    canon = {0: "jf_fine - 2", 1: "jf_fine - 1", 2: "jf_fine", 3: "jf_fine + 1", 4: "jf_fine + 2", 5: "jf_fine + 3"}
    per_col = [calls[i * 6:(i + 1) * 6] for i in range(6)]
    for col in per_col:
        if col != [canon[m] for m in range(6)]:
            print(f"[P7] ABORT: column m-map mismatch {col}")
            sys.exit(1)
    print("[P7] OK 36 chains, per-column m-order jf_fine-2..+3 correct")

    # ---- invariant 4: n-mapping (if_fine-2+n, n=0..5) ----
    n_ics = re.findall(r"int cur_if = if_fine - 2 \+ (\d);", NEW_BODY)
    if n_ics != [str(i) for i in range(6)]:
        print(f"[P7] ABORT: cur_if n-mapping = {n_ics} (want 0..5)")
        sys.exit(1)
    print("[P7] OK n-mapping if_fine-2+n for n=0..5")

    # ---- apply ----
    s2 = s.replace(OLD_BLOCK, NEW_BODY, 1)
    anchor = "__device__ __forceinline__ void d_restrict3_device("
    assert s2.count(anchor) == 1, "restrict3 anchor not unique"
    s2 = s2.replace(anchor, HELPER + anchor, 1)

    # ---- invariant 5: no tmp2 refs in d_restrict3_device; prolong3 untouched ----
    r3 = s2.split("__device__ __forceinline__ void d_restrict3_device(")[1]
    r3 = r3.split("__global__ __launch_bounds__")[0]
    if "tmp2[" in r3 or "double tmp2" in r3:
        print("[P7] ABORT: tmp2 still referenced in d_restrict3_device")
        sys.exit(1)
    p3 = s2.split("__device__ __forceinline__ void d_prolong3_device(")[1]
    p3 = p3.split("__device__ __forceinline__ void d_restrict3_device(")[0]
    if "double tmp2[6][6];" not in p3:
        print("[P7] ABORT: prolong3 tmp2 missing (scope violation)")
        sys.exit(1)
    print("[P7] OK tmp2 removed from restrict3, prolong3 untouched")

    f.write_text(s2)
    h2 = sha(f)
    print(f"[P7] APPLIED -> {f}")
    print(f"[P7] new hash: {h2}")
    print("[P7] DONE")

if __name__ == "__main__":
    main()
