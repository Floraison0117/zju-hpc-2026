#!/usr/bin/env python3
# iter15 P6 — prolong3 6x6 full unroll + Z/Y column fusion (P6a, bit-exact)
#
# Mechanism (ncu3 evidence: fixed-latency execution dependency 31.3%, 66 regs,
# 0 spill, stack 336B = tmp2[6][6]+tmp1[6] materialized in local memory):
#   1. The 6x6 Z-interp is fully unrolled into 36 explicit d_zinterp6 chains
#      (6 independent chains z0..z5 per n-column -> 36 independent loads can
#      be issued in parallel, breaking the false serial dependency).
#   2. Y-interp is fused per n-column (identical left-assoc sum as the
#      original tmp1[n] computation) -> the tmp2[6][6] local array (288B
#      stack round-trip = 36 STL + 36 LDL per thread) is eliminated.
#   3. Association order is preserved verbatim (same C_PROLONG index order,
#      same cxI_k offsets, same left-assoc val += chains) -> bit-exact.
#
# Single variable: prolongrestrict_cell_gpu.cu only (d_prolong3_device).
# Scope guard: d_restrict3_device's own tmp2[6][6] is untouched.
import sys, pathlib, hashlib, re

FORMAL_HASH = "a81cd33ebaabb8f569847ef26e241cc4f2e8ca4ff6df0beeb410bdacbb19b020"

def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

OLD_BLOCK = """    // --- 3. Interpolation ---
    double tmp2[6][6];
    double tmp1[6];

    // Z-Direction Interpolation
    for (int m = 0; m < 6; m++) {
        for (int n = 0; n < 6; n++) {
            int cur_ic = cxI_i - 2 + n;
            int cur_jc = cxI_j - 2 + m;
            
            double val = 0.0;
            // 1-based indices passed to d_get_sym_val
            if (k_even) {
                val += C_PROLONG[0] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k - 2, SoA);
                val += C_PROLONG[1] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k - 1, SoA);
                val += C_PROLONG[2] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k    , SoA);
                val += C_PROLONG[3] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k + 1, SoA);
                val += C_PROLONG[4] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k + 2, SoA);
                val += C_PROLONG[5] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k + 3, SoA);
            } else {
                val += C_PROLONG[5] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k - 2, SoA);
                val += C_PROLONG[4] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k - 1, SoA);
                val += C_PROLONG[3] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k    , SoA);
                val += C_PROLONG[2] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k + 1, SoA);
                val += C_PROLONG[1] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k + 2, SoA);
                val += C_PROLONG[0] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k + 3, SoA);
            }
            tmp2[m][n] = val;
        }
    }

    // Y-Direction Interpolation
    for (int n = 0; n < 6; n++) {
        double val = 0.0;
        if (j_even) {
            val += C_PROLONG[0] * tmp2[0][n] + C_PROLONG[1] * tmp2[1][n] + C_PROLONG[2] * tmp2[2][n] +
                   C_PROLONG[3] * tmp2[3][n] + C_PROLONG[4] * tmp2[4][n] + C_PROLONG[5] * tmp2[5][n];
        } else {
            val += C_PROLONG[5] * tmp2[0][n] + C_PROLONG[4] * tmp2[1][n] + C_PROLONG[3] * tmp2[2][n] +
                   C_PROLONG[2] * tmp2[3][n] + C_PROLONG[1] * tmp2[4][n] + C_PROLONG[0] * tmp2[5][n];
        }
        tmp1[n] = val;
    }
"""

HELPER = """// Z-direction 6-term interpolation for one (m,n) point (P6a helper).
// Body is VERBATIM the original d_prolong3_device inner if/else block:
// same C_PROLONG index order, same cxI_k offsets, same left-assoc val +=
// chain => bit-exact identical tmp2[m][n] values. __forceinline__ so the
// 36 call sites unroll into 36 explicit chains with 6 independent loads each.
__device__ __forceinline__ double d_zinterp6(
    const int* extc, const double* func, const double SoA[3],
    int cur_ic, int cur_jc, int cxI_k, bool k_even
) {
    double val = 0.0;
    // 1-based indices passed to d_get_sym_val
    if (k_even) {
        val += C_PROLONG[0] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k - 2, SoA);
        val += C_PROLONG[1] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k - 1, SoA);
        val += C_PROLONG[2] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k    , SoA);
        val += C_PROLONG[3] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k + 1, SoA);
        val += C_PROLONG[4] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k + 2, SoA);
        val += C_PROLONG[5] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k + 3, SoA);
    } else {
        val += C_PROLONG[5] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k - 2, SoA);
        val += C_PROLONG[4] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k - 1, SoA);
        val += C_PROLONG[3] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k    , SoA);
        val += C_PROLONG[2] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k + 1, SoA);
        val += C_PROLONG[1] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k + 2, SoA);
        val += C_PROLONG[0] * d_symmetry_bd_1b(3, extc, func, cur_ic, cur_jc, cxI_k + 3, SoA);
    }
    return val;
}

"""

def build_new_body():
    """6 explicit n-column blocks, each with 6 explicit d_zinterp6 chains + Y-sum.
    Same association as original (Z 6-term left-assoc chain, then Y left-assoc)."""
    canon_j = {0: "cxI_j - 2", 1: "cxI_j - 1", 2: "cxI_j", 3: "cxI_j + 1", 4: "cxI_j + 2", 5: "cxI_j + 3"}
    L = []
    L.append("    // --- 3. Interpolation ---")
    L.append("    // P6a: 6x6 Z-interp fully unrolled + Y-interp fused per n-column.")
    L.append("    // 36 explicit d_zinterp6 chains (6 per column), each with 6 independent")
    L.append("    // loads -> 36 loads schedulable in parallel; the 6x6 intermediate array")
    L.append("    // (288B stack round-trip) is eliminated. Association order preserved")
    L.append("    // verbatim -> bit-exact.")
    L.append("    double tmp1[6];")
    for n in range(6):
        L.append("")
        L.append(f"    {{  // n = {n}: cur_ic = cxI_i - 2 + {n}")
        L.append(f"        int cur_ic = cxI_i - 2 + {n};")
        for m in range(6):
            L.append(f"        double z{m} = d_zinterp6(extc, func, SoA, cur_ic, {canon_j[m]}, cxI_k, k_even);")
        L.append("        double val = 0.0;")
        L.append("        if (j_even) {")
        L.append("            val += C_PROLONG[0] * z0 + C_PROLONG[1] * z1 + C_PROLONG[2] * z2 +")
        L.append("                   C_PROLONG[3] * z3 + C_PROLONG[4] * z4 + C_PROLONG[5] * z5;")
        L.append("        } else {")
        L.append("            val += C_PROLONG[5] * z0 + C_PROLONG[4] * z1 + C_PROLONG[3] * z2 +")
        L.append("                   C_PROLONG[2] * z3 + C_PROLONG[1] * z4 + C_PROLONG[0] * z5;")
        L.append("        }")
        L.append(f"        tmp1[{n}] = val;")
        L.append("    }")
    return "\n".join(L) + "\n"

NEW_BODY = build_new_body()


def extract_pairs(block):
    """extract (C_PROLONG index, cxI_k offset) tuples in order for a given branch text"""
    pat = re.compile(r"val \+= C_PROLONG\[(\d)\] \* d_symmetry_bd_1b\(3, extc, func, cur_ic, cur_jc, cxI_k ([+-] )?(\d), SoA\);")
    out = []
    for ln in block.splitlines():
        m = pat.search(ln)
        if m:
            idx = int(m.group(1))
            off = int(m.group(3)) * (1 if m.group(2) != "-" else -1)
            out.append((idx, off))
    return out

def main():
    cand = pathlib.Path(sys.argv[1])
    f = cand / "src/prolongrestrict_cell_gpu.cu"
    s = f.read_text()
    h = sha(f)
    print(f"[P6] candidate file hash: {h}")
    if h != FORMAL_HASH:
        print(f"[P6] ABORT: hash != formal {FORMAL_HASH}")
        sys.exit(1)
    if "d_zinterp6" in s:
        print("[P6] ABORT: re-entrancy guard (d_zinterp6 already present)")
        sys.exit(1)
    if s.count(OLD_BLOCK) != 1:
        print(f"[P6] ABORT: OLD_BLOCK occurrence count = {s.count(OLD_BLOCK)} (want 1)")
        sys.exit(1)

    # ---- invariant 1: helper body must be verbatim the original inner if/else ----
    inner = re.search(r"            double val = 0\.0;\n(.*?)\n            \}\n            tmp2\[m\]\[n\] = val;", s, re.S)
    if not inner:
        print("[P6] ABORT: cannot extract original inner block")
        sys.exit(1)
    inner_lines = [l.strip() for l in inner.group(1).splitlines() if "val += C_PROLONG" in l]
    helper_lines = [l.strip() for l in HELPER.splitlines() if "val += C_PROLONG" in l]
    if inner_lines != helper_lines:
        print("[P6] ABORT: helper body != original inner block (verbatim mismatch)")
        for a, b in zip(inner_lines, helper_lines):
            if a != b:
                print(f"  ORIG: {a!r}\n  HELPER: {b!r}")
        sys.exit(1)
    print(f"[P6] OK helper verbatim match ({len(helper_lines)} C_PROLONG lines)")

    # ---- invariant 2: Y-sum z-order matches original tmp2[0..5][n] order ----
    def ybranches(block):
        # extract j_even / j_odd C_PROLONG lines from the Y-interp (unique `if (j_even)`)
        m = re.search(r"if \(j_even\) \{(.*?)\} else \{(.*?)\}\n        tmp1\[(?:n|\d)\]", block, re.S)
        assert m, "Y-interp j_even/j_odd branches not found"
        even = [l.strip() for l in m.group(1).splitlines() if "C_PROLONG" in l]
        odd = [l.strip() for l in m.group(2).splitlines() if "C_PROLONG" in l]
        return even, odd
    o_e_l, o_o_l = ybranches(OLD_BLOCK)
    n_e_l, n_o_l = ybranches(NEW_BODY)
    def zmap(lines, old):
        if old:
            pat = re.compile(r"C_PROLONG\[(\d)\] \* tmp2\[(\d)\]\[n\]")
        else:
            pat = re.compile(r"C_PROLONG\[(\d)\] \* z(\d)")
        out = []
        for l in lines:
            out += pat.findall(l)
        return out
    o_e, o_o = zmap(o_e_l, True), zmap(o_o_l, True)
    n_e, n_o = zmap(n_e_l, False), zmap(n_o_l, False)
    # original j_even reads tmp2[0..5][n] == z0..z5 -> expect (j, i=j)
    exp_e = [(str(j), str(j)) for j in range(6)]
    exp_o = [(str(5 - j), str(j)) for j in range(6)]
    assert o_e == exp_e and o_o == exp_o, f"orig Y order unexpected: {o_e} {o_o}"
    assert n_e == exp_e and n_o == exp_o, f"new Y order unexpected: {n_e} {n_o}"
    print("[P6] OK Y-interp order preserved (even z0..z5 / odd z5..z0)")

    # ---- invariant 3: 36 explicit chains + j-mapping (z<m> <-> cur_jc = cxI_j-2+m) ----
    calls = re.findall(r"d_zinterp6\(extc, func, SoA, cur_ic, (.*?), cxI_k, k_even\)", NEW_BODY)
    calls = [c.strip() for c in calls]
    if len(calls) != 36:
        print(f"[P6] ABORT: d_zinterp6 call count = {len(calls)} (want 36)")
        sys.exit(1)
    canon = {0: "cxI_j - 2", 1: "cxI_j - 1", 2: "cxI_j", 3: "cxI_j + 1", 4: "cxI_j + 2", 5: "cxI_j + 3"}
    per_col = [calls[i * 6:(i + 1) * 6] for i in range(6)]
    for col in per_col:
        if col != [canon[m] for m in range(6)]:
            print(f"[P6] ABORT: column j-map mismatch {col}")
            sys.exit(1)
    print("[P6] OK 36 chains, per-column m-order cxI_j-2..+3 correct")

    # ---- invariant 4: cur_ic n-mapping (cxI_i-2+n, n=0..5) ----
    n_ics = re.findall(r"int cur_ic = cxI_i - 2 \+ (\d);", NEW_BODY)
    if n_ics != [str(i) for i in range(6)]:
        print(f"[P6] ABORT: cur_ic n-mapping = {n_ics} (want 0..5)")
        sys.exit(1)
    print("[P6] OK n-mapping cxI_i-2+n for n=0..5")

    # ---- apply ----
    s2 = s.replace(OLD_BLOCK, NEW_BODY, 1)
    # insert helper before d_prolong3_device
    anchor = "__device__ __forceinline__ void d_prolong3_device("
    assert s2.count(anchor) == 1, "prolong3 anchor not unique"
    s2 = s2.replace(anchor, HELPER + anchor, 1)

    # ---- invariant 5: no tmp2 code refs left in d_prolong3_device (restrict3 untouched) ----
    p3 = s2.split("__device__ __forceinline__ void d_prolong3_device(")[1]
    p3 = p3.split("__device__ __forceinline__ void d_restrict3_device(")[0]
    if "tmp2[" in p3 or "double tmp2" in p3:
        print("[P6] ABORT: tmp2 still referenced in d_prolong3_device")
        sys.exit(1)
    # restrict3 must still contain its own tmp2
    r3 = s2.split("__device__ __forceinline__ void d_restrict3_device(")[1]
    if "double tmp2[6][6];" not in r3:
        print("[P6] ABORT: restrict3 tmp2 missing (scope violation)")
        sys.exit(1)
    print("[P6] OK tmp2 removed from prolong3, restrict3 untouched")

    f.write_text(s2)
    h2 = sha(f)
    print(f"[P6] APPLIED -> {f}")
    print(f"[P6] new hash: {h2}")
    print("[P6] DONE")

if __name__ == "__main__":
    main()
