#!/usr/bin/env python3
# P5 (iter14): d_fdderivs_point fh values stored as float (register-halving probe).
# Single-variable change: the 61 h_* stencil-value locals double -> float.
#   - storage precision: float (24-bit mantissa, ~6e-8 rel err per value)
#   - all formula arithmetic stays double (implicit float->double promotion)
#   - fh lambda internals, r_* row sums, m4/m2, coefficients: all untouched (double)
# Scope guard: only transforms the tail starting at the d_fdderivs_point
# signature, so d_fderivs_point (identical h_jm2/h_km2 lines!) is untouched.
# Refuses to run twice or on a drifted baseline.
import re, sys, pathlib, hashlib

EXPECTED = "3ede464623e406e1709fe1e6d1bd95ab660838a865eabc0d827d0ad9d2ac8bd1"
EXPECT_CASTS = 61  # 13 diag (5+4+4) + 48 cross (16*3)

def main(argv):
    if len(argv) != 2:
        print("usage: patch_p5_fh_float.py <derivatives.h>"); return 2
    p = pathlib.Path(argv[1])
    s = p.read_text()
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    if h != EXPECTED:
        print(f"HASH_MISMATCH baseline={EXPECTED} got={h} (drift or already patched)")
        return 2

    marker = "__device__ __forceinline__ void d_fdderivs_point("
    if marker not in s:
        print("MARKER_NOT_FOUND"); return 2
    idx = s.index(marker)
    head, tail = s[:idx], s[idx:]
    if "(float)fh(" in tail:
        print("ALREADY_PATCHED"); return 2

    # match a WHOLE declaration line (multiple fh( per line, comma-separated)
    pat = re.compile(r"^    double h_\w+ = fh\(.*\);$", re.M)
    hits = pat.findall(tail)
    if len(hits) != 15:
        print(f"DECL_LINES_EXPECT_15 got={len(hits)}"); return 2

    def fix_line(line):
        nfh = line.count("fh(")
        line = line.replace("double ", "float ", 1)
        line = line.replace("fh(", "(float)fh(", nfh)
        return line

    new_tail, n = pat.sub(lambda m: fix_line(m.group(0)), tail), 0
    n = new_tail.count("(float)fh(")
    if n != EXPECT_CASTS:
        print(f"CASTS_EXPECT_{EXPECT_CASTS} got={n}"); return 2
    # invariants: no 'double h_' declaration may remain; every fh( call cast
    if re.search(r"^    double h_\w+ = fh\(", new_tail, re.M):
        print("LEFTOVER_DOUBLE_H"); return 2
    if new_tail.count("(float)fh(") != EXPECT_CASTS:
        print(f"CASTS_EXPECT_{EXPECT_CASTS} got={new_tail.count('(float)fh(')}"); return 2
    # every remaining bare 'fh(' must live on a comment line (e.g. '// fh() args')
    for ln in new_tail.splitlines():
        if "fh(" in ln and "(float)fh(" not in ln and not ln.strip().startswith("//"):
            print("UNCAST_CALL:", ln[:100]); return 2

    p.write_text(head + new_tail)
    newh = hashlib.sha256((head + new_tail).encode("utf-8")).hexdigest()
    print(f"P5_PATCH_OK casts={n} decl_lines={len(hits)} new_sha256={newh}")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
