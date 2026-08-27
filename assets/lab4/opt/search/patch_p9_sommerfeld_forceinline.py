#!/usr/bin/env python3
# iter18 P9 PROBE — is_sommerfeld_boundary forceinline (P6b/P8 pattern, 3rd strike)
#
# Diagnostic: current deployed sommerfeld_rout_gpu.cu (hash e9ea7810) defines
# is_sommerfeld_boundary as plain `__device__` (line 14). Under
# CUDA_SEPARABLE_COMPILATION (-rdc=true) nvcc may outline it (CALL.ABS.NOINC at
# sommerfeld_rout_kernel:63 and sommerfeld_routbam_kernel:164, once per thread),
# adding call/return + 17-arg marshaling latency on the hot boundary path.
# P6b (iter15, -13.6%) and P8 (iter17, -5.3%) proved the forceinline lever for
# cross-TU helpers; this is the same mechanism applied to the last non-inline
# device helper.
#
# DEVIATION from the original .h-move plan (documented, justified):
#   is_sommerfeld_boundary is DEFINED in sommerfeld_rout_gpu.cu and called ONLY
#   from the two kernels in the SAME TU (lines 63/164). No cross-TU caller
#   exists (grep verified). Moving the body to a header would require:
#     (a) NO_SYMM/EQ_SYMM/OCTANT constexprs (defined locally at .cu:12, absent
#         from any header) to be visible at the header's parse point, and
#     (b) fmisc.h is parsed by 7 host .C/.cpp TUs; sommerfeld_rout.h by 3 host
#         .C files (bssn_class.C etc.) that include it BEFORE math.h, so the
#         body's fabs() would be undeclared at parse time.
#   In-place __forceinline__ (signature-only change, body VERBATIM) achieves
#   exactly the same inlining effect for the only TU that matters, with zero
#   header-pollution/parse risk. This is still a single-variable change.
#
# Invariants:
#   1. only sommerfeld_rout_gpu.cu is touched; all other src files must stay
#      byte-identical to the deployed baseline
#   2. function body extracted verbatim (brace-counted) and byte-identical
#      before/after the signature change
#   3. NO_SYMM/EQ_SYMM/OCTANT/CORRECTSTEP constexprs untouched
#   4. both call sites (lines 63/164) intact
import sys, pathlib, hashlib

SRC_EXPECT = "e9ea7810e7e2a2240dd4e7e114df0ffc8908b1bd1b556d2bec9f9b2604590920"
SIG_OLD = "__device__ bool is_sommerfeld_boundary("
SIG_NEW = "__device__ __forceinline__ bool is_sommerfeld_boundary("

CONTROL_EXPECT = {
    "src/fmisc.h":            "56ad30e41c347b9a97d22079707225383149c6ee69a458aa46893e89461be41f",
    "src/fmisc_gpu.cu":       "c7cc9f07a708b0afaffa4ff757fcd7913d00a71aa5a0252357cbfaa7abb3cf61",
    "src/sommerfeld_rout.h":  "08ba52697681c369f19544bb0d014a8bc6fe22596167424bece8f2fc33c5fece",
    "src/bssn_rhs_gpu.cu":    "4a4b2aabd2f2083d905f2ed838712d6c560f78fffe4bf4c64d6d8cd6741df8df",
    "src/derivatives.h":      "3ede464623e406e1709fe1e6d1bd95ab660838a865eabc0d827d0ad9d2ac8bd1",
    "src/lopsidediff.h":      "5dddaa7585ec48fd41124ad9aa7989f84acf585918ff709887f1088c6d261913",
    "src/kodiss.h":           "e88f9632d6d397ad58972ac96b1b640d519f538eb967cee405a8a3bff9a060fb",
}

def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def extract_body(text, marker):
    """Return function text from the line starting with `marker` through its
    balanced closing brace (verbatim, no trailing newline)."""
    lines = text.split("\n")
    idx = [k for k, l in enumerate(lines) if l.startswith(marker)]
    if len(idx) != 1:
        raise SystemExit(f"[P9] ABORT: marker {marker!r} count={len(idx)}")
    i = idx[0]
    depth = 0
    started = False
    j = i
    while j < len(lines):
        for ch in lines[j]:
            if ch == "{":
                depth += 1
                started = True
            elif ch == "}":
                depth -= 1
        if started and depth == 0:
            break
        j += 1
    if not started or depth != 0:
        raise SystemExit(f"[P9] ABORT: unbalanced braces near {marker!r}")
    return "\n".join(lines[i:j + 1])

def main():
    cand = pathlib.Path(sys.argv[1])
    src = cand / "src/sommerfeld_rout_gpu.cu"
    s = sha(src)
    print(f"[P9] sommerfeld_rout_gpu.cu hash: {s} (expect {SRC_EXPECT})")
    if s != SRC_EXPECT:
        raise SystemExit("[P9] ABORT: hash drift from deployed baseline")

    for rel, exp in CONTROL_EXPECT.items():
        h = sha(cand / rel)
        if h != exp:
            raise SystemExit(f"[P9] ABORT: control file {rel} drifted: {h}")
    print("[P9] control files (fmisc/fmisc_gpu/sommerfeld_rout.h/bssn_rhs/derivatives/lopsidediff/kodiss) byte-identical to deployed")

    text = src.read_text()
    if "__forceinline__ bool is_sommerfeld_boundary(" in text:
        raise SystemExit("[P9] ABORT: re-entrancy guard")
    if text.count(SIG_OLD) != 1:
        raise SystemExit(f"[P9] ABORT: signature count={text.count(SIG_OLD)}")
    body_old = extract_body(text, SIG_OLD)

    text2 = text.replace(SIG_OLD, SIG_NEW, 1)
    body_new = extract_body(text2, SIG_NEW)
    # bodies must be identical after normalizing the signature line
    norm = lambda b: b.replace(SIG_NEW, SIG_OLD, 1)
    if norm(body_new) != body_old:
        raise SystemExit("[P9] ABORT: body not verbatim (signature-normalized diff)")
    if body_new.count(SIG_NEW) != 1:
        raise SystemExit("[P9] ABORT: forceinline signature count in body != 1")

    # post-conditions on whole file
    if text2.count("is_sommerfeld_boundary(") != 3:  # def + 2 call sites
        raise SystemExit(f"[P9] ABORT: call-site count={text2.count('is_sommerfeld_boundary(')}")
    for kw in ("NO_SYMM = 0", "EQ_SYMM = 1", "OCTANT = 2", "CORRECTSTEP = 1"):
        if kw not in text2:
            raise SystemExit(f"[P9] ABORT: {kw} vanished")

    src.write_text(text2)
    print(f"[P9] APPLIED sommerfeld_rout_gpu.cu -> {sha(src)}")
    print("[P9] DONE")

if __name__ == "__main__":
    main()
