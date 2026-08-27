#!/usr/bin/env python3
# iter17 P8 PROBE — fmisc_gpu.cu helper chain forceinline (P6b pattern extension)
#
# Diagnostic: deployed build uses CUDA_SEPARABLE_COMPILATION (-rdc=true), so the
# device helper chain declared non-inline in fmisc.h and defined in fmisc_gpu.cu
# is ABI-called from cross-TU call sites. P6b (iter15) proved that forceinline
# for d_symmetry_bd_1b removed ABI call/return overhead and gave -13.6% on the
# 100-step run (job 155105, 778.24s). P8 applies the same transformation to the
# remaining 3 non-forceinline helpers:
#   - polint         (Neville interpolation, called ordn^2+ordn+1 times per
#                     d_polin3_1b call)
#   - d_polin3_1b    (3D polynomial interpolation; ABI-called cross-TU from
#                     sommerfeld_rout_kernel, sommerfeld_rout_gpu.cu:138)
#   - d_decide3d     (3D symmetry decision + ya[] fill; ABI-called cross-TU from
#                     sommerfeld_rout_kernel, sommerfeld_rout_gpu.cu:132)
# Both sommerfeld_rout_kernel (3.8% runtime) and global_interp_device (3.7%
# runtime) consume this chain.
#
# global_interp_device stays non-forceinline in this stage (bigger body, ya[216]
# locals); its calls to the now-forceinline helpers inline inside its own frame.
# Stage B (forceinline global_interp_device itself) is decided from ptxas.
#
# Mechanism: move bodies VERBATIM from fmisc_gpu.cu into fmisc.h as
# __forceinline__ definitions (extracted programmatically with brace counting,
# only the signature gains __forceinline__), remove the definitions from
# fmisc_gpu.cu. Bit-exact by construction (identical arithmetic, no
# re-association). Header prerequisites (MAX_ORDN / GPU_DEBUG_PRINT guards +
# gpu_stop) are added before the moved bodies.
import sys, pathlib, hashlib

FMISC_H_EXPECT = "8ff0acaf69bc12e9ab763662e399624b499ee4d92d17312a769dcb6c21152aa2"
FMISC_CU_EXPECT = "3fab96a9c05425321c1bad7017c4063dd45a1445c2eb6444e70df71e1786c1e0"

# Host TU compatibility: moved bodies use bare max/min (CUDA device builtins
# only). Host .C/.cpp TUs (g++) parse fmisc.h's device section; give them std
# names. Device pass (__CUDA_ARCH__) keeps CUDA builtins -> no ambiguity.
INC_COMPAT_OLD = "#ifdef USE_GPU\n#include <cuda_runtime.h>\n"
INC_COMPAT_NEW = (
"#ifdef USE_GPU\n"
"#include <cuda_runtime.h>\n"
"#include <stdlib.h>\n"
"#include <math.h>\n"
"#include <algorithm>\n"
"// P8: helper bodies in this header use bare max/min/abs/fabs (CUDA device\n"
"// builtins). Host TU passes need explicit std names.\n"
"#if !defined(__CUDA_ARCH__)\n"
"using std::max;\n"
"using std::min;\n"
"#endif\n"
)

# Exact declaration block in fmisc.h (tabs preserved; verified via cat -A)
HDR_OLD = (
"__device__ void d_polin3_1b(\n"
"\tconst double* x1a, const double* x2a, const double* x3a,\n"
"\tconst double* ya, double x1, double x2, double x3,\n"
"\tdouble& y, double& dy, int ordn\n"
");\n"
"\n"
"__device__ bool d_decide3d(\n"
"\tconst int ex[3], const double* f, const double* fpi,\n"
"\tconst int cxB[3], const int cxT[3], const double SoA[3],\n"
"\tdouble* ya, int ordn, int Symmetry\n"
");\n"
)

HDR_PRE = (
"#ifndef MAX_ORDN\n"
"#define MAX_ORDN 6\n"
"#endif\n"
"#ifndef GPU_DEBUG_PRINT\n"
"#define GPU_DEBUG_PRINT 0\n"
"#endif\n"
"\n"
"__device__ __forceinline__ void gpu_stop() {\n"
"#if GPU_STRICT_STOP\n"
"    asm(\"trap;\");\n"
"#endif\n"
"}\n"
"\n"
"// P8 PROBE: moved from fmisc_gpu.cu + __forceinline__ (bodies verbatim) so\n"
"// sommerfeld_rout_kernel + global_interp_device's d_decide3d/d_polin3_1b/\n"
"// polint call sites inline instead of ABI-calling (P6b pattern, iter15 -13.6%).\n"
)

def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def extract_body(text, marker):
    """Return function text from the line starting with `marker` through its
    balanced closing brace (verbatim, no trailing newline)."""
    lines = text.split("\n")
    idx = [k for k, l in enumerate(lines) if l.startswith(marker)]
    if len(idx) != 1:
        raise SystemExit(f"[P8] ABORT: marker {marker!r} count={len(idx)}")
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
        raise SystemExit(f"[P8] ABORT: unbalanced braces near {marker!r}")
    return "\n".join(lines[i:j + 1])

def main():
    cand = pathlib.Path(sys.argv[1])
    h = cand / "src/fmisc.h"
    g = cand / "src/fmisc_gpu.cu"
    sh, sg = sha(h), sha(g)
    print(f"[P8] fmisc.h      hash: {sh} (expect {FMISC_H_EXPECT})")
    print(f"[P8] fmisc_gpu.cu hash: {sg} (expect {FMISC_CU_EXPECT})")
    if sh != FMISC_H_EXPECT or sg != FMISC_CU_EXPECT:
        raise SystemExit("[P8] ABORT: hash drift from deployed baseline")
    hs = h.read_text()
    gs = g.read_text()
    if "P8 PROBE" in hs:
        raise SystemExit("[P8] ABORT: re-entrancy guard")
    if hs.count(HDR_OLD) != 1:
        raise SystemExit(f"[P8] ABORT: fmisc.h declaration block count={hs.count(HDR_OLD)}")
    if hs.count(INC_COMPAT_OLD) != 1:
        raise SystemExit(f"[P8] ABORT: fmisc.h include block count={hs.count(INC_COMPAT_OLD)}")
    hs = hs.replace(INC_COMPAT_OLD, INC_COMPAT_NEW, 1)

    # --- extract bodies verbatim from fmisc_gpu.cu ---
    gpu_stop = extract_body(gs, "__device__ __forceinline__ void gpu_stop() {")
    polint   = extract_body(gs, "__device__ void polint(")
    dpolin   = extract_body(gs, "__device__ void d_polin3_1b(")
    ddecide  = extract_body(gs, "__device__ bool d_decide3d(")

    # signature forceinline injection
    def fi(body, sig_old, sig_new):
        if body.count(sig_old) != 1:
            raise SystemExit(f"[P8] ABORT: sig {sig_old!r} count={body.count(sig_old)}")
        return body.replace(sig_old, sig_new, 1)

    polint_fi  = fi(polint,  "__device__ void polint(",  "__device__ __forceinline__ void polint(")
    dpolin_fi  = fi(dpolin,  "__device__ void d_polin3_1b(", "__device__ __forceinline__ void d_polin3_1b(")
    ddecide_fi = fi(ddecide, "__device__ bool d_decide3d(", "__device__ __forceinline__ bool d_decide3d(")

    # --- fmisc.h: replace declarations with definitions ---
    hdr_new = HDR_PRE + polint_fi + "\n\n" + dpolin_fi + "\n\n" + ddecide_fi + "\n"
    hs2 = hs.replace(HDR_OLD, hdr_new, 1)

    # --- fmisc_gpu.cu: remove moved definitions (body + one blank line) ---
    for body, name in ((gpu_stop, "gpu_stop"), (polint, "polint"),
                       (dpolin, "d_polin3_1b"), (ddecide, "d_decide3d")):
        blk = body + "\n\n"
        if gs.count(blk) != 1:
            raise SystemExit(f"[P8] ABORT: fmisc_gpu.cu removal block {name} count={gs.count(blk)}")
        gs = gs.replace(blk, "", 1)

    # --- post-conditions ---
    for name in ("polint", "d_polin3_1b", "d_decide3d"):
        if f"__device__ void {name}(" in gs or f"__device__ bool {name}(" in gs:
            raise SystemExit(f"[P8] ABORT: {name} definition still in fmisc_gpu.cu")
    if "__device__ __forceinline__ void gpu_stop() {" in gs:
        raise SystemExit("[P8] ABORT: gpu_stop definition still in fmisc_gpu.cu")
    # callers must remain
    for call in ("global_interp_device(", "d_polin3_1b(", "d_decide3d(", "X_at_1b("):
        if call not in gs:
            raise SystemExit(f"[P8] ABORT: caller {call!r} vanished from fmisc_gpu.cu")
    for body in (polint_fi, dpolin_fi, ddecide_fi, gpu_stop):
        if body not in hs2:
            raise SystemExit("[P8] ABORT: moved body missing from fmisc.h")

    h.write_text(hs2)
    g.write_text(gs2 := gs)
    print(f"[P8] APPLIED fmisc.h      -> {sha(h)}")
    print(f"[P8] APPLIED fmisc_gpu.cu -> {sha(g)}")
    print("[P8] DONE")

if __name__ == "__main__":
    main()
