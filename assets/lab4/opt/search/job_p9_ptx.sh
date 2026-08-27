#!/usr/bin/env bash
# iter18 P9 — PTX call-target identification (base deployed state)
# Compiles sommerfeld + fmisc TUs with -ptx (frontend only, no ptxas) and
# lists call.uni callees inside each kernel. Answers:
#   (a) what are the 126 CALLs in sommerfeld_rout_kernel?
#   (b) is global_interp_device actually ABI-called from global_interp_kernel?
set -uo pipefail
BASE=/home/h3240101033/lab4-gpu
EV=/home/h3240101033/lab4-gpu-cand-p9-20260824-111622/evidence/p9-ptx
mkdir -p "$EV"
exec > "$EV/job-ptx.log" 2>&1
echo "=== START P9 PTX $(date -u) ==="
NVCC=/usr/local/cuda-13.3/bin/nvcc
DEFS="-DMPI_CUDA_AWARE=0 -DUSE_GPU -Dfortran3 -Dnewc"
FLAGS="-O3 -std=c++14 --generate-code=arch=compute_80,code=sm_80 -rdc=true"
INC="-I$BASE/src -isystem /usr/lib/x86_64-linux-gnu/openmpi/include -isystem /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -isystem /usr/local/cuda-13.3/targets/x86_64-linux/include"

$NVCC -forward-unknown-to-host-compiler -ccbin=/usr/bin/g++-13 $DEFS $INC $FLAGS -x cu -rdc=true -ptx "$BASE/src/sommerfeld_rout_gpu.cu" -o "$EV/somm.ptx" > "$EV/ptx-somm.log" 2>&1 || { echo COMPILE_FAIL; tail -5 "$EV/ptx-somm.log"; }
$NVCC -forward-unknown-to-host-compiler -ccbin=/usr/bin/g++-13 $DEFS $INC $FLAGS -x cu -rdc=true -ptx "$BASE/src/fmisc_gpu.cu" -o "$EV/fmisc.ptx" > "$EV/ptx-fmisc.log" 2>&1 || { echo COMPILE_FAIL; tail -5 "$EV/ptx-fmisc.log"; }

python3 - "$EV" <<'PY'
import re, sys
from collections import Counter
ev = sys.argv[1]
def kern_calls(path, name):
    s = open(path).read()
    m = re.search(re.escape(name) + r".*?(?=\n\.visible|\Z)", s, re.S)
    body = m.group(0) if m else ""
    calls = re.findall(r"call\.uni\s*\([^)]*\)\s*,\s*(\S+);", body)
    return body, Counter(calls)
for f, names in (("somm.ptx", ["_Z22sommerfeld_rout_kernel", "_Z25sommerfeld_routbam_kernel"]),
                 ("fmisc.ptx", ["_Z20global_interp_kernel", "_Z24global_interp_amr_kernel"])):
    for nm in names:
        body, cc = kern_calls(f"{ev}/{f}", nm)
        print(f"--- {f} {nm}: ptx bytes={len(body)} call.uni={sum(cc.values())} ---")
        for callee, c in cc.most_common(10):
            print(f"   {callee}: {c}")
        print(f"   div.rn.f64={len(re.findall(r'div\.rn\.f64', body))} sqrt.rn.f64={len(re.findall(r'sqrt\.rn\.f64', body))} rcp={len(re.findall(r'rcp\.', body))}")
# who calls global_interp_device?
for f in ("somm.ptx", "fmisc.ptx"):
    s = open(f"{ev}/{f}").read()
    for m in re.finditer(r"call\.uni\s*\([^)]*\)\s*,\s*(\S+);", s):
        if "global_interp_device" in m.group(1):
            print(f"CALLER of global_interp_device found in {f}")
print("=== DONE P9 PTX ===")
PY
echo "=== DONE P9 PTX $(date -u) ==="
