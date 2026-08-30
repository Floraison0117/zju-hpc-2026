#!/bin/bash
# scratch: build + install the custom op ON THE LOGIN NODE (no NPU needed for
# compilation). Installs into ~/custom_opp (shared NFS home), which the
# lab3p5 job containers see at runtime. Avoids the ~8 min in-job rebuild.
# Usage: bash scratch/build_login.sh
set -eo pipefail 2>/dev/null || set -e
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/env.sh"
: "${ASCEND_TOOLKIT_HOME:?env.sh did not set ASCEND_TOOLKIT_HOME}"
# Different load-balanced login backends expose different default PATHs; make
# the compiler tools (asc_opc etc.) resolvable everywhere.
export PATH="$ASCEND_TOOLKIT_HOME/compiler/bin:$ASCEND_TOOLKIT_HOME/aarch64-linux/bin:$ASCEND_TOOLKIT_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$ASCEND_TOOLKIT_HOME/aarch64-linux/lib64:$ASCEND_TOOLKIT_HOME/aarch64-linux/lib:$ASCEND_TOOLKIT_HOME/runtime/lib64:$ASCEND_TOOLKIT_HOME/aarch64-linux/lib64/device/lib64:$ASCEND_TOOLKIT_HOME/aarch64-linux/devlib/linux/aarch64:$ASCEND_TOOLKIT_HOME/aarch64-linux/devlib:$LD_LIBRARY_PATH"

OP_DIR="$ROOT/src/ascendc"
echo "=== [build_login] patch op CMakePresets -> system CANN ==="
python3 - <<PY
import json, os
p = os.path.join("$OP_DIR", "CMakePresets.json")
cfg = json.load(open(p))
cv = cfg["configurePresets"][0]["cacheVariables"]
cv["ASCEND_CANN_PACKAGE_PATH"] = {"type": "PATH", "value": "$ASCEND_TOOLKIT_HOME"}
cv["ASCEND_PYTHON_EXECUTABLE"] = {"type": "STRING", "value": "python3"}
json.dump(cfg, open(p, "w"), indent=4)
print("[build_login] CANN ->", cv["ASCEND_CANN_PACKAGE_PATH"]["value"])
PY

echo "=== [build_login] build + install custom op ==="
cd "$OP_DIR"
rm -rf build_out
export ASCEND_HOME_PATH="$ASCEND_TOOLKIT_HOME"
export DDK_PATH="$ASCEND_HOME_PATH"
export NPU_HOST_LIB="$ASCEND_HOME_PATH/$(arch)-$(uname -s | tr '[:upper:]' '[:lower:]')/devlib"
bash build_op.sh 2>&1 | tail -8
echo "=== [build_login] installed kernels under ~/custom_opp: ==="
find "$HOME/custom_opp/vendors/customize/op_impl/ai_core/tbe/kernel/ascend910b/fused_add_rms_norm" -name '*.json' -newer "$OP_DIR/op_kernel/fused_add_rms_norm.cpp" 2>/dev/null | head -3
echo "=== [build_login] done ==="
