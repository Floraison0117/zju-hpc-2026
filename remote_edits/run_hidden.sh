#!/bin/bash
# scratch: run hidden-shape robustness tests inside the HPC container
set -e
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/env.sh"
export PYTHONPATH="$ROOT:$ROOT/checker:${PYTHONPATH:-}"
if ! python3 -c "import custom_ops_lib" >/dev/null 2>&1; then bash checker/build.sh; fi
source "$ROOT/env.sh"
python3 -u scratch/hidden_test.py
