#!/usr/bin/env bash
# iter17 P8 Level-2 — full 100-step run + check.sh FINAL PASS (authoritative gate)
set -uo pipefail
CAND="$1"
EV="$CAND/evidence/p8-l2"
mkdir -p "$EV"
mkdir -p "$EV/out/GW250118"   # program does single-level os.mkdir; parent must pre-exist
exec > "$EV/job-l2.log" 2>&1
echo "=== START P8 Level-2 $(date -u) ==="
nvidia-smi -L 2>/dev/null | head -2
cd "$CAND"
echo "=== binary ==="
ls -lh build/ABEGPU
echo "=== FULL 100-step run ==="
AMSS_BUILD_DIR="$CAND/build" AMSS_OUTPUT_ROOT="$EV/out" AMSS_CACHE_DIR=/home/h3240101033/lab4-gpu/twopuncture_cache \
  ./run.sh --twop-cache > "$EV/full_run.log" 2>&1
echo "=== Program Cost ==="
grep -iE 'Program Cost|Total Evolve|After Step: 100' "$EV/full_run.log" | tail -3
echo "=== check.sh (RESULT vs golden) ==="
RESULT_DIR="$EV/out/GW250118/AMSS_NCKU_output"
GOLDEN_DIR="$CAND/golden"
./check.sh "$RESULT_DIR" "$GOLDEN_DIR" > "$EV/check_result.txt" 2>&1
cat "$EV/check_result.txt"
echo "LVL2_DONE"
