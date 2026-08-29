#!/bin/bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
evidence="$root/evidence/analysis_profile-20260819-083900"
cache="$root/twopuncture_cache"
timings="$evidence/run_wall_times.tsv"

printf 'run_id\tprofile\telapsed_s\n' > "$timings"

run_one() {
  local run_id="$1"
  local profile="$2"
  local build="$evidence/build-$profile"
  local work="$evidence/run-$run_id"
  local raw="$evidence/raw-$profile"
  local start_ns
  local end_ns
  local elapsed

  echo "==> $run_id profile=$profile"
  start_ns="$(date +%s%N)"
  (
    cd "$work"
    env AMSS_BUILD_DIR="$build" \
        AMSS_OUTPUT_ROOT="$work" \
        AMSS_CACHE_DIR="$cache" \
        AMSS_ANALYSIS_PROFILE_DIR="$raw" \
        AMSS_ANALYSIS_PROFILE_RUN_ID="$run_id" \
        OMP_NUM_THREADS=1 \
        ./run.sh --twop-cache
  ) > "$evidence/$run_id.log" 2>&1
  end_ns="$(date +%s%N)"
  elapsed="$(awk -v start="$start_ns" -v end="$end_ns" 'BEGIN {printf "%.6f", (end-start)/1000000000}')"
  printf '%s\t%s\t%s\n' "$run_id" "$profile" "$elapsed" >> "$timings"
  tail -n 8 "$evidence/$run_id.log"
}

run_one on1 on
run_one off1 off
run_one on2 on
run_one off2 off
run_one on3 on
run_one off3 off
