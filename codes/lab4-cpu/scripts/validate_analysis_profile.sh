#!/bin/bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
evidence="$root/evidence/analysis_profile-20260819-083900"

echo "Build configuration"
grep -E '^(AMSS_ENABLE_ANALYSIS_PROFILE|AMSS_ENABLE_OPENMP|AMSS_OPT):' \
  "$evidence/build-off/CMakeCache.txt" "$evidence/build-on/CMakeCache.txt"

echo
echo "Profile file counts"
on_files="$(find "$evidence/raw-on" -type f -name 'analysis_profile_run-*_rank-*.tsv' | wc -l)"
off_files="$(find "$evidence/raw-off" -type f | wc -l)"
echo "ON TSV files: $on_files (expected 90)"
echo "OFF files: $off_files (expected 0)"
test "$on_files" -eq 90
test "$off_files" -eq 0

echo
echo "Numerical output byte comparison"
for index in 1 2 3; do
  on_dir="$evidence/run-on$index/GW250118/AMSS_NCKU_output/binary_output"
  off_dir="$evidence/run-off$index/GW250118/AMSS_NCKU_output/binary_output"
  compared=0
  while IFS= read -r relative; do
    case "$relative" in
      *.dat) cmp <(tail -n +2 "$on_dir/$relative") <(tail -n +2 "$off_dir/$relative") ;;
      *) cmp "$on_dir/$relative" "$off_dir/$relative" ;;
    esac
    compared=$((compared + 1))
  done < <(cd "$on_dir" && find . -type f \( -name 'Lev*.bin' -o -name 'bssn_*.dat' -o -name 'interp_constraint_*.dat' \) -printf '%P\n' | sort)
  echo "pair $index: PASS ($compared files; binaries byte-identical, text identical after timestamp header)"
done

echo
echo "Source hashes"
sha256sum "$root/CMakeLists.txt" "$root/src/analysis_profiler.h" \
  "$root/src/analysis_profiler.C" "$root/src/bssn_class.C" \
  "$root/src/surface_integral.C" "$root/scripts/analyze_analysis_profile.py"
