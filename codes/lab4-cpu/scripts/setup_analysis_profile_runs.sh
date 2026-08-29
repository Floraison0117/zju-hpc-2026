#!/bin/bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
evidence="$root/evidence/analysis_profile-20260819-083900"

rmdir "$evidence/run-" "$evidence/run-;" 2>/dev/null || true
mkdir -p "$evidence/raw-on" "$evidence/raw-off"

for name in on1 on2 on3 off1 off2 off3; do
  work="$evidence/run-$name"
  mkdir -p "$work"
  cp "$root/AMSS_NCKU_Input.py" "$root/AMSS_NCKU_Program.py" "$root/run.sh" "$work/"
  cp -a "$root/scripts" "$work/"
  sed -i 's/Final_Evolution_Time     = 100.0 if GPU_Calculation == "yes" else 40.0/Final_Evolution_Time     = 100.0 if GPU_Calculation == "yes" else 1.0/' "$work/AMSS_NCKU_Input.py"
  sed -i 's/Analysis_Time            = 1000.0/Analysis_Time            = 0.1/' "$work/AMSS_NCKU_Input.py"
done

grep -nE 'MPI_processes|OMP_threads|Final_Evolution_Time|Analysis_Time' "$evidence/run-on1/AMSS_NCKU_Input.py"
