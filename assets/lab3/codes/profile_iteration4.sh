#!/usr/bin/env bash
set -euo pipefail

cp student/tilelang_fwd.py.iter3_final student/tilelang_fwd.py
ncu \
  --clock-control none \
  --force-overwrite \
  --kernel-name regex:gdn_persistent_pingpong \
  --launch-count 1 \
  --section InstructionStats \
  --section Occupancy \
  --section SchedulerStats \
  --section WarpStateStats \
  --section SourceCounters \
  --csv \
  --log-file iteration3_scalar_ncu.csv \
  /opt/lab3-venv/bin/python run.py \
    --case chain_equal \
    --warmup 0 \
    --repetitions 1

cp student/tilelang_fwd.py.iter4_work student/tilelang_fwd.py
ncu \
  --clock-control none \
  --force-overwrite \
  --kernel-name regex:gdn_persistent_tensorcore \
  --launch-count 1 \
  --section InstructionStats \
  --section Occupancy \
  --section SchedulerStats \
  --section WarpStateStats \
  --section SourceCounters \
  --csv \
  --log-file iteration4_tensorcore_ncu.csv \
  /opt/lab3-venv/bin/python run.py \
    --case chain_equal \
    --warmup 0 \
    --repetitions 1
