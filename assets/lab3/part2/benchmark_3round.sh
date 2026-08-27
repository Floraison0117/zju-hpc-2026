#!/bin/bash
SRC=/home/h3240101033/HPC101/src/lab3
VENV=/opt/lab3-venv/bin/python
ROUNDS=3
WARMUP=10
REPS=100
PREFIX=${HOME}/lab3/part2

for r in 1 2 3; do
    echo === Round ${r} ===
    OUTFILE=${PREFIX}/bench_v16gateexp_r${r}.csv
    ${VENV} ${SRC}/run.py --warmup ${WARMUP} --repetitions ${REPS} --output-format csv 2>&1 | tee ${OUTFILE} | grep -E '^(case|[a-z])' > ${PREFIX}/bench_v16gateexp_r${r}_clean.csv
done
echo Done
