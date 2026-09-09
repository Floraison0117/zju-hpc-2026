#!/bin/bash
# run.sh v7 — ZGEMM submission (Shenzhen SC Kunpeng 920, SVE, NUMA v4 + NC=192)
set -u
cd "$(dirname "$0")"

if command -v module >/dev/null 2>&1; then
    module use /home/HPC/HPCKit/latest/modulefiles 2>/dev/null || true
    module load gcc/compiler12.3.1/gccmodule 2>/dev/null || true
fi

ulimit -c 0
# HPCKit modules may export GOMP_CPU_AFFINITY, which silently overrides
# OMP_PLACES/OMP_PROC_BIND below. Make sure ours win.
unset GOMP_CPU_AFFINITY 2>/dev/null || true

# ---- source selection: SVE version when the CPU supports it.
# aarch64 lists ISA flags inside the "Features" line, so match the word
# anywhere in /proc/cpuinfo (-w avoids false hits such as "ssbs").
SRC=zgemm_nc192.c
grep -qw sve /proc/cpuinfo 2>/dev/null || echo "note: sve flag not in /proc/cpuinfo"
gcc -O3 bench_zgemm.c $SRC -o zgemm_test -lm -fopenmp 2> gcc_sve.err
if [ $? -ne 0 ]; then
    echo "note: SVE build failed, falling back to FCMLA source"
    SRC=zgemm.c
    gcc -O3 bench_zgemm.c $SRC -o zgemm_test -lm -fopenmp || { echo "BUILD FAIL"; exit 1; }
fi

# ---- SVE runtime smoke test (SIGILL on non-SVE hardware -> FCMLA fallback)
./zgemm_test 8 8 8 1 >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "note: SVE smoke test failed, falling back to FCMLA source"
    SRC=zgemm.c
    gcc -O3 bench_zgemm.c $SRC -o zgemm_test -lm -fopenmp || { echo "BUILD FAIL"; exit 1; }
fi
echo "using $SRC"

# ---- threads & NUMA pin ------------------------------------------------
# Two measured essentials on this many-NUMA box (see 优化日志 §3/§9):
#  1) OMP_PROC_BIND=close + OMP_PLACES=cores: unbound threads migrate across
#     NUMA nodes -> 35x slowdown.
#  2) taskset -c node0: keeps the WHOLE process (incl. the bench's first-touch
#     matrix init) on node0 so every page is local to the compute threads
#     (~2x). The batch system's default affinity may expose only a few
#     scattered CPUs (nproc=16 here) even on a whole-node job, while
#     taskset CAN still widen to node0's 38 CPUs -- so take the thread count
#     from the PINNED process, not from nproc.
NT=$(nproc 2>/dev/null || echo 38)
PIN=""
if command -v taskset >/dev/null 2>&1 && [ -r /sys/devices/system/node/node0/cpulist ]; then
    N0=$(cat /sys/devices/system/node/node0/cpulist)
    if taskset -c "$N0" true 2>/dev/null; then
        NT_PIN=$(taskset -c "$N0" nproc 2>/dev/null || echo 0)
        case "$NT_PIN" in
            ''|*[!0-9]*) NT_PIN=0 ;;
        esac
        if [ "$NT_PIN" -ge "$NT" ]; then
            PIN="taskset -c $N0"
            NT=$NT_PIN
        fi
    fi
fi
[ "$NT" -gt 38 ] && NT=38
[ "$NT" -lt 1 ] && NT=1
export OMP_NUM_THREADS=$NT
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "diag: host=$(hostname) nproc=$(nproc) NT=$NT pin=[$PIN] allowed=$(awk '/Cpus_allowed_list/{print $2}' /proc/self/status)" >&2

run_case () {
    if [ -n "$PIN" ]; then
        $PIN ./zgemm_test "$1" "$2" "$3" "$4"
    else
        ./zgemm_test "$1" "$2" "$3" "$4"
    fi
}

run_case 7427  7427 256 3
run_case 14848 14848 256 3
run_case 37360 8192 512 3

echo "run.sh DONE"
