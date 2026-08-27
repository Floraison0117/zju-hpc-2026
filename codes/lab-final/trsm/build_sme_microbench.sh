#!/bin/bash
set -euo pipefail

TRSM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$TRSM_DIR"
source ./env.sh

CC="${CC:-gcc}"
CFLAGS="${CFLAGS:--O3 -mcpu=native -fopenmp}"
ASMFLAGS="${ASMFLAGS:--march=armv8.2-a+sve}"
OUT="${1:-sme_microbench_build}"
mkdir -p "$OUT"

"$CC" $CFLAGS -I. -c microbench/sme_update_test.c -o "$OUT/sme_update_test.o"
"$CC" $CFLAGS -I. -c microbench/sme_pipeline_bench.c -o "$OUT/sme_pipeline_bench.o"
"$CC" $CFLAGS -I. -c microbench/sme_inplace_bench.c -o "$OUT/sme_inplace_bench.o"
"$CC" $CFLAGS -I. -c microbench/sme_za_inplace_bench.c -o "$OUT/sme_za_inplace_bench.o"
"$CC" $CFLAGS -c sme_update.c -o "$OUT/sme_update.o"
"$CC" $ASMFLAGS -c sme_update_16x32.S -o "$OUT/sme_update_16x32.o"
"$CC" $CFLAGS "$OUT/sme_update_test.o" "$OUT/sme_update.o" \
    "$OUT/sme_update_16x32.o" -o "$OUT/sme_update_test"
"$CC" $CFLAGS "$OUT/sme_pipeline_bench.o" "$OUT/sme_update.o" \
    "$OUT/sme_update_16x32.o" -o "$OUT/sme_pipeline_bench"
"$CC" $CFLAGS "$OUT/sme_inplace_bench.o" "$OUT/sme_update.o" \
    "$OUT/sme_update_16x32.o" -o "$OUT/sme_inplace_bench"
"$CC" $CFLAGS "$OUT/sme_za_inplace_bench.o" "$OUT/sme_update.o" \
    "$OUT/sme_update_16x32.o" -o "$OUT/sme_za_inplace_bench"
echo "built $OUT/sme_update_test"
echo "built $OUT/sme_pipeline_bench"
echo "built $OUT/sme_inplace_bench"
echo "built $OUT/sme_za_inplace_bench"
