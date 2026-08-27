#!/bin/bash
set -euo pipefail

TRSM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$TRSM_DIR"
source ./env.sh

CC="${CC:-gcc}"
CFLAGS="${CFLAGS:--O3 -mcpu=native -fopenmp}"
ASMFLAGS="${ASMFLAGS:--march=armv8.2-a+sve}"
# The cluster image exposes a versioned KBLAS shared object without the
# unversioned libkblas.so linker name.  Override this when a development
# symlink or a different KBLAS release is provided by the target environment.
KBLAS_LIB="${KBLAS_LIB:--l:libkblas.so.25.2.1}"

# The contest image ships the versioned file, whose SONAME is libkblas.so.1,
# without a matching loader-visible symlink.  Keep the workaround local to
# this disposable candidate directory; the default V9 build is untouched.
if [[ ! -e "$TRSM_DIR/libkblas.so.1" && ! -L "$TRSM_DIR/libkblas.so.1" &&
      -f "$KBLAS_DIR/libkblas.so.25.2.1" ]]; then
    ln -s "$KBLAS_DIR/libkblas.so.25.2.1" "$TRSM_DIR/libkblas.so.1"
fi

build_one() {
    local name="$1"
    local case_macro="$2"
    local out="trsm_test_${name}"
    local objdir=".sme-build-${name}"
    mkdir -p "$objdir"

    # The official V9 build remains the separate command in README.md.  This
    # candidate object directory is intentionally disposable and never copies
    # over trsm_test or the submitted source files.
    "$CC" $CFLAGS -DTRSM_SME_PIPELINE "$case_macro" -c trsm.c \
        -o "$objdir/trsm.o"
    "$CC" $CFLAGS -c trsm_sme_candidate.c -o "$objdir/candidate.o"
    "$CC" $CFLAGS -c sme_update.c -o "$objdir/update.o"
    "$CC" $ASMFLAGS -c sme_update_16x32.S -o "$objdir/update_asm.o"
    "$CC" $CFLAGS bench_trsm.c \
        "$objdir/trsm.o" "$objdir/candidate.o" "$objdir/update.o" \
        "$objdir/update_asm.o" -o "$out" -lm $KBLAS_LIB
    echo "built $out with $case_macro"
}

build_one case3 -DTRSM_SME_CASE3
build_one case2 -DTRSM_SME_CASE2
