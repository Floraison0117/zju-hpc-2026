        .text
        .align 2

/*
 * The assembler shipped with the contest image predates SME mnemonics, so
 * SME instructions remain encoded with .inst.  The target has FEAT_SME and
 * FEAT_SME_F64F64.  The C merge uses the supported ZA-to-Z MOVA form, then
 * subtracts the positive FMOPA result from C with ordinary SVE instructions.
 *
 * For SVL=512, one .D tile slice is 8 doubles (64 bytes).  The tile/slice
 * order below is the same order used by the old MOVA extraction sequence.
 * The macro arguments t0..t3 are ZA tile numbers and rs is the 2-bit field
 * selecting W12..W15 (0..3).  x16 is the address scratch register.
 */
.macro ZA_LD4 base, t0, t1, t2, t3, rs, off
        mov     x16, \base
        .inst   (0xe0df8000 + (16 << 5) + (\rs << 13) + (\t0 << 1) + \off)
        add     x16, x16, #64
        .inst   (0xe0df8000 + (16 << 5) + (\rs << 13) + (\t1 << 1) + \off)
        add     x16, x16, #64
        .inst   (0xe0df8000 + (16 << 5) + (\rs << 13) + (\t2 << 1) + \off)
        add     x16, x16, #64
        .inst   (0xe0df8000 + (16 << 5) + (\rs << 13) + (\t3 << 1) + \off)
.endm

.macro ZA_ST4 base, t0, t1, t2, t3, rs, off
        mov     x16, \base
        .inst   (0xe0ff8000 + (16 << 5) + (\rs << 13) + (\t0 << 1) + \off)
        add     x16, x16, #64
        .inst   (0xe0ff8000 + (16 << 5) + (\rs << 13) + (\t1 << 1) + \off)
        add     x16, x16, #64
        .inst   (0xe0ff8000 + (16 << 5) + (\rs << 13) + (\t2 << 1) + \off)
        add     x16, x16, #64
        .inst   (0xe0ff8000 + (16 << 5) + (\rs << 13) + (\t3 << 1) + \off)
.endm

.macro ZA_LOAD_C_TILE base
        mov     x5, \base
        add     x6, \base, x4
        add     x7, x6, x4
        add     x8, x7, x4
        ZA_LD4 x5, 0, 2, 4, 6, 0, 0
        ZA_LD4 x6, 0, 2, 4, 6, 0, 1
        ZA_LD4 x7, 0, 2, 4, 6, 1, 0
        ZA_LD4 x8, 0, 2, 4, 6, 1, 1
        add     x5, x5, x4, lsl #2
        add     x6, x6, x4, lsl #2
        add     x7, x7, x4, lsl #2
        add     x8, x8, x4, lsl #2
        ZA_LD4 x5, 0, 2, 4, 6, 2, 0
        ZA_LD4 x6, 0, 2, 4, 6, 2, 1
        ZA_LD4 x7, 0, 2, 4, 6, 3, 0
        ZA_LD4 x8, 0, 2, 4, 6, 3, 1
        add     x5, \base, x4, lsl #3
        add     x6, x5, x4
        add     x7, x6, x4
        add     x8, x7, x4
        ZA_LD4 x5, 1, 3, 5, 7, 0, 0
        ZA_LD4 x6, 1, 3, 5, 7, 0, 1
        ZA_LD4 x7, 1, 3, 5, 7, 1, 0
        ZA_LD4 x8, 1, 3, 5, 7, 1, 1
        add     x5, x5, x4, lsl #2
        add     x6, x6, x4, lsl #2
        add     x7, x7, x4, lsl #2
        add     x8, x8, x4, lsl #2
        ZA_LD4 x5, 1, 3, 5, 7, 2, 0
        ZA_LD4 x6, 1, 3, 5, 7, 2, 1
        ZA_LD4 x7, 1, 3, 5, 7, 3, 0
        ZA_LD4 x8, 1, 3, 5, 7, 3, 1
.endm

.macro ZA_STORE_C_TILE base
        mov     x5, \base
        add     x6, \base, x4
        add     x7, x6, x4
        add     x8, x7, x4
        ZA_ST4 x5, 0, 2, 4, 6, 0, 0
        ZA_ST4 x6, 0, 2, 4, 6, 0, 1
        ZA_ST4 x7, 0, 2, 4, 6, 1, 0
        ZA_ST4 x8, 0, 2, 4, 6, 1, 1
        add     x5, x5, x4, lsl #2
        add     x6, x6, x4, lsl #2
        add     x7, x7, x4, lsl #2
        add     x8, x8, x4, lsl #2
        ZA_ST4 x5, 0, 2, 4, 6, 2, 0
        ZA_ST4 x6, 0, 2, 4, 6, 2, 1
        ZA_ST4 x7, 0, 2, 4, 6, 3, 0
        ZA_ST4 x8, 0, 2, 4, 6, 3, 1
        add     x5, \base, x4, lsl #3
        add     x6, x5, x4
        add     x7, x6, x4
        add     x8, x7, x4
        ZA_ST4 x5, 1, 3, 5, 7, 0, 0
        ZA_ST4 x6, 1, 3, 5, 7, 0, 1
        ZA_ST4 x7, 1, 3, 5, 7, 1, 0
        ZA_ST4 x8, 1, 3, 5, 7, 1, 1
        add     x5, x5, x4, lsl #2
        add     x6, x6, x4, lsl #2
        add     x7, x7, x4, lsl #2
        add     x8, x8, x4, lsl #2
        ZA_ST4 x5, 1, 3, 5, 7, 2, 0
        ZA_ST4 x6, 1, 3, 5, 7, 2, 1
        ZA_ST4 x7, 1, 3, 5, 7, 3, 0
        ZA_ST4 x8, 1, 3, 5, 7, 3, 1
.endm

/*
 * Extract one vertical ZA.D slice to z0 and merge it into C with the
 * ordinary SVE load/subtract/store sequence.  The target accepts the SME
 * tile-to-vector MOVA form; V slices match the direct ZA slice layout used
 * by the original C-tile load/store path.
 * For the 64-bit encoding, ZAn is tile*0x40, the slice offset is bit 5,
 * and Rs (W12..W15) is bits 15:13.  p1 is full and is part of the base.
 */
.macro MOVA_SUB4 base, t0, t1, t2, t3, rs, off
        mov     x16, \base
        ldr     z24, [x16]
        ldr     z25, [x16, #1, mul vl]
        ldr     z26, [x16, #2, mul vl]
        ldr     z27, [x16, #3, mul vl]
        .inst   (0xc0c28400 + (\rs << 13) + (\off << 5) + (\t0 << 6) + 0)
        .inst   (0xc0c28400 + (\rs << 13) + (\off << 5) + (\t1 << 6) + 1)
        .inst   (0xc0c28400 + (\rs << 13) + (\off << 5) + (\t2 << 6) + 2)
        .inst   (0xc0c28400 + (\rs << 13) + (\off << 5) + (\t3 << 6) + 3)
        fsub    z24.d, p0/m, z24.d, z0.d
        fsub    z25.d, p0/m, z25.d, z1.d
        fsub    z26.d, p0/m, z26.d, z2.d
        fsub    z27.d, p0/m, z27.d, z3.d
        str     z24, [x16]
        str     z25, [x16, #1, mul vl]
        str     z26, [x16, #2, mul vl]
        str     z27, [x16, #3, mul vl]
.endm

.macro MOVA_SUB_C_TILE base
        mov     x5, \base
        add     x6, \base, x4
        add     x7, x6, x4
        add     x8, x7, x4
        MOVA_SUB4 x5, 0, 2, 4, 6, 0, 0
        MOVA_SUB4 x6, 0, 2, 4, 6, 0, 1
        MOVA_SUB4 x7, 0, 2, 4, 6, 1, 0
        MOVA_SUB4 x8, 0, 2, 4, 6, 1, 1
        add     x5, x5, x4, lsl #2
        add     x6, x6, x4, lsl #2
        add     x7, x7, x4, lsl #2
        add     x8, x8, x4, lsl #2
        MOVA_SUB4 x5, 0, 2, 4, 6, 2, 0
        MOVA_SUB4 x6, 0, 2, 4, 6, 2, 1
        MOVA_SUB4 x7, 0, 2, 4, 6, 3, 0
        MOVA_SUB4 x8, 0, 2, 4, 6, 3, 1
        add     x5, \base, x4, lsl #3
        add     x6, x5, x4
        add     x7, x6, x4
        add     x8, x7, x4
        MOVA_SUB4 x5, 1, 3, 5, 7, 0, 0
        MOVA_SUB4 x6, 1, 3, 5, 7, 0, 1
        MOVA_SUB4 x7, 1, 3, 5, 7, 1, 0
        MOVA_SUB4 x8, 1, 3, 5, 7, 1, 1
        add     x5, x5, x4, lsl #2
        add     x6, x6, x4, lsl #2
        add     x7, x7, x4, lsl #2
        add     x8, x8, x4, lsl #2
        MOVA_SUB4 x5, 1, 3, 5, 7, 2, 0
        MOVA_SUB4 x6, 1, 3, 5, 7, 2, 1
        MOVA_SUB4 x7, 1, 3, 5, 7, 3, 0
        MOVA_SUB4 x8, 1, 3, 5, 7, 3, 1
.endm

/* Load an existing C tile into ZA through the inverse MOVA direction. */
.macro MOVA_LOAD4 base, t0, t1, t2, t3, rs, off
        mov     x16, \base
        ldr     z0, [x16]
        ldr     z1, [x16, #1, mul vl]
        ldr     z2, [x16, #2, mul vl]
        ldr     z3, [x16, #3, mul vl]
        /* Vector-to-tile fields are Zn[9:5], ZAd[3:1], and o1[0]. */
        .inst   (0xc0c08400 + (\rs << 13) + (0 << 5) + (\t0 << 1) + \off)
        .inst   (0xc0c08400 + (\rs << 13) + (1 << 5) + (\t1 << 1) + \off)
        .inst   (0xc0c08400 + (\rs << 13) + (2 << 5) + (\t2 << 1) + \off)
        .inst   (0xc0c08400 + (\rs << 13) + (3 << 5) + (\t3 << 1) + \off)
.endm

.macro MOVA_LOAD_C_TILE base
        mov     x5, \base
        add     x6, \base, x4
        add     x7, x6, x4
        add     x8, x7, x4
        MOVA_LOAD4 x5, 0, 2, 4, 6, 0, 0
        MOVA_LOAD4 x6, 0, 2, 4, 6, 0, 1
        MOVA_LOAD4 x7, 0, 2, 4, 6, 1, 0
        MOVA_LOAD4 x8, 0, 2, 4, 6, 1, 1
        add     x5, x5, x4, lsl #2
        add     x6, x6, x4, lsl #2
        add     x7, x7, x4, lsl #2
        add     x8, x8, x4, lsl #2
        MOVA_LOAD4 x5, 0, 2, 4, 6, 2, 0
        MOVA_LOAD4 x6, 0, 2, 4, 6, 2, 1
        MOVA_LOAD4 x7, 0, 2, 4, 6, 3, 0
        MOVA_LOAD4 x8, 0, 2, 4, 6, 3, 1
        add     x5, \base, x4, lsl #3
        add     x6, x5, x4
        add     x7, x6, x4
        add     x8, x7, x4
        MOVA_LOAD4 x5, 1, 3, 5, 7, 0, 0
        MOVA_LOAD4 x6, 1, 3, 5, 7, 0, 1
        MOVA_LOAD4 x7, 1, 3, 5, 7, 1, 0
        MOVA_LOAD4 x8, 1, 3, 5, 7, 1, 1
        add     x5, x5, x4, lsl #2
        add     x6, x6, x4, lsl #2
        add     x7, x7, x4, lsl #2
        add     x8, x8, x4, lsl #2
        MOVA_LOAD4 x5, 1, 3, 5, 7, 2, 0
        MOVA_LOAD4 x6, 1, 3, 5, 7, 2, 1
        MOVA_LOAD4 x7, 1, 3, 5, 7, 3, 0
        MOVA_LOAD4 x8, 1, 3, 5, 7, 3, 1
.endm

        .global sme_update_16x32
        .type sme_update_16x32, %function
/* C[16,32] -= A_pack[K,16] * B_pack[K,32]. */
sme_update_16x32:
        sub     sp, sp, #80
        stp     d8, d9, [sp, #0]
        stp     d10, d11, [sp, #16]
        stp     d12, d13, [sp, #32]
        stp     d14, d15, [sp, #48]
        stp     d16, d17, [sp, #64]
        .inst 0xd503477f            // smstart
        .inst 0xd5033fdf            // isb
        .inst 0x2558e3e0            // ptrue p0.h
        .inst 0x2558e3e1            // ptrue p1.h
        .inst 0x2558e3e2            // ptrue p2.h
        .inst 0x2558e3e3            // ptrue p3.h
        .inst 0xc00800ff            // zero {za}
        mov     w12, #0
        mov     w13, #2
        mov     w14, #4
        mov     w15, #6

        /* ZA is zeroed; MOVA_SUB_C_TILE merges AB with the old C tile. */
1:
        ld1d    {z16.d}, p0/z, [x0]
        ld1d    {z17.d}, p0/z, [x0, #1, mul vl]
        ld1d    {z0.d}, p1/z, [x1]
        ld1d    {z1.d}, p1/z, [x1, #1, mul vl]
        ld1d    {z2.d}, p1/z, [x1, #2, mul vl]
        ld1d    {z3.d}, p2/z, [x1, #3, mul vl]
        .inst 0x80d00400            // fmopa za0.d, p1/m, p0/m, z0.d, z16.d
        .inst 0x80d00422
        .inst 0x80d00444
        .inst 0x80d00866
        .inst 0x80d16401
        .inst 0x80d16423
        .inst 0x80d16445
        .inst 0x80d16867
        add     x0, x0, #128
        add     x1, x1, #256
        subs    x3, x3, #1
        b.ne    1b

        MOVA_SUB_C_TILE x2
        .inst 0xd503467f            // smstop
        ldp     d8, d9, [sp, #0]
        ldp     d10, d11, [sp, #16]
        ldp     d12, d13, [sp, #32]
        ldp     d14, d15, [sp, #48]
        ldp     d16, d17, [sp, #64]
        add     sp, sp, #80
        ret
        .size sme_update_16x32, .-sme_update_16x32

        .global sme_update_16x32_strided_b
        .type sme_update_16x32_strided_b, %function
/* C[16,32] -= A_pack[K,16] * B[K,32], with a byte stride between B rows. */
sme_update_16x32_strided_b:
        sub     sp, sp, #80
        stp     d8, d9, [sp, #0]
        stp     d10, d11, [sp, #16]
        stp     d12, d13, [sp, #32]
        stp     d14, d15, [sp, #48]
        stp     d16, d17, [sp, #64]
        mov     x11, x3             // B row stride in bytes
        mov     x3, x2              // K loop counter
        mov     x2, x4              // C tile
        mov     x4, x5              // C row stride in bytes
        .inst 0xd503477f            // smstart
        .inst 0xd5033fdf            // isb
        .inst 0x2558e3e0            // ptrue p0.h
        .inst 0x2558e3e1            // ptrue p1.h
        .inst 0x2558e3e2            // ptrue p2.h
        .inst 0x2558e3e3            // ptrue p3.h
        .inst 0xc00800ff            // zero {za}
        mov     w12, #0
        mov     w13, #2
        mov     w14, #4
        mov     w15, #6
1:
        ld1d    {z16.d}, p0/z, [x0]
        ld1d    {z17.d}, p0/z, [x0, #1, mul vl]
        ld1d    {z0.d}, p1/z, [x1]
        ld1d    {z1.d}, p1/z, [x1, #1, mul vl]
        ld1d    {z2.d}, p1/z, [x1, #2, mul vl]
        ld1d    {z3.d}, p2/z, [x1, #3, mul vl]
        .inst 0x80d00400
        .inst 0x80d00422
        .inst 0x80d00444
        .inst 0x80d00866
        .inst 0x80d16401
        .inst 0x80d16423
        .inst 0x80d16445
        .inst 0x80d16867
        add     x0, x0, #128
        add     x1, x1, x11
        subs    x3, x3, #1
        b.ne    1b
        MOVA_SUB_C_TILE x2
        .inst 0xd503467f            // smstop
        ldp     d8, d9, [sp, #0]
        ldp     d10, d11, [sp, #16]
        ldp     d12, d13, [sp, #32]
        ldp     d14, d15, [sp, #48]
        ldp     d16, d17, [sp, #64]
        add     sp, sp, #80
        ret
        .size sme_update_16x32_strided_b, .-sme_update_16x32_strided_b

        .global sme_update_16x32_strided_b_grid
        .type sme_update_16x32_strided_b_grid, %function
/* Persistent raw-B grid: one SME region covers nt columns and mt rows. */
sme_update_16x32_strided_b_grid:
        sub     sp, sp, #112
        stp     d8, d9, [sp, #0]
        stp     d10, d11, [sp, #16]
        stp     d12, d13, [sp, #32]
        stp     d14, d15, [sp, #48]
        stp     d16, d17, [sp, #64]
        stp     x18, x19, [sp, #80]
        stp     x20, x21, [sp, #96]
        mov     x18, x3             // B row stride in bytes
        mov     x19, x5             // C row stride in bytes
        mov     x20, x1             // B base, reused for every row tile
        mov     x21, x6             // number of B column tiles
        mov     x10, x7             // remaining A/C row tiles
        mov     x11, x2             // K, preserved across all tiles
        mov     x17, x4             // current C row-tile base
        mov     x4, x19             // MOVA_SUB_C_TILE consumes x4
        .inst 0xd503477f            // smstart
        .inst 0xd5033fdf            // isb
        .inst 0x2558e3e0            // ptrue p0.h
        .inst 0x2558e3e1            // ptrue p1.h
        .inst 0x2558e3e2            // ptrue p2.h
        .inst 0x2558e3e3            // ptrue p3.h
        mov     w12, #0
        mov     w13, #2
        mov     w14, #4
        mov     w15, #6
1:
        mov     x9, x21             // reset column count
        mov     x1, x20             // first B column for this row tile
        mov     x2, x17             // first C tile for this row tile
2:
        mov     x3, x11
        .inst 0xc00800ff            // zero {za}
3:
        ld1d    {z16.d}, p0/z, [x0]
        ld1d    {z17.d}, p0/z, [x0, #1, mul vl]
        ld1d    {z0.d}, p1/z, [x1]
        ld1d    {z1.d}, p1/z, [x1, #1, mul vl]
        ld1d    {z2.d}, p1/z, [x1, #2, mul vl]
        ld1d    {z3.d}, p2/z, [x1, #3, mul vl]
        .inst 0x80d00400
        .inst 0x80d00422
        .inst 0x80d00444
        .inst 0x80d00866
        .inst 0x80d16401
        .inst 0x80d16423
        .inst 0x80d16445
        .inst 0x80d16867
        add     x0, x0, #128
        add     x1, x1, x18
        subs    x3, x3, #1
        b.ne    3b
        sub     x0, x0, x11, lsl #7
        mul     x16, x11, x18
        sub     x1, x1, x16
        MOVA_SUB_C_TILE x2
        add     x1, x1, #256
        add     x2, x2, #256
        subs    x9, x9, #1
        b.ne    2b
        add     x0, x0, x11, lsl #7
        add     x17, x17, x19, lsl #4
        subs    x10, x10, #1
        b.ne    1b
        .inst 0xd503467f            // smstop
        ldp     d8, d9, [sp, #0]
        ldp     d10, d11, [sp, #16]
        ldp     d12, d13, [sp, #32]
        ldp     d14, d15, [sp, #48]
        ldp     d16, d17, [sp, #64]
        ldp     x18, x19, [sp, #80]
        ldp     x20, x21, [sp, #96]
        add     sp, sp, #112
        ret
        .size sme_update_16x32_strided_b_grid, .-sme_update_16x32_strided_b_grid

        .global sme_update_16x32_ptr_batch
        .type sme_update_16x32_ptr_batch, %function
/* Persistent task-list form: one SME region processes count independent tiles. */
sme_update_16x32_ptr_batch:
        sub     sp, sp, #96
        stp     d8, d9, [sp, #0]
        stp     d10, d11, [sp, #16]
        stp     d12, d13, [sp, #32]
        stp     d14, d15, [sp, #48]
        stp     d16, d17, [sp, #64]
        stp     x18, x19, [sp, #80]
        mov     x10, x0             // A tile pointer list
        mov     x11, x1             // B tile pointer list
        mov     x17, x2             // C tile pointer list
        mov     x18, x3             // K
        mov     x19, x4             // C row stride in bytes
        mov     x9, x5              // remaining tile tasks
        mov     x4, x19             // MOVA_SUB_C_TILE consumes x4
        .inst 0xd503477f            // smstart
        .inst 0xd5033fdf            // isb
        .inst 0x2558e3e0            // ptrue p0.h
        .inst 0x2558e3e1            // ptrue p1.h
        .inst 0x2558e3e2            // ptrue p2.h
        .inst 0x2558e3e3            // ptrue p3.h
        mov     w12, #0
        mov     w13, #2
        mov     w14, #4
        mov     w15, #6
1:
        ldr     x0, [x10], #8
        ldr     x1, [x11], #8
        ldr     x2, [x17], #8
        mov     x3, x18
        .inst 0xc00800ff            // zero {za}
2:
        ld1d    {z16.d}, p0/z, [x0]
        ld1d    {z17.d}, p0/z, [x0, #1, mul vl]
        ld1d    {z0.d}, p1/z, [x1]
        ld1d    {z1.d}, p1/z, [x1, #1, mul vl]
        ld1d    {z2.d}, p1/z, [x1, #2, mul vl]
        ld1d    {z3.d}, p2/z, [x1, #3, mul vl]
        .inst 0x80d00400
        .inst 0x80d00422
        .inst 0x80d00444
        .inst 0x80d00866
        .inst 0x80d16401
        .inst 0x80d16423
        .inst 0x80d16445
        .inst 0x80d16867
        add     x0, x0, #128
        add     x1, x1, #256
        subs    x3, x3, #1
        b.ne    2b
        MOVA_SUB_C_TILE x2
        subs    x9, x9, #1
        b.ne    1b
        .inst 0xd503467f            // smstop
        ldp     d8, d9, [sp, #0]
        ldp     d10, d11, [sp, #16]
        ldp     d12, d13, [sp, #32]
        ldp     d14, d15, [sp, #48]
        ldp     d16, d17, [sp, #64]
        ldp     x18, x19, [sp, #80]
        add     sp, sp, #96
        ret
        .size sme_update_16x32_ptr_batch, .-sme_update_16x32_ptr_batch

        .global sme_update_16x32_za_init
        .type sme_update_16x32_za_init, %function
/* Experimental in-place path: C is moved into ZA, then FMOPS subtracts AB. */
sme_update_16x32_za_init:
        sub     sp, sp, #80
        stp     d8, d9, [sp, #0]
        stp     d10, d11, [sp, #16]
        stp     d12, d13, [sp, #32]
        stp     d14, d15, [sp, #48]
        stp     d16, d17, [sp, #64]
        .inst 0xd503477f            // smstart
        .inst 0xd5033fdf            // isb
        .inst 0x2558e3e0            // ptrue p0.h
        .inst 0x2558e3e1            // ptrue p1.h
        .inst 0x2558e3e2            // ptrue p2.h
        .inst 0x2558e3e3            // ptrue p3.h
        mov     w12, #0
        mov     w13, #2
        mov     w14, #4
        mov     w15, #6

        MOVA_LOAD_C_TILE x2
1:
        ld1d    {z16.d}, p0/z, [x0]
        ld1d    {z17.d}, p0/z, [x0, #1, mul vl]
        ld1d    {z0.d}, p1/z, [x1]
        ld1d    {z1.d}, p1/z, [x1, #1, mul vl]
        ld1d    {z2.d}, p1/z, [x1, #2, mul vl]
        ld1d    {z3.d}, p2/z, [x1, #3, mul vl]
        .inst 0x80d00410
        .inst 0x80d00432
        .inst 0x80d00454
        .inst 0x80d00876
        .inst 0x80d16411
        .inst 0x80d16433
        .inst 0x80d16455
        .inst 0x80d16877
        add     x0, x0, #128
        add     x1, x1, #256
        subs    x3, x3, #1
        b.ne    1b

        ZA_STORE_C_TILE x2
        .inst 0xd503467f            // smstop
        ldp     d8, d9, [sp, #0]
        ldp     d10, d11, [sp, #16]
        ldp     d12, d13, [sp, #32]
        ldp     d14, d15, [sp, #48]
        ldp     d16, d17, [sp, #64]
        add     sp, sp, #80
        ret
        .size sme_update_16x32_za_init, .-sme_update_16x32_za_init

        .global sme_update_16x32_za_ldst
        .type sme_update_16x32_za_ldst, %function
/* Experimental direct ZA load/store form of the in-place update. */
sme_update_16x32_za_ldst:
        sub     sp, sp, #80
        stp     d8, d9, [sp, #0]
        stp     d10, d11, [sp, #16]
        stp     d12, d13, [sp, #32]
        stp     d14, d15, [sp, #48]
        stp     d16, d17, [sp, #64]
        .inst 0xd503477f            // smstart
        .inst 0xd5033fdf            // isb
        .inst 0x2558e3e0            // ptrue p0.h
        .inst 0x2558e3e1            // ptrue p1.h
        .inst 0x2558e3e2            // ptrue p2.h
        .inst 0x2558e3e3            // ptrue p3.h
        mov     w12, #0
        mov     w13, #2
        mov     w14, #4
        mov     w15, #6
        ZA_LOAD_C_TILE x2
1:
        ld1d    {z16.d}, p0/z, [x0]
        ld1d    {z17.d}, p0/z, [x0, #1, mul vl]
        ld1d    {z0.d}, p1/z, [x1]
        ld1d    {z1.d}, p1/z, [x1, #1, mul vl]
        ld1d    {z2.d}, p1/z, [x1, #2, mul vl]
        ld1d    {z3.d}, p2/z, [x1, #3, mul vl]
        .inst 0x80d00410
        .inst 0x80d00432
        .inst 0x80d00454
        .inst 0x80d00876
        .inst 0x80d16411
        .inst 0x80d16433
        .inst 0x80d16455
        .inst 0x80d16877
        add     x0, x0, #128
        add     x1, x1, #256
        subs    x3, x3, #1
        b.ne    1b
        ZA_STORE_C_TILE x2
        .inst 0xd503467f            // smstop
        ldp     d8, d9, [sp, #0]
        ldp     d10, d11, [sp, #16]
        ldp     d12, d13, [sp, #32]
        ldp     d14, d15, [sp, #48]
        ldp     d16, d17, [sp, #64]
        add     sp, sp, #80
        ret
        .size sme_update_16x32_za_ldst, .-sme_update_16x32_za_ldst

        .global sme_update_16x32_za_init_grid_batch
        .type sme_update_16x32_za_init_grid_batch, %function
/* Persistent grid form of sme_update_16x32_za_init. */
sme_update_16x32_za_init_grid_batch:
        sub     sp, sp, #80
        stp     d8, d9, [sp, #0]
        stp     d10, d11, [sp, #16]
        stp     d12, d13, [sp, #32]
        stp     d14, d15, [sp, #48]
        stp     d16, d17, [sp, #64]
        mov     x9, x5              // number of B/C column tiles
        mov     x10, x6             // remaining A/C row tiles
        mov     x11, x3             // K, preserved across all tiles
        mov     x17, x2             // current C row-tile base
        mul     x6, x9, x11
        add     x2, x1, x6, lsl #8  // one-past-last packed B tile
        .inst 0xd503477f            // smstart
        .inst 0xd5033fdf            // isb
        .inst 0x2558e3e0            // ptrue p0.h
        .inst 0x2558e3e1            // ptrue p1.h
        .inst 0x2558e3e2            // ptrue p2.h
        .inst 0x2558e3e3            // ptrue p3.h
        mov     w12, #0
        mov     w13, #2
        mov     w14, #4
        mov     w15, #6
1:
3:
        mov     x3, x11
        /* Direct ZA load avoids the vector-to-tile MOVA staging path. */
        ZA_LOAD_C_TILE x17
4:
        ld1d    {z16.d}, p0/z, [x0]
        ld1d    {z17.d}, p0/z, [x0, #1, mul vl]
        ld1d    {z0.d}, p1/z, [x1]
        ld1d    {z1.d}, p1/z, [x1, #1, mul vl]
        ld1d    {z2.d}, p1/z, [x1, #2, mul vl]
        ld1d    {z3.d}, p2/z, [x1, #3, mul vl]
        .inst 0x80d00410
        .inst 0x80d00432
        .inst 0x80d00454
        .inst 0x80d00876
        .inst 0x80d16411
        .inst 0x80d16433
        .inst 0x80d16455
        .inst 0x80d16877
        add     x0, x0, #128
        add     x1, x1, #256
        subs    x3, x3, #1
        b.ne    4b
        ZA_STORE_C_TILE x17
        add     x17, x17, #256
        sub     x0, x0, x11, lsl #7
        cmp     x1, x2
        b.ne    3b
        mul     x6, x9, x11
        sub     x1, x2, x6, lsl #8
        sub     x17, x17, x9, lsl #8
        add     x17, x17, x4, lsl #4
        add     x0, x0, x11, lsl #7
        subs    x10, x10, #1
        b.ne    1b
        .inst 0xd503467f            // smstop
        ldp     d8, d9, [sp, #0]
        ldp     d10, d11, [sp, #16]
        ldp     d12, d13, [sp, #32]
        ldp     d14, d15, [sp, #48]
        ldp     d16, d17, [sp, #64]
        add     sp, sp, #80
        ret
        .size sme_update_16x32_za_init_grid_batch, .-sme_update_16x32_za_init_grid_batch

        .global sme_update_16x32_batch
        .type sme_update_16x32_batch, %function
/*
 * Apply one A tile to nt adjacent B/C tiles.  x5 is nt.  Keeping this loop
 * inside one SME region is the persistent-worker primitive used by Case3.
 */
sme_update_16x32_batch:
        sub     sp, sp, #80
        stp     d8, d9, [sp, #0]
        stp     d10, d11, [sp, #16]
        stp     d12, d13, [sp, #32]
        stp     d14, d15, [sp, #48]
        stp     d16, d17, [sp, #64]
        mov     x9, x5              // remaining B/C tiles
        mov     x10, x3             // K, preserved across tiles
        mov     x17, x2             // current C tile
        mov     x2, x0              // A tile base
        .inst 0xd503477f            // smstart
        .inst 0xd5033fdf            // isb
        .inst 0x2558e3e0            // ptrue p0.h
        .inst 0x2558e3e1            // ptrue p1.h
        .inst 0x2558e3e2            // ptrue p2.h
        .inst 0x2558e3e3            // ptrue p3.h
        mov     w12, #0
        mov     w13, #2
        mov     w14, #4
        mov     w15, #6
1:
        mov     x11, x2
        mov     x0, x1
        mov     x3, x10
        .inst 0xc00800ff            // zero {za}
2:
        ld1d    {z16.d}, p0/z, [x11]
        ld1d    {z17.d}, p0/z, [x11, #1, mul vl]
        ld1d    {z0.d}, p1/z, [x0]
        ld1d    {z1.d}, p1/z, [x0, #1, mul vl]
        ld1d    {z2.d}, p1/z, [x0, #2, mul vl]
        ld1d    {z3.d}, p2/z, [x0, #3, mul vl]
        .inst 0x80d00400
        .inst 0x80d00422
        .inst 0x80d00444
        .inst 0x80d00866
        .inst 0x80d16401
        .inst 0x80d16423
        .inst 0x80d16445
        .inst 0x80d16867
        add     x11, x11, #128
        add     x0, x0, #256
        subs    x3, x3, #1
        b.ne    2b

        mov     x1, x0

        MOVA_SUB_C_TILE x17

        subs    x9, x9, #1
        add     x17, x17, #256
        b.ne    1b
        .inst 0xd503467f            // smstop
        ldp     d8, d9, [sp, #0]
        ldp     d10, d11, [sp, #16]
        ldp     d12, d13, [sp, #32]
        ldp     d14, d15, [sp, #48]
        ldp     d16, d17, [sp, #64]
        add     sp, sp, #80
        ret
        .size sme_update_16x32_batch, .-sme_update_16x32_batch

        .global sme_update_16x32_rows_batch
        .type sme_update_16x32_rows_batch, %function
/*
 * Apply one packed B[K,32] panel to mt consecutive packed A[K,16] tiles.
 * The C tiles are separated by 16 rows of ldc_bytes.  SME stays enabled
 * across all row tiles; this is the persistent primitive for Case2.
 */
sme_update_16x32_rows_batch:
        sub     sp, sp, #80
        stp     d8, d9, [sp, #0]
        stp     d10, d11, [sp, #16]
        stp     d12, d13, [sp, #32]
        stp     d14, d15, [sp, #48]
        stp     d16, d17, [sp, #64]
        mov     x9, x5              // remaining A/C row tiles
        mov     x10, x3             // K, preserved across row tiles
        mov     x11, x0             // current A tile
        mov     x17, x2             // current C tile
        .inst 0xd503477f            // smstart
        .inst 0xd5033fdf            // isb
        .inst 0x2558e3e0            // ptrue p0.h
        .inst 0x2558e3e1            // ptrue p1.h
        .inst 0x2558e3e2            // ptrue p2.h
        .inst 0x2558e3e3            // ptrue p3.h
        mov     w12, #0
        mov     w13, #2
        mov     w14, #4
        mov     w15, #6
1:
        mov     x0, x11
        mov     x2, x17
        mov     x3, x10
        .inst 0xc00800ff            // zero {za}
2:
        ld1d    {z16.d}, p0/z, [x0]
        ld1d    {z17.d}, p0/z, [x0, #1, mul vl]
        ld1d    {z0.d}, p1/z, [x1]
        ld1d    {z1.d}, p1/z, [x1, #1, mul vl]
        ld1d    {z2.d}, p1/z, [x1, #2, mul vl]
        ld1d    {z3.d}, p2/z, [x1, #3, mul vl]
        .inst 0x80d00400
        .inst 0x80d00422
        .inst 0x80d00444
        .inst 0x80d00866
        .inst 0x80d16401
        .inst 0x80d16423
        .inst 0x80d16445
        .inst 0x80d16867
        add     x0, x0, #128
        add     x1, x1, #256
        subs    x3, x3, #1
        b.ne    2b
        sub     x1, x1, x10, lsl #8
        MOVA_SUB_C_TILE x2
        add     x11, x11, x10, lsl #7
        add     x17, x17, x4, lsl #4
        subs    x9, x9, #1
        b.ne    1b
        .inst 0xd503467f            // smstop
        ldp     d8, d9, [sp, #0]
        ldp     d10, d11, [sp, #16]
        ldp     d12, d13, [sp, #32]
        ldp     d14, d15, [sp, #48]
        ldp     d16, d17, [sp, #64]
        add     sp, sp, #80
        ret
        .size sme_update_16x32_rows_batch, .-sme_update_16x32_rows_batch

        .global sme_update_16x32_grid_batch
        .type sme_update_16x32_grid_batch, %function
/*
 * Apply mt packed A[K,16] tiles to nt packed B[K,32] tiles.  A and C are
 * advanced by row tile, while B is reused for every row tile.  The complete
 * row/column grid is covered by one SME region for a persistent worker.
 */
sme_update_16x32_grid_batch:
        sub     sp, sp, #80
        stp     d8, d9, [sp, #0]
        stp     d10, d11, [sp, #16]
        stp     d12, d13, [sp, #32]
        stp     d14, d15, [sp, #48]
        stp     d16, d17, [sp, #64]
        mov     x9, x5              // number of B/C column tiles
        mov     x10, x6             // remaining A/C row tiles
        mov     x11, x3             // K, preserved across all tiles
        mov     x17, x2             // current C row-tile base
        mul     x6, x9, x11
        add     x2, x1, x6, lsl #8  // one-past-last packed B tile
        .inst 0xd503477f            // smstart
        .inst 0xd5033fdf            // isb
        .inst 0x2558e3e0            // ptrue p0.h
        .inst 0x2558e3e1            // ptrue p1.h
        .inst 0x2558e3e2            // ptrue p2.h
        .inst 0x2558e3e3            // ptrue p3.h
        mov     w12, #0
        mov     w13, #2
        mov     w14, #4
        mov     w15, #6
1:
3:
        mov     x3, x11
        .inst 0xc00800ff            // zero {za}
4:
        ld1d    {z16.d}, p0/z, [x0]
        ld1d    {z17.d}, p0/z, [x0, #1, mul vl]
        ld1d    {z0.d}, p1/z, [x1]
        ld1d    {z1.d}, p1/z, [x1, #1, mul vl]
        ld1d    {z2.d}, p1/z, [x1, #2, mul vl]
        ld1d    {z3.d}, p2/z, [x1, #3, mul vl]
        .inst 0x80d00400
        .inst 0x80d00422
        .inst 0x80d00444
        .inst 0x80d00866
        .inst 0x80d16401
        .inst 0x80d16423
        .inst 0x80d16445
        .inst 0x80d16867
        add     x0, x0, #128
        add     x1, x1, #256
        subs    x3, x3, #1
        b.ne    4b
        MOVA_SUB_C_TILE x17
        add     x17, x17, #256
        sub     x0, x0, x11, lsl #7
        cmp     x1, x2
        b.ne    3b
        mul     x6, x9, x11
        sub     x1, x2, x6, lsl #8
        sub     x17, x17, x9, lsl #8
        add     x17, x17, x4, lsl #4
        add     x0, x0, x11, lsl #7
        subs    x10, x10, #1
        b.ne    1b
        .inst 0xd503467f            // smstop
        ldp     d8, d9, [sp, #0]
        ldp     d10, d11, [sp, #16]
        ldp     d12, d13, [sp, #32]
        ldp     d14, d15, [sp, #48]
        ldp     d16, d17, [sp, #64]
        add     sp, sp, #80
        ret
        .size sme_update_16x32_grid_batch, .-sme_update_16x32_grid_batch
