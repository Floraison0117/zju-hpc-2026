        .text
        .align 2

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

/*
 * K-blocked 16x32 SME update kernel: C[16,32] -= A_pack[K,16] * B_pack[K,32].
 * K processed in 64-value blocks (working packs 8KB+16KB fit L1); the 8
 * FMOPAs per k are issued back-to-back; prfm pulls the next block.  C stays
 * in ZA across all blocks; single store at the end.
 * A-pack: chunk c at +c*128B.  B-pack: chunk c at +c*256B.
 */
        .global sme_update_16x32_kblock
        .type sme_update_16x32_kblock, %function
sme_update_16x32_kblock:
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

        /* x0 = a_pack, x1 = b_pack, x2 = c, x3 = k, x4 = ldc_bytes */
        mov     x8, #8192           /* prefetch distance A */
        mov     x9, #16384          /* prefetch distance B */

2:      /* next K block: 64 values */
        cmp     x3, #64
        b.lt    4f
        sub     x3, x3, #64
        prfm    pldl1keep, [x0, x8]
        prfm    pldl1keep, [x1, x9]
        mov     x7, #64
3:      /* one k-value */
        ld1d    {z16.d}, p0/z, [x0]
        ld1d    {z17.d}, p0/z, [x0, #1, mul vl]
        ld1d    {z0.d}, p1/z, [x1]
        ld1d    {z1.d}, p1/z, [x1, #1, mul vl]
        ld1d    {z2.d}, p1/z, [x1, #2, mul vl]
        ld1d    {z3.d}, p2/z, [x1, #3, mul vl]
        .inst 0x80d00400            // za0.d, p1/m, p0/m, z0.d, z16.d
        .inst 0x80d00422            // za2.d, p1/m, p0/m, z1.d, z16.d
        .inst 0x80d00444            // za4.d, p1/m, p0/m, z2.d, z16.d
        .inst 0x80d00866            // za6.d, p2/m, p0/m, z3.d, z16.d
        .inst 0x80d16401            // za1.d, p1/m, p3/m, z0.d, z17.d
        .inst 0x80d16423            // za3.d, p1/m, p3/m, z1.d, z17.d
        .inst 0x80d16445            // za5.d, p1/m, p3/m, z2.d, z17.d
        .inst 0x80d16867            // za7.d, p2/m, p3/m, z3.d, z17.d
        add     x0, x0, #128
        add     x1, x1, #256
        subs    x7, x7, #1
        b.ne    3b
        b       2b

4:      /* tail: remaining k < 64 */
        cbz     x3, 5f
        mov     x7, x3
6:      /* one k-value (tail) */
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
        subs    x7, x7, #1
        b.ne    6b

5:
        MOVA_SUB_C_TILE x2
        .inst 0xd503467f            // smstop
        ldp     d8, d9, [sp, #0]
        ldp     d10, d11, [sp, #16]
        ldp     d12, d13, [sp, #32]
        ldp     d14, d15, [sp, #48]
        ldp     d16, d17, [sp, #64]
        add     sp, sp, #80
        ret
        .size sme_update_16x32_kblock, .-sme_update_16x32_kblock
