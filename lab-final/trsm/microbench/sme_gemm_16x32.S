        .text
        .align 2
        .global sme_gemm_16x32
        .type sme_gemm_16x32, %function
/* C[16,32] = A_packed[K,16] * B_packed[K,32]. */
sme_gemm_16x32:
        sub     sp, sp, #80
        stp     d8, d9, [sp, #0]
        stp     d10, d11, [sp, #16]
        stp     d12, d13, [sp, #32]
        stp     d14, d15, [sp, #48]
        stp     d16, d17, [sp, #64]
        .inst 0xd503477f        // smstart
        .inst 0xd5033fdf        // isb
        .inst 0x2558e3e0        // ptrue p0.h
        .inst 0x2558e3e1        // ptrue p1.h
        .inst 0x2558e3e2        // ptrue p2.h
        .inst 0x2558e3e3        // ptrue p3.h
        .inst 0xc00800ff        // zero {za}
1:
        ld1d    {z16.d}, p0/z, [x0]
        ld1d    {z17.d}, p0/z, [x0, #1, mul vl]
        ld1d    {z0.d}, p1/z, [x1]
        ld1d    {z1.d}, p1/z, [x1, #1, mul vl]
        ld1d    {z2.d}, p1/z, [x1, #2, mul vl]
        ld1d    {z3.d}, p2/z, [x1, #3, mul vl]
        .inst 0x80d00400        // fmopa za0.d, p1/m, p0/m, z0.d, z16.d
        .inst 0x80d00422        // fmopa za2.d, p1/m, p0/m, z1.d, z16.d
        .inst 0x80d00444        // fmopa za4.d, p1/m, p0/m, z2.d, z16.d
        .inst 0x80d00866        // fmopa za6.d, p2/m, p0/m, z3.d, z16.d
        .inst 0x80d16401        // fmopa za1.d, p1/m, p3/m, z0.d, z17.d
        .inst 0x80d16423        // fmopa za3.d, p1/m, p3/m, z1.d, z17.d
        .inst 0x80d16445        // fmopa za5.d, p1/m, p3/m, z2.d, z17.d
        .inst 0x80d16867        // fmopa za7.d, p2/m, p3/m, z3.d, z17.d
        add     x0, x0, #128
        add     x1, x1, #256
        subs    x3, x3, #1
        b.ne    1b

        mov     x5, x2
        add     x6, x2, x4
        add     x7, x6, x4
        add     x8, x7, x4
        mov     w12, #0
        mov     w13, #2
        mov     w14, #4
        mov     w15, #6
        // rows 0..7, columns 0..31
        .inst 0xc0c28400; st1d {z0.d}, p0, [x5]
        .inst 0xc0c28482; st1d {z2.d}, p0, [x5, #1, mul vl]
        .inst 0xc0c28504; st1d {z4.d}, p0, [x5, #2, mul vl]
        .inst 0xc0c28986; st1d {z6.d}, p0, [x5, #3, mul vl]
        .inst 0xc0c28428; st1d {z8.d}, p0, [x6]
        .inst 0xc0c284aa; st1d {z10.d}, p0, [x6, #1, mul vl]
        .inst 0xc0c2852c; st1d {z12.d}, p0, [x6, #2, mul vl]
        .inst 0xc0c289ae; st1d {z14.d}, p0, [x6, #3, mul vl]
        add     x5, x5, x4, lsl #2
        add     x6, x6, x4, lsl #2
        .inst 0xc0c2a410; st1d {z16.d}, p0, [x7]
        .inst 0xc0c2a492; st1d {z18.d}, p0, [x7, #1, mul vl]
        .inst 0xc0c2a514; st1d {z20.d}, p0, [x7, #2, mul vl]
        .inst 0xc0c2a996; st1d {z22.d}, p0, [x7, #3, mul vl]
        .inst 0xc0c2a438; st1d {z24.d}, p0, [x8]
        .inst 0xc0c2a4a1; st1d {z1.d}, p0, [x8, #1, mul vl]
        .inst 0xc0c2a523; st1d {z3.d}, p0, [x8, #2, mul vl]
        .inst 0xc0c2a9a5; st1d {z5.d}, p0, [x8, #3, mul vl]
        add     x7, x7, x4, lsl #2
        add     x8, x8, x4, lsl #2
        .inst 0xc0c2c407; st1d {z7.d}, p0, [x5]
        .inst 0xc0c2c489; st1d {z9.d}, p0, [x5, #1, mul vl]
        .inst 0xc0c2c50b; st1d {z11.d}, p0, [x5, #2, mul vl]
        .inst 0xc0c2c98d; st1d {z13.d}, p0, [x5, #3, mul vl]
        .inst 0xc0c2c42f; st1d {z15.d}, p0, [x6]
        .inst 0xc0c2c4b1; st1d {z17.d}, p0, [x6, #1, mul vl]
        .inst 0xc0c2c52c; st1d {z12.d}, p0, [x6, #2, mul vl]
        .inst 0xc0c2c9ac; st1d {z12.d}, p0, [x6, #3, mul vl]
        add     x5, x5, x4, lsl #2
        add     x6, x6, x4, lsl #2
        .inst 0xc0c2e417; st1d {z23.d}, p0, [x7]
        .inst 0xc0c2e480; st1d {z0.d}, p0, [x7, #1, mul vl]
        .inst 0xc0c2e502; st1d {z2.d}, p0, [x7, #2, mul vl]
        .inst 0xc0c2e984; st1d {z4.d}, p0, [x7, #3, mul vl]
        .inst 0xc0c2e426; st1d {z6.d}, p0, [x8]
        .inst 0xc0c2e4a8; st1d {z8.d}, p0, [x8, #1, mul vl]
        .inst 0xc0c2e52a; st1d {z10.d}, p0, [x8, #2, mul vl]
        .inst 0xc0c2e9ac; st1d {z12.d}, p0, [x8, #3, mul vl]

        // rows 8..15, columns 0..31
        add     x5, x2, x4, lsl #3
        add     x6, x5, x4
        add     x7, x6, x4
        add     x8, x7, x4
        .inst 0xc0c28440; st1d {z0.d}, p0, [x5]
        .inst 0xc0c284c2; st1d {z2.d}, p0, [x5, #1, mul vl]
        .inst 0xc0c28544; st1d {z4.d}, p0, [x5, #2, mul vl]
        .inst 0xc0c289c6; st1d {z6.d}, p0, [x5, #3, mul vl]
        .inst 0xc0c28468; st1d {z8.d}, p0, [x6]
        .inst 0xc0c284ea; st1d {z10.d}, p0, [x6, #1, mul vl]
        .inst 0xc0c2856c; st1d {z12.d}, p0, [x6, #2, mul vl]
        .inst 0xc0c289ee; st1d {z14.d}, p0, [x6, #3, mul vl]
        add     x5, x5, x4, lsl #2
        add     x6, x6, x4, lsl #2
        .inst 0xc0c2a450; st1d {z16.d}, p0, [x7]
        .inst 0xc0c2a4d2; st1d {z18.d}, p0, [x7, #1, mul vl]
        .inst 0xc0c2a554; st1d {z20.d}, p0, [x7, #2, mul vl]
        .inst 0xc0c2a9d6; st1d {z22.d}, p0, [x7, #3, mul vl]
        .inst 0xc0c2a478; st1d {z24.d}, p0, [x8]
        .inst 0xc0c2a4e1; st1d {z1.d}, p0, [x8, #1, mul vl]
        .inst 0xc0c2a563; st1d {z3.d}, p0, [x8, #2, mul vl]
        .inst 0xc0c2a9e5; st1d {z5.d}, p0, [x8, #3, mul vl]
        add     x7, x7, x4, lsl #2
        add     x8, x8, x4, lsl #2
        .inst 0xc0c2c447; st1d {z7.d}, p0, [x5]
        .inst 0xc0c2c4c9; st1d {z9.d}, p0, [x5, #1, mul vl]
        .inst 0xc0c2c54b; st1d {z11.d}, p0, [x5, #2, mul vl]
        .inst 0xc0c2c9cd; st1d {z13.d}, p0, [x5, #3, mul vl]
        .inst 0xc0c2c46f; st1d {z15.d}, p0, [x6]
        .inst 0xc0c2c4f1; st1d {z17.d}, p0, [x6, #1, mul vl]
        .inst 0xc0c2c56c; st1d {z12.d}, p0, [x6, #2, mul vl]
        .inst 0xc0c2c9ec; st1d {z12.d}, p0, [x6, #3, mul vl]
        add     x5, x5, x4, lsl #2
        add     x6, x6, x4, lsl #2
        .inst 0xc0c2e457; st1d {z23.d}, p0, [x7]
        .inst 0xc0c2e4c0; st1d {z0.d}, p0, [x7, #1, mul vl]
        .inst 0xc0c2e542; st1d {z2.d}, p0, [x7, #2, mul vl]
        .inst 0xc0c2e9c4; st1d {z4.d}, p0, [x7, #3, mul vl]
        .inst 0xc0c2e466; st1d {z6.d}, p0, [x8]
        .inst 0xc0c2e4e8; st1d {z8.d}, p0, [x8, #1, mul vl]
        .inst 0xc0c2e56a; st1d {z10.d}, p0, [x8, #2, mul vl]
        .inst 0xc0c2e9ec; st1d {z12.d}, p0, [x8, #3, mul vl]
        .inst 0xd503467f        // smstop
        ldp     d8, d9, [sp, #0]
        ldp     d10, d11, [sp, #16]
        ldp     d12, d13, [sp, #32]
        ldp     d14, d15, [sp, #48]
        ldp     d16, d17, [sp, #64]
        add     sp, sp, #80
        ret
        .size sme_gemm_16x32, .-sme_gemm_16x32
