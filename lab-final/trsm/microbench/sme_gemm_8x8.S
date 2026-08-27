        .text
        .align 2
        .global sme_gemm_8x8
        .type sme_gemm_8x8, %function
/* C = A_packed * B_packed, A/B are K x 8 row-major packed panels. */
sme_gemm_8x8:
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
        .inst 0xc00800ff        // zero {za}
1:
        ld1d    {z0.d}, p0/z, [x1]
        ld1d    {z16.d}, p1/z, [x0]
        .inst 0x80d00400        // fmopa za0.d, p1/m, p0/m, z0.d, z16.d
        add     x0, x0, #64
        add     x1, x1, #64
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
        .inst 0xc0c28401        // mov z1.d, p1/m, za0v.d[w12, 0]
        st1d    {z1.d}, p0, [x5]
        add     x5, x5, x4, lsl #2
        .inst 0xc0c28429        // mov z9.d, p1/m, za0v.d[w12, 1]
        st1d    {z9.d}, p0, [x6]
        add     x6, x6, x4, lsl #2
        .inst 0xc0c2a411        // mov z17.d, p1/m, za0v.d[w13, 0]
        st1d    {z17.d}, p0, [x7]
        add     x7, x7, x4, lsl #2
        .inst 0xc0c2a438        // mov z24.d, p1/m, za0v.d[w13, 1]
        st1d    {z24.d}, p0, [x8]
        add     x8, x8, x4, lsl #2
        .inst 0xc0c2c407        // mov z7.d, p1/m, za0v.d[w14, 0]
        st1d    {z7.d}, p0, [x5]
        add     x5, x5, x4, lsl #2
        .inst 0xc0c2c42f        // mov z15.d, p1/m, za0v.d[w14, 1]
        st1d    {z15.d}, p0, [x6]
        add     x6, x6, x4, lsl #2
        .inst 0xc0c2e417        // mov z23.d, p1/m, za0v.d[w15, 0]
        st1d    {z23.d}, p0, [x7]
        add     x7, x7, x4, lsl #2
        .inst 0xc0c2e426        // mov z6.d, p1/m, za0v.d[w15, 1]
        st1d    {z6.d}, p0, [x8]
        .inst 0xd503467f        // smstop
        ldp     d8, d9, [sp, #0]
        ldp     d10, d11, [sp, #16]
        ldp     d12, d13, [sp, #32]
        ldp     d14, d15, [sp, #48]
        ldp     d16, d17, [sp, #64]
        add     sp, sp, #80
        ret
        .size sme_gemm_8x8, .-sme_gemm_8x8
