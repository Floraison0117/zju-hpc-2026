        .text
        .align 2
        .global sme_fmopa_hot
        .type sme_fmopa_hot, %function
sme_fmopa_hot:
        sub     sp, sp, #96
        stp     d8, d9, [sp, #0]
        stp     d10, d11, [sp, #16]
        stp     d12, d13, [sp, #32]
        stp     d14, d15, [sp, #48]
        stp     d16, d17, [sp, #64]
        str     x26, [sp, #80]
        .inst 0xd503477f        // smstart
        .inst 0xd5033fdf        // isb
        .inst 0x2558e3e0        // ptrue p0.h
        .inst 0x2558e3e1        // ptrue p1.h
        .inst 0x2558e3e2        // ptrue p2.h
        .inst 0x2558e3e3        // ptrue p3.h
        .inst 0xc00800ff        // zero {za}
        mov     x26, #1
        .inst 0x05e03b40        // mov z0.d, x26
        .inst 0x05e03b50        // mov z16.d, x26
1:
        .inst 0x80d00400        // fmopa za0.d, p1/m, p0/m, z0.d, z16.d
        .inst 0x80d16401        // fmopa za1.d, p1/m, p3/m, z0.d, z16.d
        .inst 0x80d00422        // fmopa za2.d, p1/m, p0/m, z1.d, z16.d
        .inst 0x80d16423        // fmopa za3.d, p1/m, p3/m, z1.d, z17.d
        .inst 0x80d00444        // fmopa za4.d, p1/m, p0/m, z2.d, z16.d
        .inst 0x80d16445        // fmopa za5.d, p1/m, p3/m, z2.d, z17.d
        .inst 0x80d00866        // fmopa za6.d, p2/m, p0/m, z3.d, z16.d
        .inst 0x80d16867        // fmopa za7.d, p2/m, p3/m, z3.d, z17.d
        subs    x0, x0, #1
        b.gt    1b
        .inst 0xd503467f        // smstop
        ldr     x26, [sp, #80]
        ldp     d8, d9, [sp, #0]
        ldp     d10, d11, [sp, #16]
        ldp     d12, d13, [sp, #32]
        ldp     d14, d15, [sp, #48]
        ldp     d16, d17, [sp, #64]
        add     sp, sp, #96
        ret
        .size sme_fmopa_hot, .-sme_fmopa_hot
