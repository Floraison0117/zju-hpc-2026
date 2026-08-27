        .text
        .align 2
        .global sme_mova_probe
        .type sme_mova_probe, %function
sme_mova_probe:
        .inst 0xd503477f            // smstart
        .inst 0xd5033fdf            // isb
        .inst 0x2558e3e0            // ptrue p0.h
        .inst 0x2558e3e1            // ptrue p1.h
        mov     w12, #0
        .inst 0xe0df0000            // ld1d za0h.d[w12, #0], p0/z, [x0]
        .inst 0xc0c28400            // mov z0.d, p1/m, za0v.d[w12, 0]
        .inst 0xe5e0e000            // st1d z0.d, p0, [x0]
        .inst 0xd503467f            // smstop
        ret
        .size sme_mova_probe, .-sme_mova_probe
