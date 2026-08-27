        .text
        .align 2
        .global sme_f64f64_probe_inst
        .type sme_f64f64_probe_inst, %function
sme_f64f64_probe_inst:
        .inst 0xd503477f        // smstart
        .inst 0xd5033fdf        // isb
        .inst 0x2558e3e0        // ptrue p0.h
        .inst 0x2558e3e1        // ptrue p1.h
        .inst 0xc00800ff        // zero {za}
        .inst 0x05e03b50        // mov z16.d, x26
        .inst 0x80d00400        // fmopa za0.d, p1/m, p0/m, z0.d, z16.d
        .inst 0xd503467f        // smstop
        ret
        .size sme_f64f64_probe_inst, .-sme_f64f64_probe_inst
