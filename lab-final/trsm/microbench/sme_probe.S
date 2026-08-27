        .arch armv9-a+sme
        .text
        .align 2
        .global sme_f64f64_probe
        .type sme_f64f64_probe, %function
sme_f64f64_probe:
        smstart
        ptrue p0.d
        dup z0.d, xzr
        fmopa za0.d, p0/m, p0/m, z0.d, z0.d
        smstop
        ret
        .size sme_f64f64_probe, .-sme_f64f64_probe
