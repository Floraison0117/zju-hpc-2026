#include <stdio.h>

extern void sme_f64f64_probe_inst(void);

int main(void)
{
    sme_f64f64_probe_inst();
    puts("sme_f64f64_probe_inst=ok");
    return 0;
}
