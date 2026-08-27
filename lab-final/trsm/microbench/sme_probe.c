#include <stdio.h>

extern void sme_f64f64_probe(void);

int main(void)
{
    sme_f64f64_probe();
    puts("sme_f64f64_probe=ok");
    return 0;
}
