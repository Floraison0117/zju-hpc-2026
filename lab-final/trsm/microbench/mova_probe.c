#include <stdio.h>
#include <stdlib.h>

extern void sme_mova_probe(double *x);

int main(void)
{
    double *x = aligned_alloc(64, 8 * sizeof(double));
    if (x == NULL) return 2;
    for (int i = 0; i < 8; ++i) x[i] = 1.0 + i;
    sme_mova_probe(x);
    printf("mova_probe_out=%.17g %.17g %.17g %.17g\n",
           x[0], x[1], x[2], x[3]);
    free(x);
    return 0;
}
