#include <stdio.h>
#include <stdlib.h>

extern void sme_mova_h_probe(double *x);

int main(void)
{
    double *x = aligned_alloc(64, 8 * sizeof(double));
    if (x == NULL) return 2;
    for (int i = 0; i < 8; ++i) x[i] = 1.0 + i;
    sme_mova_h_probe(x);
    printf("mova_h_probe_out=");
    for (int i = 0; i < 8; ++i) printf(" %.17g", x[i]);
    putchar('\n');
    free(x);
    return 0;
}
