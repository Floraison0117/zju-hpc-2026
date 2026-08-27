#include <stdio.h>
#include <time.h>

extern void sme_fmopa_hot(long iterations);

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

int main(void)
{
    const double start = now();
    sme_fmopa_hot(1);
    const double end = now();
    printf("start=%.9f end=%.9f delta_ns=%.0f\n",
           start, end, (end - start) * 1e9);
    return 0;
}
