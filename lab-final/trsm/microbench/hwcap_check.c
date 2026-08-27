#include <stdio.h>
#include <sys/auxv.h>
#include <asm/hwcap.h>

int main(void)
{
    const unsigned long hwcap = getauxval(AT_HWCAP);
    const unsigned long hwcap2 = getauxval(AT_HWCAP2);
#ifdef HWCAP2_SVEF64MM
    const unsigned long svef64mm = HWCAP2_SVEF64MM;
    const int has_svef64mm = (hwcap2 & svef64mm) != 0;
    const int header_defined = 1;
#else
    /* Older vendor headers omit the upstream HWCAP2 name. */
    const unsigned long svef64mm = 1UL << 11;
    const int has_svef64mm = (hwcap2 & svef64mm) != 0;
    const int header_defined = 0;
#endif
    printf("AT_HWCAP=0x%lx\n", hwcap);
    printf("AT_HWCAP2=0x%lx\n", hwcap2);
    printf("HWCAP2_SVEF64MM=0x%lx\n", svef64mm);
    printf("header_defined=%d\n", header_defined);
    printf("has_svef64mm=%d\n", has_svef64mm);
    return 0;
}
