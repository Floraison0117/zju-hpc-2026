#include <stdio.h>
#include <arm_sve.h>

int main(void)
{
    printf("svcntb=%lu\n", (unsigned long)svcntb());
    printf("svcntd=%lu\n", (unsigned long)svcntd());
    return 0;
}
