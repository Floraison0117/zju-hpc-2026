#include <arm_sve.h>

__attribute__((noinline))
svfloat64_t fmmla_probe(svfloat64_t acc, svfloat64_t a, svfloat64_t b)
{
    return svmmla_f64(acc, a, b);
}

int main(void)
{
    svbool_t pg = svptrue_b64();
    svfloat64_t one = svdup_f64(1.0);
    svfloat64_t acc = svdup_f64(0.0);
    acc = fmmla_probe(acc, one, one);
    return (int)svlastb_f64(pg, acc);
}
