#ifndef SME_H
#define SME_H
extern void sme_update_16x32_pipe(const double *a_pack, const double *b_pack, double *c, int k, int ldc_bytes);
extern void sme_update_16x32(const double *a_pack, const double *b_pack, double *c, int k, int ldc_bytes);
#endif
