#ifndef TRSM_SME_CANDIDATE_H
#define TRSM_SME_CANDIDATE_H

/* Return non-zero when the SME candidate performed the update. */
int trsm_sme_case3_update(int rows, int n, int bs,
                          const double *l21, int lda,
                          const double *bi, int ldb,
                          double *b2, int ldb2);

int trsm_sme_case2_update(int rows, int n, int bs,
                          const double *l21, int lda,
                          const double *bi, int ldb,
                          double *b2, int ldb2);

int trsm_sme_case3_left_update(int rows, int n, int k,
                               const double *lblock, int lda,
                               const double *b, int ldb,
                               double *c, int ldc);

#endif
