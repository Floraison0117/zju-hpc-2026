#ifndef KBLAS_H
#define KBLAS_H
enum CBLAS_ORDER { CblasRowMajor=101, CblasColMajor=102 };
enum CBLAS_TRANSPOSE { CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113 };
void cblas_dgemm(int order,int ta,int tb,int m,int n,int k,double a,const double* A,int lda,const double* B,int ldb,double b,double* C,int ldc);
void BlasSetNumThreads(int n);
#endif
