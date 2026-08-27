# 鲲鹏高性能计算全球挑战赛（S2赛季）ZGEMM 优化赛题

## 编译
``` shell
gcc -O3 bench_zgemm.c zgemm.c -o zgemm_test -lm -fopenmp
```

## 运行
本次三个测试用例，运行方法如下：
``` shell
OMP_NUM_THREADS=38 numactl -N 1 ./zgemm_test 7427 7427 256  1    #参数依次是 M N K test_runs； test_runs表示重复测试次数，使性能稳定即可
OMP_NUM_THREADS=38 numactl -N 1 ./zgemm_test 14848 14848 256  1
OMP_NUM_THREADS=38 numactl -N 1 ./zgemm_test 37360 8192 512  1
```

## 调优
修改zgemm.c源文件的cblas_zgemm函数优化性能。

校验结果无误，按性能（GFLOPS）排名。