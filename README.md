# HPC 高性能计算实验仓库

2025-2026 暑期 HPC 课程实验。包含 Lab1-Lab5(含 Lab3.5、Lab4.5)的实验报告与最终代码,以及大作业(鲲鹏挑战赛 S2 赛季)的优化代码。所有实验均已交付,报告以 Typst 编写。

## 目录结构

```
labs/            实验报告源文件 (.typ) 与编译产物 (.pdf)
assets/labN/     报告引用的截图与素材
codes/           各实验的最终提交代码
misc/            报告写作工作流等杂项文档
hpc101/          课程提供的 LLM 推理框架 (Lab5)
ascend-c/        昇腾 Ascend C 教程与示例 (Lab3.5)
AGENTS.md        仓库工作规范: 实验环境、编辑规则、Typst 约定、验证流程
trsm_submission.zip  大作业 TRSM 提交包
```

## 实验概览

| Lab | 主题 | 平台 | 代码位置 |
|---|---|---|---|
| Lab1 | 集群搭建、作业调度与 HPL 并行基准测试 | 本地 WSL2 Docker 4 容器, OpenMPI | (报告中含命令记录) |
| Lab2 | W8A8 MoE 前向优化, AVX-512 VNNI / AMX-INT8 | Intel Xeon Gold 5418Y | `codes/lab2/moe_opt.cpp` |
| Lab3 | GDN prefill forward kernel, TileLang | NVIDIA H800 MIG 1g.10gb | `codes/lab3/tilelang_fwd.py` |
| Lab3.5 | FusedAddRmsNorm 融合算子优化 | 华为昇腾 910B4 NPU | `codes/lab3p5/` |
| Lab4 | AMSS-NCKU 双黑洞并合: 任务一 ABE CPU, 任务二 ABEGPU | 鲲鹏 920B / A100 MIG 1g.10gb | `codes/lab4-cpu/`, `codes/lab4-gpu/` |
| Lab4.5 | INT8 Tensor Core 模拟 FP64 GEMM | A100 MIG 1g.10gb | `codes/lab4p5/my_int8_fp64.cu` |
| Lab5 | Gemma4-12B INT4 量化与推理吞吐优化 | H800 MIG 1g.10gb | `codes/lab5/` |
| Final | 鲲鹏挑战赛 S2: CONV / TRSM / ZGEMM | 鲲鹏 920F, 深超算 NSCC-SZ | `codes/lab-final/{conv,trsm,zgemm}/` |

大作业已完成 TRSM 优化(报告见 `labs/lab_final/lab_final_trsm.pdf`),提交包为根目录 `trsm_submission.zip`。

## 编译报告

```powershell
typst compile --root . .\labs\lab1.typ .\labs\lab1.pdf
```

报告排版约定(字体 `("Palatino Linotype", "KaiTi")`、表格居中等)详见 `AGENTS.md` 与 `misc/typst-lab-report-workflow.md`。

## 实验环境速查

- Lab1 在本地 WSL2 Docker 中完成,其余实验均在 ZJU `zju-hpc-lab2` / `zju-hpc-arm` 远程主机上运行,GPU 分区为单用户 MIG 切片,单作业 30 分钟墙钟。
- 大作业使用深超算(NSCC-SZ)鲲鹏 920F 集群,经天融信 VPN 接入,多瑙(Donau)调度器。
- 详细访问方式、集群凭据注意事项与各实验硬件配置见 `AGENTS.md`。
