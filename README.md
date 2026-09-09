# HPC 高性能计算实验与大作业

2025–2026 暑期 HPC 课程实验仓库，记录 Lab1–Lab5（含 Lab3.5、Lab4.5）的实验报告、实验代码与配套素材，并包含鲲鹏高性能计算全球挑战赛 S2 赛季大作业的最终交付文件。报告使用 Typst 编写。

## 内容概览

| 内容 | 主题 | 主要平台 | 入口 |
|---|---|---|---|
| Lab1 | 集群搭建、作业调度与 HPL 并行基准测试 | WSL2 Docker、OpenMPI | [报告](labs/lab1.pdf) · [源码](labs/lab1.typ) |
| Lab2 | W8A8 MoE 前向优化 | Intel Xeon Gold 5418Y、AVX-512 VNNI / AMX-INT8 | [报告](labs/lab2.pdf) · [`moe_opt.cpp`](codes/lab2/moe_opt.cpp) |
| Lab3 | GDN prefill forward kernel | NVIDIA H800 MIG、TileLang | [报告](labs/lab3.pdf) · [`tilelang_fwd.py`](codes/lab3/tilelang_fwd.py) |
| Lab3.5 | FusedAddRmsNorm 融合算子优化 | 华为昇腾 910B4 | [报告](labs/lab3.5.pdf) · [`codes/lab3p5/`](codes/lab3p5/) |
| Lab4 | AMSS-NCKU 双黑洞并合，ABE CPU 与 ABEGPU | 鲲鹏 920B、NVIDIA A100 MIG | [报告](labs/lab4.pdf) · [`codes/lab4-cpu/`](codes/lab4-cpu/) · [`codes/lab4-gpu/`](codes/lab4-gpu/) |
| Lab4.5 | INT8 Tensor Core 模拟 FP64 GEMM | NVIDIA A100 MIG | [报告](labs/lab4.5.pdf) · [`my_int8_fp64.cu`](codes/lab4p5/my_int8_fp64.cu) |
| Lab5 | Gemma4-12B INT4 量化与推理吞吐优化 | NVIDIA H800 MIG | [报告](labs/lab5.pdf) · [`codes/lab5/`](codes/lab5/) |
| Final | CONV、TRSM、ZGEMM 优化 | 鲲鹏 920F、深超算 NSCC-SZ | [交付目录](HPC_final/) · [TRSM 报告](labs/lab_final_trsm.pdf) |

## 目录结构

```text
labs/                 实验报告源文件（.typ）与 PDF
assets/labN/          报告引用的截图与素材
codes/                各实验源码、脚本与实验记录
codes/lab-final/      大作业源码及各算子的说明
HPC_final/            大作业最终交付包
  report.pdf          大作业报告
  presentation.pptx  大作业汇报幻灯片
  CONV/ TRSM/ ZGEMM/ 三个独立算子的源码、脚本与结果
hpc101/               课程资料与 HPC 学习笔记
ascend-c/             Ascend C 教程与示例
misc/                 报告写作工作流等辅助文档
AGENTS.md             仓库协作、实验环境与构建约定
```

## 大作业

大作业包含三个独立内核：

- CONV：二维卷积优化，源码与运行说明见 [`HPC_final/CONV/`](HPC_final/CONV/)。
- TRSM：三角矩阵求解优化，当前交付版本与实验记录见 [`HPC_final/TRSM/`](HPC_final/TRSM/)；报告见 [`labs/lab_final_trsm.pdf`](labs/lab_final_trsm.pdf)。
- ZGEMM：复数矩阵乘法优化，结果与历史版本见 [`HPC_final/ZGEMM/`](HPC_final/ZGEMM/)。

正式运行使用鲲鹏 920F 计算节点、`OMP_NUM_THREADS=38` 和 `numactl -N 1`。各算子的编译命令、测试规模和运行脚本以对应目录中的 README 与脚本为准。

## 编译 Typst 报告

在仓库根目录执行：

```powershell
typst compile --root . .\labs\lab1.typ .\labs\lab1.pdf
```

将 `lab1` 替换为其他报告文件名即可。报告排版约定（字体 `("Palatino Linotype", "KaiTi")`、表格居中等）见 [`AGENTS.md`](AGENTS.md) 和 [`misc/typst-lab-report-workflow.md`](misc/typst-lab-report-workflow.md)。

## 实验环境

- Lab1 在本地 WSL2 Docker 环境完成，其余课程实验主要使用 ZJU HPC 主机与 NVIDIA MIG 分区。
- Final 使用深超算 NSCC-SZ 的鲲鹏 920F 集群和 Donau 调度器，与 ZJU 集群相互独立。
- 环境、构建方式、验证规则及远程运行注意事项见 [`AGENTS.md`](AGENTS.md)。

仓库中的 `tmp/`、`.remote_work/` 等目录仅用于临时工作，不属于正式交付物。
