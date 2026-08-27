# Lab4 优化知识库索引 (KB Index)

> 本目录是 lab4 闭环优化 agent 的持久化知识库，对应 KernelEvolve 的 hierarchical KB + KernelPro 的 search memory。
> agent 每轮迭代：读运行时上下文 (profiling + error) → 诊断瓶颈 → 检索本 KB 匹配模式 → 合成提示 → 生成候选 → 验证 → 量化 fitness → 回写本 KB。

## 目录结构

```
assets/lab4/kb/
  index.md                      # 本文件：层级索引（KernelEvolve index.md 模式）
  hardware/
    task1-arm-cpu.md            # 任务一：鲲鹏 920B / MPI Fortran / BSSN stencil
    task2-a100-gpu.md           # 任务二：A100 MIG / CUDA / rhs_kernel
  patterns/
    cpu-stencil.md              # CPU 瓶颈类 → 触发条件 → 优化指令（KernelPro 微剖析工具模式）
    gpu-kernel.md              # GPU 瓶颈类 → 触发条件 → 优化指令
  search-memory.md             # 搜索记忆：28+ 已测杠杆 + 结果 + 死路（防重复）
  fitness-gates.md             # fitness F、正确性硬约束、三级验证、墙钟、OJ 配置
```

## 检索约定（KernelEvolve 两段式）

1. context-memory 诊断：从 profiling 原始输出识别瓶颈类（compute/memory/latency-bound，CPU 还区分 straggler/analysis）。
2. deep-search 检索：按瓶颈类到 `patterns/` 取匹配模式；命中后读 `search-memory.md` 确认该杠杆是否已测及其结果，避免重复死路。

## 当前交付状态（2026-08-23，KB 基线）

| 任务 | 主机 | 当前 OJ | 目标 | 主瓶颈 | 状态 |
|---|---|---|---|---|---|
| 任务一 ABE CPU | zju-hpc-arm | 408.9s / 86 分 | <340s(100分) | compute_rhs_bssn_ 30%，IPC 1.69，数据依赖 stencil | 约束内已达极限，<340s 需算法重写 |
| 任务二 ABEGPU | zju-hpc-lab2 | 1044.26s / 64 分（部署态实测 901.24s，job 153420） | <340s(100分) | rhs_kernel 69.1%，latency-bound，natural regs 255 = sm_80 上限 | **寄存器/live-set/spill/缓存路径杠杆已穷尽（iter14 P5 收官），GPU 侧优化正式收敛；<340s 需算法级重构（越边界，需授权）** |

详细已部署栈见 `search-memory.md` 的"已部署栈"小节。
