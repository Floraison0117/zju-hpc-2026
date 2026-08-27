# Lab4 闭环优化 Agent 工具包 (opt)

> 适配 KernelPro (arXiv 2606.26453) 与 KernelEvolve (arXiv 2512.23236) 的闭环内核优化工作流，专门用于 Lab4 的两个任务。
> 由 pi subagent `lab4-opt`（见 `.pi/agents/lab4-opt.md`）驱动，也可人工调用。

## 两个任务的映射

| 任务 | 主机 | 范式来源 |
|---|---|---|
| 任务一 ABE CPU | zju-hpc-arm（鲲鹏 920B, MPI Fortran） | perf stat/flat/objdump/fopt-info 微剖析 + BSSN stencil 模式 |
| 任务二 ABEGPU | zju-hpc-lab2（A100 MIG, CUDA） | ncu roofline + SASS + nvcc ptxas 微剖析 + rhs_kernel 模式 |

## 目录结构

```
assets/lab4/
  kb/                          # 持久化知识库（KernelEvolve KB + KernelPro search memory）
    index.md                   # 层级索引
    hardware/task1-arm-cpu.md  # 任务一硬件约束
    hardware/task2-a100-gpu.md # 任务二硬件约束
    patterns/cpu-stencil.md    # CPU 瓶颈类 → 优化指令（语义反馈算子）
    patterns/gpu-kernel.md     # GPU 瓶颈类 → 优化指令
    search-memory.md           # 28+ 已测杠杆全表（防重复死路）
    fitness-gates.md           # F、正确性硬约束、三级验证、墙钟、OJ 配置
  opt/
    diagnose/diag_cpu.sh       # CPU Stage-1/2 诊断 → 中文指令
    diagnose/diag_gpu.sh       # GPU Stage-1/2 诊断 → 中文指令
    fitness/level0_static.sh   # Level 0 静态检查（秒级）
    fitness/level1_short_ab.sh # Level 1 短跑 A/B（分钟级）
    fitness/level2_full_check.sh # Level 2 全量 + check.sh（权威门）
    fitness/fitness.sh         # 计算 F + 写节点元数据
    search/search_loop.sh      # 图搜索驱动（greedy/MCTS）
    search/metadata_store.sh   # 搜索树元数据 CRUD
    search/snapshot.sh          # 部署前快照 + hash 校验
    run_opt.sh                 # 顶层入口
```

## 典型工作流（一次迭代）

```bash
# 1. 诊断（Stage-1 瓶颈分类 + Stage-2 微剖析 → 语义反馈指令）
bash assets/lab4/opt/run_opt.sh cpu diagnose ~/lab4-cpu-cand
# 2. 读 KB 模式 + search-memory，确认候选杠杆未测
# 3. 搜索循环（Level 0 → Level 1 A/B → fitness → 回写）
bash assets/lab4/opt/run_opt.sh cpu search
# 4. 正收益且 bit-exact → 部署前快照
bash assets/lab4/opt/run_opt.sh cpu snapshot
# 5. Level 2 全量验收（用户授权后，主 agent 部署）
bash assets/lab4/opt/run_opt.sh cpu full ~/lab4-cpu-cand
```

## 诚信边界

- 一次只改一个变量；bit-exact 优先；check.sh PASS 是硬约束。
- formal 源只读；候选 `cp -r` 隔离；改动用 patch 脚本记录源哈希。
- 不改 runner/timing/评测用例刷分；不硬编码输出；候选不直接提交 OJ（主 agent 走）。

## 与论文的对应

| 论文组件 | 本工具包实现 |
|---|---|
| KernelPro 语义反馈算子（微剖析工具） | `diagnose/*.sh` + `kb/patterns/*.md`（原始指标 → 中文指令） |
| KernelPro 两阶段调用（roofline 分类 → 过滤工具） | `diag_*.sh` Stage-1 分类 + Stage-2 主动式全跑 |
| KernelPro search memory + dead-end pruning | `kb/search-memory.md`（28+ 杠杆死路表） |
| KernelEvolve `(F, π_sel, O, τ)` | `fitness/fitness.sh`(F) + `search/search_loop.sh`(π/τ) + agent(O) |
| KernelEvolve 持久化 KB + 两段式检索 | `kb/`（context-memory 诊断 → deep-search 检索模式） |
| KernelEvolve metadata+object store | `search/tree/*.md` 节点元数据 + `metadata_store.sh` |
| KernelEvolve universal operator | 单一 agent + 检索增强提示（不用 debug/improve 多模板） |
