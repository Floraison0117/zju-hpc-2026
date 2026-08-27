# ABEGPU 340 秒冲刺：高层优化计划

> 状态：计划原文（2026-08-25 用户提供，主 agent 保存）。证据对照审查见 `plan-340s-sprint-review.md`。
> 基线：已部署 P1–P10 优化栈，743 s / RMS=0（job 157367，check.sh FINAL PASS bit-exact）。
> 执行记录回写 README-lab4.md §15-§16 与 assets/lab4/kb/search-memory.md。

目标：在不改变物理问题、网格规模、演化时间和必要输出的前提下，将正式端到端时间从当前 743 s 压到 340 s 以内，同时通过完整正确性检查。

本计划以当前已经验证的 P1–P10 优化栈为新基线。它不是继续堆叠 inline、编译参数或局部 shared-memory 试验，而是一次以 RHS 计算图、AMR 执行模型和全局调度数据流 为对象的架构级改造。

## 1. 先明确：340 秒意味着什么

当前 743 s 到 340 s，需要端到端再加速：

S_required = 743/340 = 2.185×，ΔT = 403 s = 54.2%。

因此，任何预期只有 1%–10% 的单点优化都不可能单独完成目标。官方实验说明也明确要求以完整程序的 This Program Cost 作为性能判据，并强调不能以减少物理计算、缩短演化或跳过输出来换取速度；任务二运行于单个 A100 MIG 实例上。参见 HPC101 Lab 4 实验说明。

基于目前 profile 中 RHS 约占 65%、prolong3 约占 16.5% 的结果，不能把各模块预算刚好相加到 340 s；为抵抗节点抖动和网格演化差异，应以 330 s 工程预算 冲刺，为正式阈值保留约 10 s 余量：

| 模块 | 当前规划占比 | 当前估算 | 目标加速 | 目标预算 |
|---|---|---|---|---|
| BSSN RHS | 约 65% | 约 483 s | ≥2.5× | ≤190 s |
| Prolongation | 约 16.5% | 约 123 s | ≥2.0× | ≤60 s |
| 其余演化、同步、通信、分析及固定开销 | 约 18.5% | 约 137 s | ≥1.7× | ≤80 s |
| 合计 | 100% | 743 s | 2.25× | ≤330 s |

这张表不是最终测量结论，而是项目的 硬验收预算。第一阶段必须用干净的端到端 trace 重新校准三项时间；若固定不可优化开销高于预期，则 RHS 的目标必须相应提高。

## 2. 总体技术路线

核心思路是建立一条新的 GPU 数据通路：

- 把当前由 host 逐 patch、逐变量、逐操作发射的碎片化工作，改成 GPU 可批处理的 patch descriptor + work queue；
- 对 RHS 不再尝试简单降低寄存器上限，而是按照方程依赖图重新安排"加载—求导—消费—释放"的顺序，形成 低 live-set 的 tiled producer–consumer kernel；
- 把 prolongation 从大量细粒度、依赖链很长的调用，重写成 跨 patch 批处理、内点与边界分流的并行插值执行器；
- 在新执行模型上融合适合融合的 RK 更新、边界处理和逐点操作，并仅在时间线证明有空隙时引入 streams 或 CUDA Graphs；
- 最后才利用正确性容限做选择性的浮点表达式重排或混合精度，绝不直接降低演化状态和 Ricci 主路径的精度。

现有结果已经证明，"强制压寄存器换 occupancy"会因 spill 变慢。NVIDIA 的 profiler 指南也将 excessive register pressure、local-memory spill 和 long-scoreboard stall 视为需要结合源级依赖与访存局部性一起处理的问题，而不是单看 occupancy；参见 Nsight Compute Profiling Guide。

## 3. Phase 0：建立可用于决策的端到端时间账本

### 目标

在改动架构前，把 743 s 分解为可加总、可回归的时间账本，并确认早期与晚期网格状态的差异。

### 工作内容

- 保留当前 743 s、RMS=0 的版本为不可变基线，并立即提交一次 OJ，锁定当前确定得分。
- 在短运行中给以下阶段加范围标记：RHS、RK、KO dissipation、prolong、restrict、Sommerfeld、ghost exchange、analysis、I/O、device 同步和 host 调度。
- 采集至少三个代表窗口：早期、中期、晚期演化；不能只 profile 网格最小的前两步。
- 对每个热点同时记录：调用次数、总 GPU 时间、平均时长、寄存器、spill load/store、DRAM/L2 流量、指令数、eligible warps、主要 stall reason。
- 将 TwoPuncture、输出整理和其他固定开销从主演化中单列。

### 通过条件

- 各阶段时间之和与端到端时间误差不超过 3%；
- 能回答"340 s 中每个模块最多允许多少秒"；
- 重新确认 RHS 与 prolongation 是否仍约占 65% 和 16.5%。

如果 profiler attach 受限，不把它当作停止条件：使用程序内 CUDA event、host monotonic clock、现有日志和可运行的轻量 trace 建账；完整 profiler 只用于少量代表 kernel。

## 4. Phase 1：把 AMR 工作改成批处理 GPU 执行模型

### 为什么这是第一项架构改造

单独消除 11,821 次 launch 只能省约 1%，但当前"每个 patch/变量各发一次"的组织方式还造成了三个更大的间接损失：

- 每个 kernel 的有效网格较小，难以稳定填满 MIG 实例；
- 相同元数据、边界判断和索引逻辑被重复执行；
- 后续无法跨 patch 做统一的数据布局、内外边界分流和 producer–consumer 融合。

所以此阶段的目的不是只省 launch，而是为 RHS 和 prolongation 提供足够大的、规则化的执行域。

### 设计

- 为每个 AMR level 构造紧凑的 patch descriptor：基址、尺寸、stride、ghost 宽度、变量属性、边界类型和插值元数据。
- 按"尺寸 + 操作类型 + 边界类型"对 patch 分桶；一次 kernel 处理一个桶，block 从 device work queue 领取 tile。
- descriptor 和不变插值系数常驻 GPU；每步只更新真正变化的字段。
- 把 interior tile 与 physical/AMR boundary tile 分成不同队列，使热路径无边界分支。
- 保留原 kernel 作为逐桶 fallback，便于逐阶段验证数学等价性。

### 验收门槛

- 仅改变执行组织、不改变数学表达式时，短运行应 bit-exact；
- kernel 数至少下降 5×，但更重要的是小 kernel 的平均有效工作量显著上升；
- 若端到端收益低于 5%，仍可保留该基础设施，但必须证明它能支撑 Phase 2/3；否则不继续扩展通用框架。

CUDA Graphs 只作为重复拓扑的提交层优化。官方文档说明它可以摊销重复工作流的提交开销，但它不会缩短 kernel 内部依赖链；参见 CUDA Graphs。因此 graph 不能替代上述数据流重构。

## 5. Phase 2：RHS 计算图重构——决定项目成败的主战场

### 目标

在代表性早/中/晚网格上，使 RHS 总时间达到 ≥2.5× 加速。如果 RHS 做不到约 2.2×，340 s 基本失去可达性，应尽早止损。

### 5.1 从方程依赖图，而不是函数边界，重新划分阶段

- 将 RHS 表达式转成显式 DAG：节点为原始场、一次/二次导数、Christoffel/Ricci 中间量、每个输出 RHS；边为真正的数据依赖。
- 每个中间量记录使用次数、最后一次使用点、重算代价和存储代价。
- 基于 DAG 自动或半自动生成 2–4 种合法调度：
  - 即时消费调度：导数生成后立刻用于对应方程，缩短 live range；
  - 分组输出调度：按共享输入强度把输出方程分为 2–3 组；
  - 选择性重算调度：便宜的一阶表达式允许重算，昂贵的二阶导数保留；
  - 选择性物化调度：只把跨组共享且重算昂贵的少量中间量写入紧凑 scratch buffer。

这与已经失败的 noinline/maxrregcount 不同：后者只是让编译器把同一 live-set spill 到 local memory；这里是在数学依赖层面真正减少同时存活的量。

### 5.2 采用空间 tile + z 方向滚动窗口

- 一个 block 处理连续的 interior tile；连续线程沿内存连续方向映射，保证 warp 合并访问。
- 对半径固定的四阶 stencil，使用 z 方向滑动平面/窗口复用邻域；只缓存会被同一 block 多次消费的数据。
- shared memory 只服务于明确有跨线程复用的 x/y halo，不把整套场变量全部搬入 shared memory。
- 每次只流入当前方程组需要的字段，消费后立即释放，避免"为减少读取而把 50+ 变量全部常驻寄存器"。
- interior kernel 完全去除边界分支；稀少边界 tile 交给独立 kernel。
- NVIDIA 的最佳实践强调 shared memory 的价值在于消除真正的重复 global load 或重排非合并访问，而不是机械地缓存所有数据；参见 CUDA C++ Best Practices Guide。这也是本阶段与此前局部 shared-memory 试验的本质区别。

### 5.3 候选实现矩阵

只保留以下四个有明确机制差异的候选：

| 候选 | 方程分组 | 空间复用 | 预期风险 |
|---|---|---|---|
| R1 | 单 kernel，DAG 重排 | 无 | 仍可能 128 regs/spill |
| R2 | 2 组，少量 scratch | 无 | 增加一次中间写回 |
| R3 | 2 组，少量 scratch | tiled rolling window | 复杂度中等，主候选 |
| R4 | 3 组，选择性重算 | tiled rolling window | 指令增加，但 live-set 最低 |

不要再做没有新机制的 R5/R6 微调。每个候选先只实现最热 interior 路径，边界沿用 fallback，快速判断上限。

### RHS 通过条件

- RHS 汇总时间 ≥2.5×；
- spill bytes/point 至少下降 70%，而不是只看 registers/thread；
- 主要 stall 从 spill/long-scoreboard 转向 FP64 pipeline 或不可避免的 global load；
- 10 步正确性通过，100 步最终 RMS 和 constraint 通过；
- 端到端至少下降 270 s、争取约 290 s，才进入最终整合。

## 6. Phase 3：重写 prolongation 的并行算法形态

### 目标

把 prolong3 总时间降低至少 2×，预期贡献 55–70 s。

### 设计

- 先从现有实现抽取精确插值公式、权重与边界规则，建立逐点 CPU/GPU 单元测试。
- 将"一个细网格点内的长依赖链"展开成固定系数的独立乘加，并把可分离的插值维度按数学等价形式重新组织；是否采用分阶段 tensor-product 形式由实际额外流量决定。
- 所有相同形状 patch 跨 patch、跨变量批处理；block 处理一个或多个连续输出 tile。
- 将 interior、AMR 接口、physical boundary 三类工作完全分流；interior 使用无分支 fast path。
- 预计算并常驻索引映射和插值权重，避免每次 launch 重复整数除模、坐标判断和系数构造。
- 若单个输出存在可并行的多项求和，使用展开后的独立 accumulator 增加 ILP，打断当前 fixed-latency dependency chain。

### 验收门槛

- 单点插值单元测试覆盖所有奇偶位置与边界类型；
- 10 步结果至少与当前版本达到 checker 等价；
- prolong3 总时间 ≥2×，端到端收益 ≥50 s；
- 如果只能降低 launch 次数、内核本体不变且收益 <10 s，则停止该实现。

## 7. Phase 4：压缩剩余 80 秒预算

只有 Phase 2 和 Phase 3 达标后，才投入以下工作。

### 7.1 融合内存往返，而不是盲目融合 kernel

优先寻找具有相同迭代域、明确 producer–consumer 关系的组合：

- RHS 输出与紧随其后的 RK update；
- RK update 与简单逐点约束/过滤操作；
- ghost pack/unpack 与边界写回；
- 多变量的同构更新操作。

融合的唯一硬指标是减少 global bytes/step；如果融合导致 live-set、spill 或串行依赖重新上升，则回退为 batching。

### 7.2 消除全局同步与 host 往返

- 逐个审计同步点，区分真实跨阶段依赖与仅用于错误检查/计时的同步；
- 对独立 AMR level、analysis 或数据搬运建立 stream/event 依赖；
- host staging 仅在 trace 显示其占比显著时处理；单 rank 情况下不要预设 CUDA-aware MPI 一定有收益；
- 将频繁 allocation 改为预分配或 stream-ordered 重用。

### 7.3 端到端固定开销

TwoPuncture 会计入正式时间，因此必须测量但不应凭感觉优化。若它超过总目标的 5%，再单独 profile 并优化；否则保持稳定，避免扩大风险。分析与 I/O 只允许重叠或减少冗余搬运，不允许省略必需输出。

### 验收门槛

- "其他"部分从约 137 s 降到 ≤80 s；
- GPU 时间线没有可见的非依赖空洞；
- 新增 stream/graph 后连续两次运行和同一容器二次运行均成功。

## 8. Phase 5：有限度使用数值容限

当前 RMS=0 说明实现保留了远高于评分所需的数值一致性，但这不代表可以直接把主计算降成 FP32。已有试验已表明 RHS live-set 的 FP32 方案会使 RMS 超限，因此应明确禁止：

- 演化状态、Ricci/BSSN 主路径整体 FP32；
- 改变 stencil 阶数；
- 减少 RK 阶段；
- 缩短演化或跳过边界/输出。

只允许按以下顺序探索：

1. 数学等价的运算重排与 FMA 组织；
2. 只读、静态、低敏感度系数或索引元数据的低宽度存储，FP64 计算；
3. prolongation 中局部权重/中间量的混合精度，FP64 累加；
4. 对单个变量组做敏感度扫描，而不是全局开启 fast-math。

每个改动都必须先做 10 步误差趋势，再做 100 步完整验证；不得用短运行通过替代最终正确性。

## 9. 实验和集成纪律

### 三级验证

| 级别 | 用途 | 必做检查 |
|---|---|---|
| L0：1–2 步 | 编译、崩溃、kernel 微基准 | 输出存在、无 CUDA error、局部数值差 |
| L1：10 步 | A/B 性能筛选 | 同节点多次计时、checker、分阶段时间 |
| L2：100 步 | 最终候选 | 完整端到端时间、RMS、四个 constraint、容器二次运行 |

### A/B 规则

- 同一计算资源、同一输入、同一编译环境比较；
- 短测至少交替运行 OFF/ON，避免把网格阶段或节点抖动当收益；
- 记录 median、最小值和离散程度，不只保留最快一次；
- 每个候选只引入一个机制变化；通过后再叠加；
- 任何架构候选若在 L1 端到端回退超过 3%，立即回滚。

### 每轮必须记录的 scorecard

| 字段 | 内容 |
|---|---|
| Hypothesis | 预计减少哪一类时间或流量 |
| Scope | RHS / prolong / scheduler / fixed cost |
| Kernel metrics | time、calls、regs、spill、DRAM/L2、stall |
| End-to-end | OFF/ON 与加速比 |
| Correctness | RMS、Ham、Px/Py/Pz、是否 bit-exact |
| Decision | keep / revise / kill |

## 10. 建议实施顺序与里程碑

- Milestone A：两天内完成时间账本：锁定 743 s 基线并重新 OJ 提交；校准 RHS/prolong/other/fixed 四项预算；完成 patch descriptor 与分桶设计。
- Milestone B：优先证明 RHS 上限：先实现 R2 与 R3 的最热 interior prototype；代表步骤 RHS 未达到 2× 时，不继续做边界和通用化；最佳候选达到 2.5× 后，再接入完整边界路径并跑 10 步。
- Milestone C：prolongation 达到 2×：建立插值单元测试；批处理 interior fast path；再加入边界队列与 fallback。
- Milestone D：全局数据流整合：接入 batch scheduler；按 bytes/step 选择 RK fusion；用 streams/graphs 清除剩余可见空洞。
- Milestone E：340 秒冲线：连续完成至少两次 100 步运行；每次均 <340 s，而不是只取单次偶然最快值；RMS ≤0.1%，Grid Level 0 的 Ham/Px/Py/Pz 全部 ≤2；同一容器第二次运行正常；生成干净提交包并在 OJ 复核。

## 11. 明确停止重复的方向

以下方向已有充分反证，不应再次作为主线：

- `__noinline__`、maxrregcount 或 launch bounds 强压寄存器；
- 把普通乘加机械替换为 CUDA intrinsic；
- 仅靠 -O3、-fmad 或通用编译 flag；
- 对现有 RHS 做局部 LDG/shared-memory 包装而不改变跨方程数据流；
- 只减少 prolong launch、不打断内核依赖链；
- 期待 CUDA Graphs 单独贡献几十个百分点；
- 整体 FP32 化 RHS。

这些优化可以作为新架构中的次级参数重新测量，但不能再独立立项。

## 12. Go / No-Go 决策

340 s 是一个 组合目标，不是对单项优化的乐观外推。项目继续冲线必须满足：

| 检查点 | Go 条件 | No-Go 含义 |
|---|---|---|
| RHS prototype | 代表步骤 ≥2.0×，完整目标可见 ≥2.5× | 若 <1.7×，当前 340 s 路线基本不可达 |
| Full RHS | 端到端至少节省 270 s | 否则其余模块没有足够 Amdahl 空间 |
| Prolong | 端到端至少节省 50 s | 不再投入复杂边界重构 |
| 组合 10 步 | 按阶段外推 ≤360 s | 才值得做完整 100 步冲刺 |
| 最终 | 两次完整运行均 <340 s 且正确 | 达成目标 |

如果 Full RHS 只能达到约 2×，则应诚实地把最终预期调整到约 380–430 s，并优先交付稳定版本；不要通过违反题目约束或不可解释的数值近似强行冲线。
