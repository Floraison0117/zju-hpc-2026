#import "@preview/cuti:0.2.1": show-cn-fakebold

#show: show-cn-fakebold
#set text(font: ("Palatino Linotype", "KaiTi"))
#set math.equation(numbering: "(1)")
#set page(numbering: "1")
#set heading(numbering: "1.1")
#show enum: it => {
  set block(spacing: 0.5em)
  pad(left: 2em, it)
}
#show table: it => align(center, it)

#let centertitle(body) = [
  #set text(size: 24pt, weight: "bold")
  #align(center)[
    #v(1em)
    #body
    #v(0.5em)
  ]
]

#let codeblock(body) = block(
  fill: rgb("#f6f8fa"),
  inset: 8pt,
  radius: 2pt,
  width: 100%,
)[#body]

#centertitle[HPC Lab3.5 Report]

#set par(first-line-indent: (amount: 2em, all: true), spacing: 1em, leading: 0.8em)

#show outline.entry.where(level: 1): it => {
  v(1.2em, weak: true)
  strong(it)
}

#outline(
  title: none,
  indent: 1.5em,
)
#pagebreak()

= 实验目标
#v(0.5em)

本实验在华为昇腾 910B4 NPU 上实现并优化融合算子 FusedAddRmsNorm。官方评测 `Task Duration = 6.02 us`，
得分 #strong[76/120]，相对 baseline 加速 #strong[2.49 倍]。

= 算子语义与硬件背景
#v(0.5em)

== FusedAddRmsNorm
#v(0.5em)

设 $x$、`residual` 为 $B times H$ 的 FP16 张量，$w$ 为 $H$ 维 FP16 权重。对每一行
$b$，算子计算

$ R_b = x_b + "residual"_b, quad "rms"_b = sqrt(1/H sum_(i=0)^(H-1) R_(b,i)^2 + epsilon), $
$ y_b = R_b / "rms"_b dot.op w, quad "residual_out"_b = R_b $

其中 $epsilon = 10^(-6)$。算子同时输出 `y` 与 `residual_out` 两个 $B times H$ 张量。
正确性判据为逐元素 $|o_i - g_i| <= 10^(-3)$ 或相对误差 $<= 10^(-3)$。

== 昇腾 910B 达芬奇架构要点
#v(0.5em)

910B 系列（A2）的 AI Core 采用 AIC/AIV 分离设计：AIC 负责矩阵类运算，AIV 负责向量
与元素级运算。本算子不含矩阵乘，全部工作在 AIV 上。每个 AIV 核关键资源如下：

#figure(
  table(
    columns: (auto, auto, auto),
    align: left + horizon,
    stroke: none,
    table.hline(stroke: 1pt),
    table.header([资源], [规格], [用途]),
    table.hline(stroke: 0.5pt),
    [UB（Unified Buffer）], [192 KiB / AIV], [向量指令操作数唯一存放处],
    [Vector 单元], [VLEN 256 B], [Cast/Add/Mul/Reduce 等逐元素指令],
    [MTE2], [-], [GM → UB 数据搬入],
    [MTE3], [-], [UB → GM 数据搬出],
    [Scalar 单元], [-], [标量运算、地址计算、控制流],
    table.hline(stroke: 1pt),
  ),
  caption: [910B4 AIV 核关键资源],
)

MTE2、V、MTE3 三条流水线相互独立，可以并行；同步通过 `TQue` 队列语义自动插入的
事件或显式 `SetFlag/WaitFlag` 完成。Vector 指令必须操作 UB 上的数据，FP32 按
256 B/次处理 64 个元素，FP16 处理 128 个元素，32 B 对齐是向量指令与搬运的基本粒度。

选择 #strong[Ascend C] 开发路径。

= 性能基线
#v(0.5em)

课程提供的 Ascend C 基线（下称 V0）按行处理：`inQueX/inQueRes`（FP16，双缓冲）、
`outQueY/outQueResOut` 四个 TQue，以及 `weightHalf/weightFp32/resoFp32/sq/scalar`
等 TBuf。每行数据流为：DataCopyPad 搬入 x 与 residual → Cast 到 FP32 并 Add 得
$R$（同时 Cast FP16 写 `residual_out`）→ 平方与 Block/WholeReduceSum 规约 →
`GetValue` 取标量后 Duplicate + Sqrt + Div + Mul(weight) → Cast FP16 搬出。同步方面，
CopyIn/CopyOut 周围有大量 `PipeBarrier<PIPE_ALL>`，使 MTE2/V/MTE3 跨行完全串行。

基线正确性 5/5 通过。case 2 性能 5 次顺序测量：

#codeblock[
```text
15.1012, 14.7812, 15.0412, 15.2812, 15.1200 us
```
]

baseline 合计 8 个样本（含同条件复测 3 次）中位 #strong[15.01 us]（范围 14.58-15.28）。
baseline 的 msprof（`aic-metrics=Default`）显示：Task 14.98 us，Block Dim 40，
核 0-35 各 7 行、核 36 仅 4 行、核 37-39 空转；流水占比中位 vec 19-23%（2.72 us）、
scalar 37-51%（4.7-7.1 us）、MTE2 35-42%（4.2-5.5 us）、MTE3 14-16%（1.9 us）；
L2 命中率 6.6%，GM 到 UB 带宽利用率约 2%。没有任何流水线被占满，说明瓶颈是串行化、同步与负载不均，而非计算或带宽。

= 优化迭代
#v(0.5em)

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: left + horizon,
    stroke: none,
    table.hline(stroke: 1pt),
    table.header([版本], [主要修改], [5 case], [case2 中位 / us], [相对基线]),
    table.hline(stroke: 0.5pt),
    [V0 baseline], [按行 TQue + PIPE_ALL], [5/5], [15.01], [1.00×],
    [V2], [去除 CopyIn/Out 的 PIPE_ALL，TQue 双缓冲生效], [5/5], [9.92], [1.52×],
    [V4], [批量 chunk 路径：chunk 级拷贝、R 留 UB、一次 V_S 同步], [5/5], [7.62], [1.98×],
    [V5], [rstd 全标量，省每行 3 个微型向量操作], [5/5], [7.60], [1.97×],
    [V9], [批量路径去 TQue，raw TBuf + 手动事件，weight 移出关键路径], [5/5], [6.94], [2.16×],
    [V13 final], [tiling: blockDim=32 + rowsPerChunk=8；chunk 宽 phase A/B], [5/5], [6.28], [2.39×],
    table.hline(stroke: 1pt),
  ),
  caption: [优化版本汇总（case 2，`checker/profile.sh` 中位数）],
)

== 消除 PIPE_ALL 串行化（V2）
#v(0.5em)

#strong[瓶颈]：baseline 的 msprof 显示 MTE2/V/MTE3 均远未占满（利用率 19-51%），
scalar 与同步开销反而最高，说明 `PipeBarrier<PIPE_ALL>` 把三条流水线完全串行化。

#strong[修改]：去掉 CopyIn/CopyOut 周围的 `PIPE_ALL` 与冗余 `V_MTE3` flag，只依赖
TQue `EnQue/DeQue` 自动插入的 MTE2→V 与 V→MTE3 同步事件，并配合编译器
`--cce-auto-sync` 处理 V 内部 RAW 依赖。

#strong[收益]：中位 15.01 → 9.92 us（-34%）。`TQue` 的队列事件提供了正确的跨流水同步，
而 `PIPE_ALL` 是多余的全局屏障。

== 批量 chunk 路径（V4）
#v(0.5em)

#strong[瓶颈]：simulator 显示单行（H=4096）约 852 条 SCALAR 指令，其中约 85% 是队列
API、地址与掩码 setup；逐行的队列调用是 scalar 主要开销，且逐行小拷贝无法利用
MTE2 带宽。

#strong[修改]：对齐且 $H <= 4096$ 时走批量路径：一个 chunk 的多行用一条多块 `DataCopy`
搬入（行在 GM 中连续，`blockCount=n, blockLen=H/16`），$R$ 保留在 UB，逐行规约写入 `sumSqArr[row*8]`（32 B 对齐 lane），整个 chunk 只做一次 V→S 同步；输出
`residual_out`、`y` 各用一条多块 DataCopy 写回。

#strong[收益]：中位 9.92 → 7.62 us（-21%）。msprof 中 MTE3 从 1.7 us 降到 0.24 us
（chunk 级大拷贝），scalar 从 4.2-4.7 us 降到 2.6 us。

== 标量 rstd（V5）
#v(0.5em)

#strong[瓶颈]：原实现对每行做 Duplicate + Sqrt + Div 三个微型向量操作，rstd 的计算串在
V 链上。

#strong[修改]：用标量内建 sqrt：`rstd = 1/sqrt(sumSq*invH + eps)`，每行一次标量运算，
随后 `Muls(R, rstd)` 与 `Mul(R, weight)` 两个向量操作即可，rstd 完全移出 V 链。

#strong[收益]：中位 7.62 → 7.04 us（-8%）。精度方面，标量 `1/sqrt` 与 golden 的
`R / sqrt(...)` 在 FP32 下相差约 1-2 ULP，最终误差仍由 FP16 舍入边界翻转主导。

== raw TBuf 替换队列簿记，weight 移出关键路径（V9）
#v(0.5em)

#strong[瓶颈]：V5 的批量路径仍用 TQue 管理输入输出，每 chunk 的 Alloc/EnQue/DeQue/Free 与基线 weight 加载处的 `PipeBarrier<PIPE_ALL>` 仍在消耗标量周期；且 2 KB 的
weight 拷贝被 PIPE_ALL 放到 kernel 启动的关键路径上。

#strong[修改]：批量路径改用 raw `TBuf` + 显式 `SetFlag/WaitFlag`（每类事件用高位硬编码
ID，如 `EVENT_ID5`，避免与框架事件池冲突）。weight 拷贝首条发出、紧邻
Set/Wait(MTE2_V) 并 Cast，与 chunk 数据拷贝互不阻塞。

#strong[收益]：中位 7.60 → 6.94 us。同时修复了原代码中 `sumSqBuf` 只按 8 个 float 分配、
而每行按 32 B 步长写 `sumSqArr[row*8]` 导致的 UB 越界隐患。

== Tiling 与 chunk 宽向量化（V13）
#v(0.5em)

#strong[瓶颈]：V9 在 blockDim=40 下，256 行被切成 16 个核各 7 行、24 个核各 6 行，
7 行核成为拖尾（profile 中 aiv 最大值明显高于中位）；且 phase A/B 仍逐行发指令
（每行 5 个元素级操作 + 3 个规约操作）。

#strong[修改]（host 与 kernel 协同）：
#v(0.5em)
1. blockDim 设为 $min("AIV", B, 32)$。256 行 = 32 核 × 8 行，负载完全均衡，
   同时减少 kernel 分派开销（课程文档的空 kernel 图显示分派成本随核数线性增长）；
2. `rowsPerChunk` 预算从 144 KiB 提到 160 KiB，H=1024 时 chunk=8，每核恰好 1 个
   chunk，避免跨 chunk 同步；
3. phase A 改为 chunk 宽：行在 UB 中连续，`Cast/Cast/Add/Cast/Mul` 由每行 5 次
   变成每 chunk 5 次（8 行合并）；phase B 的 y 输出 Cast 也合并为 chunk 宽一次。
#v(0.5em)
#strong[收益]：中位 6.94 → 6.28 us（官方评测 6.02 us，76 分）。V13 的 msprof：
Task 6.54 us，aiv 中位 5.00 us（最大 5.89），vec 1.52 us，scalar 2.33 us，
MTE2 1.51 us，MTE3 约 0.26 us。

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: left + horizon,
    stroke: none,
    table.hline(stroke: 1pt),
    table.header([版本], [Task / us], [aiv / us], [vec / us], [scalar / us], [MTE2 / us], [MTE3 / us]),
    table.hline(stroke: 0.5pt),
    [V0 baseline], [14.98], [12-14], [2.7], [4.7-7.1], [4.2-5.5], [1.9],
    [V5], [7.42], [5.22], [1.56], [2.62], [1.81], [0.24],
    [V13 final], [6.54], [5.00], [1.52], [2.33], [1.51], [0.26],
    table.hline(stroke: 1pt),
  ),
  caption: [各版本 msprof 中位流水时间],
)

== 最终实现的数据流（V13）
#v(0.5em)

对齐批量路径（$H % 16 == 0$ 且 $H <= 4096$）的最终数据流：

#codeblock[
```text
GM x/residual →(2 条多块 DataCopy, blockCount=n, blockLen=H/16) → inBuf(F16, x|res)
  → Cast x32 | Cast res32（存入 rFp32 备用半区）→ Add → R32（覆盖 x32 区）
  → Cast → residual_out（F16 chunk 缓冲）
  → Mul sq = R^2 → per-row Block/WholeReduceSum → sumSqArr[row*8]
  → 1 次 V→S 同步 → per-row 标量 rstd = 1/sqrt(sumSq*invH+eps)
  → per-row Muls(R, rstd); Mul(R, weight) → chunk 宽 Cast → y
  → residual_out、y 各 1 条多块 DataCopy 写回 GM
```
]

同步设计：weight 拷贝首条发出 + 紧邻 Set/Wait(MTE2_V) + Cast（不在 phase A 关键
路径上）；chunk 数据拷贝后 Set/Wait(MTE2_V)；规约后一次 V→S；输出前紧邻
Set/Wait(V_MTE3)；多 chunk 时 chunk 边界 Set/Wait(MTE3_V) 与 V_MTE2 保护单缓冲
复用。精度保持全程 FP32 + 标量 rstd，最大相对误差由最终 FP16 舍入边界翻转决定，
理论有界 9.77e-4，恒小于 1e-3。

= 尝试过但未采用的方案
#v(0.5em)

+ 向量 `Rsqrt`（V7）：910B 向量 Rsqrt 精度约 0.2-0.5%（error_ratio 0.19-0.47），
  5 个公开 case 中 4 个 FAIL，因精度拒绝。
+ 更小的 chunk（V6，rowsPerChunk=3）：更多 chunk 的每-chunk 同步/API 开销大于
  MTE2/V 重叠收益（7.80 vs 7.04 us），回退。
+ 软件流水 + y 预乘 weight（V8）：7.46-7.58 us，比 V5 慢，回退。
+ `residual_out` 的 MTE3 提前到 phase A 后（E-C）：噪声级，未采纳。
+ rstd 循环拆分（V16，先一次性 GetValue 全部 8 个标量再批量发向量指令）：
  6.86 us，比 V13 慢，疑似寄存器/栈副作用，回退。
+ TQue 双缓冲管线（V17）：prologue 发首 tile、循环内 phase A 后立即发下一 tile
  以重叠 MTE2 与 phase B，5 case 全过但 6.50 us，队列簿记开销大于重叠收益，回退。
+ 多 tile 手动管线（V10/V14/V15/V18/V19）：目标是把 MTE2 藏到 V 下面以逼近
  5.08 us（90 分），但多 tile 结构在 CANN 8.5.0 下反复出现 vector core timeout。
  诊断：事件 flag 为"一次 Set 只能被一次 Wait 消费"的语义，且 `FetchEventID`
  与 TQue 内部事件共享从 0 开始的 ID 池，手动事件与队列事件发生 FLAGID 冲突
  （simulator 指令流显示内核末尾存在一个等不到 MTE2→V flag 0 的 V 等待）。
  由于无法静态验证编译器 auto-sync 与手动事件的全部交互，最终放弃多 tile 管线，
  采用稳定的单 chunk 结构。

= 最终结果
#v(0.5em)

== 正确性
#v(0.5em)

最终版本通过全部 5 个公开 case（两次独立运行），并独立于 checker 另测 22 个隐藏
shape 防护 case（含 $7 times 5003$ 一类非对齐大 H、$H>4096$ 两遍流式路径、
$B=1$ 单核、极小 H=3/8、$[-1000, 1000]$ 大范围数据），全部 PASS。官方评测
（5 公开 + 5 隐藏 case）正确性通过。`residual_out` 在所有测试中均为 0 误差，
`y` 最大相对误差 9.7e-4，低于 1e-3 容限。

== 性能
#v(0.5em)

最终版本 case 2 的 5 个新样本：5.90, 6.28, 6.60, 6.46, 6.10 us，中位 #strong[6.28 us]，
范围 5.90-6.60。官方 OJ 评测 `Task Duration = 6.02 us`，得分 #strong[76/120]。

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: left + horizon,
    stroke: none,
    table.hline(stroke: 1pt),
    table.header([指标], [基线 V0], [最终 V13], [说明]),
    table.hline(stroke: 0.5pt),
    [正确性], [5/5], [5/5 + 22 隐藏 PASS], [官方评测通过],
    [case2 中位 / us], [15.01], [6.28], [本机 `checker/profile.sh`],
    [官方 Task / us], [约 15], [6.02], [OJ 评测 `513c599c-r7`],
    [官方得分], [0], [76/120], [课程评分曲线],
    table.hline(stroke: 1pt),
  ),
  caption: [最终结果汇总],
)

= 思考题
#v(0.5em)

== GM 读写量、FLOP 与算术强度
#v(0.5em)

$[256, 1024]$、FP16 下：读 x、residual 各 512 KiB，写 y、residual_out 各 512 KiB，
合计 2 MiB GM 流量。每元素主要运算：残差加 1 次 Add、平方 1 次 Mul、规约约
1 次 Add、rstd 缩放 1 次 Muls、权重 1 次 Mul，约 5 FLOP，总计约 1.31 MFLOP。
算术强度约 $0.63$ FLOP/B，是典型的访存类算子。

但 msprof 显示 GM 到 UB 带宽利用率仅约 2%（baseline）到 20% 量级（最终版），Task 6.02 us 对应等效带宽约 2 MiB/6.02 us ≈ 350 GB/s，远低于 910B4 的 HBM 带宽。因此本实现既不是计算瓶颈
也不是 HBM 带宽瓶颈，而是指令发射与同步延迟受限，MTE2 与 V 的串行链是主要耗时。

== Double Buffer 流水
#v(0.5em)

最终版（V13）在评测 shape 下每核 8 行、恰好 1 个 chunk，不存在跨 chunk 的
双缓冲；其加速来自消除逐行队列 API 与同步开销。早期版本（V4/V5）使用
`TQue` 的 `BUFFER_NUM=2` 在 chunk 粒度做双缓冲，并用 `msprof op simulator --soc-version=Ascend910B4` 验证：560×1024（每核 14 行、
2 chunks）的 core0 指令流显示 `MOV_OUT_TO_UB`（chunk2 输入，1047 cyc）与
`MOV_UB_TO_OUT`（chunk1 输出，1047+685 cyc）并发执行，即 chunk 级 MTE2/MTE3
确已重叠。

== TQue 是否真实队列
#v(0.5em)

`TQue` 不是硬件队列，而是编译期/运行期的"所有权与同步簿记"抽象。`EnQue` 把
LocalTensor 在 UB 中的地址与一个事件 ID 登记进 TPipe 的队列记录，并在对应流水线
上发出 `SetFlag`；`DeQue` 取出同一地址的句柄并发出对应的 `WaitFlag`。数据本身
始终留在 UB 中，`EnQue/DeQue` 之间不发生任何拷贝或搬移。
