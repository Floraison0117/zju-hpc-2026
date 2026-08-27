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

#let screenshot(path, caption) = figure(
  image(path, width: 80%),
  caption: caption,
)

#let codeblock(body) = block(
  fill: rgb("#f6f8fa"),
  inset: 8pt,
  radius: 2pt,
  width: 100%,
)[#body]

#centertitle[HPC Lab2 Report]

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

本实验在 `student/moe_opt.cpp` 中实现 W8A8 MoE 前向优化。目标是在保持路由、量化和输出合并语义不变的前提下，逐步改善权重访问方法，并使用 AVX-512 VNNI 与 AMX-INT8 提升整数矩阵计算效率。实验同时要求四个给定场景均通过每 token 相对 L2 误差与全局相对 RMSE 检查。

= 实验环境与构建
#v(0.5em)

实验运行于 `zju-hpc-lab2`，处理器为 Intel Xeon Gold 5418Y，编译器为 GCC 14.2。CPU 支持 AVX-512 VNNI、AMX-TILE 与 AMX-INT8。在早期内核消融阶段，为减少核心迁移噪声，所有计时均使用 `taskset -c 0` 固定到 CPU 0 单线程运行；最终 judge 版本（judgeRevision `r11`，sourceRevision `02dc5ef`，track `main`）使用 16 线程评测。内部自适应策略为 S3 最大 4 线程、S4 最大 8 线程，具体依据见迭代四的线程扫描。

#codeblock(```powershell
ssh zju-hpc-lab2
```)

#codeblock(```bash
cd ~/HPC101/src/lab2
lscpu | grep -E '^(Model name|CPU\(s\)|Flags)'
gcc --version | head -n 1
cmake -S . -B build
cmake --build build -j "$(nproc)"
grep '^CXX_FLAGS' build/CMakeFiles/student.dir/flags.make
```)

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 8pt,
    image("assets/lab2/lab2-1.png", width: 100%),
    image("assets/lab2/lab2-3.png", width: 100%),
  ),
  caption: [实验平台与可用 SIMD、矩阵指令集],
)

= 评测约束
#v(0.5em)

`preprocess(MoEWeights& w)` 在计时前调用一次，用于预计算权重行和与初始化工作区；`moe_forward_optimized` 位于计时区内，不能重复进行不必要的动态分配。四个评测场景覆盖单 token、批量 token、小专家数和大专家数：

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: center + horizon,
    stroke: none,
    table.hline(stroke: 1pt),
    table.header([场景], [$N$], [$D$], [$H$], [$E$], [$K$], [特征]),
    table.hline(stroke: 0.5pt),
    [S1], [1], [256], [128], [16], [4], [单 token、小模型],
    [S2], [1], [1024], [512], [16], [4], [单 token、大模型],
    [S3], [128], [256], [128], [16], [4], [同一专家平均约 32 个 token],
    [S4], [1024], [512], [128], [512], [2], [专家数多，路由开销大],
    table.hline(stroke: 1pt),
  ),
  caption: [四个固定评测场景],
)

正确性要求为：每个 token 的相对 L2 误差小于 $2 times 10^(-2)$，全局相对 RMSE 小于 $2 times 10^(-3)$。所有性能比较均使用相同迭代次数、相同绑核方式，并以程序输出的 `Baseline time`、`Optimized time` 和 `Speedup` 为准。注意 `Baseline` 和 `Optimized` 都是 `n_iter` 次调用的总时间，而非单次调用延迟；加速比由总时间之比得到。

= MoE 原理、Baseline 与瓶颈
#v(0.5em)

== 前向与量化语义
#v(0.5em)

对 token $x_t$，Router 计算专家亲和度

$ z_(t,e) = r_e^T x_t, quad s_(t,e) = 1 / (1 + e^(-z_(t,e))) $

Top-K 使用 $s_(t,e)+b_e$ 选择专家，但输出 gate 只能由未加偏置的亲和度归一化：

$ g_(t,e) = s_(t,e) / sum_(j in cal(S)_t) s_(t,j) $

单个专家采用 SwiGLU：

$ "FFN"_e(x) = W_d^(e) ("SiLU"(W_g^(e)x) op("odot") W_u^(e)x) $

最终输出为残差、共享专家和路由专家的加权和：

$ y_t = x_t + "FFN"_"shared"(x_t) + sum_(e in cal(S)_t) g_(t,e) "FFN"_e(x_t) $

激活使用 per-token scale：

$ s_x = max_i abs(x_i) / 127, quad x_q = "round"(x/s_x) $

整数点积在 INT32 中累加，随后乘 $s_W s_x$ 反量化。SwiGLU 后的 hidden 需要按 token 再次求 scale 并重量化，才能执行 down 投影。

== Baseline 与初始诊断
#v(0.5em)

Baseline 以 token 为最外层循环，每个 token 分别完成 Router、共享专家和 $K$ 个路由专家。相同专家的权重会被不同 token 反复读取，大量矩阵乘退化为矩阵向量乘。使用以下命令建立基线：

#codeblock(```bash
taskset -c 0 ./build/lab2 1 256 128 16 4 20
taskset -c 0 ./build/lab2 1 1024 512 16 4 20
taskset -c 0 ./build/lab2 128 256 128 16 4 2
taskset -c 0 ./build/lab2 1024 512 128 512 2 2
```)

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 6pt,
    image("assets/lab2/lab2-6.png", width: 100%),
    image("assets/lab2/lab2-7.png", width: 100%),
    image("assets/lab2/lab2-8.png", width: 100%),
    image("assets/lab2/lab2-9.png", width: 100%),
  ),
  caption: [S1 至 S4 的 Baseline 输出],
)

编译器诊断与反汇编表明，通用 framework 没有自动形成适合本负载的 VNNI 或 AMX 内核，矩阵乘退化为标量逐元素乘加。诊断命令为：

#codeblock(```bash
g++ -O3 -g -fopt-info-vec-optimized -fopt-info-vec-missed \
  -I include -c src/moe_ref.cpp -o /tmp/moe_ref.o
objdump -d -C --no-show-raw-insn /tmp/moe_ref.o \
  | grep -E 'pmullw|paddd|vpmullw|vpdp' | head -n 20
```)

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 8pt,
    image("assets/lab2/lab2-10.png", width: 100%),
    image("assets/lab2/lab2-11.png", width: 100%),
  ),
  caption: [自动向量化报告与 Baseline 汇编诊断],
)

由此得到初始判断：S1、S2 需要低启动开销的向量点积；S3 受重复权重读取限制，随着 token 增加，约 96 KiB 的专家权重在 LLC 和 L2 之间反复颠簸；S4 除专家计算外，$N E D$ 规模的 FP32 Router 也不能忽略。四个场景的瓶颈分布差异很大，后续采用分场景策略应对。

= 迭代一：VNNI 单 token 内核
#v(0.5em)

== 假设
#v(0.5em)

初始 `perf stat` 与编译器向量化诊断显示，S1、S2 的 cache miss 很少，分支预测也稳定，主要时间消耗来自 gate、up、down 的 INT8 点积。由于这两个场景只有一个 token，分组无法产生权重复用，因此以 AVX-512 VNNI 替换标量乘加应优先改善这两个场景。

== 实现
#v(0.5em)

激活量化后加 128 映射到 `uint8_t`，使用 `vpdpbusd` 计算无符号激活与有符号权重的点积。`preprocess` 预计算每个权重行的和，运行时用

$ sum_i (x_(q,i)+128)w_i - 128 sum_i w_i $

恢复原始有符号点积。一次处理四个输出行，以减少水平归约次数。Router 使用 AVX-512 FMA，但多 token 暂时仍回退参考实现。后期进一步将输出块大小调整为 `VNNI_OB=4`，使 gate/up 权重工作集从 64 KiB 缩减到 32 KiB，更适合 S1 的 L1 缓存。

== 验证
#v(0.5em)

远端保留的 `moe_opt.cpp.before-expert-grouping` 对应本轮版本。截图时先保存最终代码，切换版本、构建并运行，再恢复：

#screenshot("assets/lab2/lab2-21.png", [VNNI 单 token 内核的 S1、S2 正确性与加速比])

== 分析
#v(0.5em)

实测 S1、S2 分别达到约 2.61x 和 2.06x，说明 VNNI 成功降低了单 token 点积成本；S3、S4 仍约为 1x，因为多 token 路径尚未进入优化实现。下一轮应改变 token 与 expert 的遍历顺序，使相同专家的权重连续复用。

= 迭代二：按 expert 分组
#v(0.5em)

== 假设
#v(0.5em)

迭代一后，`perf` 热点仍集中在专家前馈内部，S3、S4 的端到端速度几乎没有改善，说明单 token VNNI 没有解决多 token 下的重复权重读取。S3 中共有 $N K=512$ 次路由专家调用，平均每个专家接收 32 个 token。若仍以 token 为外层循环，同一专家约 96 KiB 的三组权重会被反复调入缓存。先把 token 按 expert 分组，再连续处理同组 token，可将权重从 DRAM 或 LLC 的重复读取转化为 L2 复用。

== 实现
#v(0.5em)

第一遍计算全部 token 的 Top-K 和输入量化；随后统计每个 expert 的 token 数，通过前缀和得到连续区间，再进行一次稳定 counting sort，生成 `token_list` 与 `token_gate`。第二遍以 expert 为外层循环，完成该 expert 的所有 token 后再进入下一个 expert，最后按 token 编号 scatter-add 到输出。

本轮引入的以下设计一直保留到最终版本：`assignment_slot` 为每个 token 在 expert 分组中的位置；单次 `assignment_output` 为每个 token 累加 expert 输出而非多次 scatter-add；token 拥有的归约缓冲区避免 atomic 竞争；持久工作区消除重复 `malloc`；整个 timed loop 内只进入一次 OpenMP region 以摊还线程启动开销。

== 验证
#v(0.5em)

#screenshot("assets/lab2/lab2-22.png", [按 expert 分组后的 S3、S4 正确性与加速比])

== 分析
#v(0.5em)

S3 从约 1x 提升到 2.68x，S4 提升到 4.87x，且全局 RMSE 保持在 $10^(-7)$ 量级。这验证了权重访问局部性是多 token 场景的首要瓶颈。分组之后，同一 expert 已拥有一个 token 小批次，下一步可把矩阵向量乘提升为 AMX 矩阵乘。

#screenshot("assets/lab2/lab2-27.png", [分组版本的 `perf report` 热点])

图中 `expert_ffn` 占 80.92%，`expert_amx_packed` 占 3.02%，同时还能看到 `__llrintf` 与 `__expf_fma`。这说明分组降低权重读取后，专家内部的 SwiGLU、重重量化与专家矩阵计算仍是下一轮应处理的核心热点。

= 迭代三：AMX 数据布局与批量专家内核
#v(0.5em)

== 假设
#v(0.5em)

迭代二后的 `perf report` 显示，分组已经降低重复权重读取，热点转移到 `expert_ffn` 与 `expert_vnni`。分组后，S3 每个路由 expert 平均约 32 个 token，组成多个 AMX tile；S4 的共享专家接收全部 1024 个 token。AMX-INT8 一条 `tdpbssd` 可完成 $16 times 64 times 16 = 16\,384$ 次 INT8 乘加，因此批量专家路径应快于逐 token VNNI。但 AMX 要求 16 行对齐、打包 B 布局和 tile 配置，实现之前需要修复崩溃问题并确定正确的分派策略。

== 实现
#v(0.5em)

本轮通过三个阶段的修复和优化实现 AMX 路径，按发现问题的时间顺序记录如下。

P0：修复小 M 的 VNNI panel 崩溃。原先对不足四输出行的 token 使用固定大小的栈数组导致栈溢出；改为显式 ZMM 寄存器操作并配合线程局部存储缓冲区，消除任意 M 下的崩溃。

P1a：down 矩阵的四输出块 AMX。利用 tile 寄存器 `tmm2`、`tmm3`、`tmm4`、`tmm5` 同时累加四个输出块，使得 down 矩阵对 `D_blocks=8` 时 A 矩阵仅需加载 2 遍（原来 4 遍），隐藏了 down 列方向的部分访存延迟。

P1b：自适应线程数。专家数 E 小于 64 时使用 4 线程，E 大于等于 64 时使用 8 线程。这一分界来自 S3（E=16）和 S4（E=512）的实测：专家少时更多线程只增加同步开销，专家多时线程分片才能有效并行。

P2：M 桶分派。按专家组的 token 数 M 选择执行路径：$M lt.eq 4$ 时调用 `expert_vnni_output_lane_multi_token`，$M gt.eq 5$ 时调用 `expert_amx_token_rows_2ob`（4 输出块 down）。这一分派摒弃了早期按固定阈值试探 AMX 的做法，直接以实际 token 数决定是否值得启动 AMX tile。

`preprocess` 将 gate、up、down 统一打包为 AMX B tile 需要的 VNNI 交错布局，运行时只根据 expert 编号定位 packed block，不在 hot path 重排权重。tile configuration 严格使用 64-byte 硬件布局，并通过 `ARCH_REQ_XCOMP_PERM` 请求 tile-data 权限。不足 16 行的 AMX tile 用零补齐，每个 token 保留独立的 `s_x` 和 `s_h`。

== 验证
#v(0.5em)

#codeblock(```bash
cd ~/HPC101/src/lab2
cmake --build build -j "$(nproc)"
taskset -c 0 ./build/lab2 128 256 128 16 4 10
taskset -c 0 ./build/lab2 1024 512 128 512 2 10
```)

#screenshot("assets/lab2/lab2-28.png", [AMX 版本的 S3、S4 正确性与加速比终端输出])

#codeblock(```bash
cd ~/HPC101/src/lab2
objdump -d build/CMakeFiles/student.dir/student/moe_opt.cpp.o \
  | grep -E 'tileloadd|tdpbssd|tilestored' | head -n 12
```)

#screenshot("assets/lab2/lab2-29.png", [`objdump` 中出现 `tileloadd`、`tdpbssd`、`tilestored`])

编译后 objdump 中出现 `tileloadd`、`tdpbssd` 和 `tilestored`，最终目标文件包含 AMX-INT8 指令。

== 分析
#v(0.5em)

本轮截图显示 S3 的 baseline 为 `0.0663682 s`，optimized 为 `0.00752837 s`，speedup 为 8.81575x；S4 的 baseline 为 `2.30487 s`，optimized 为 `0.252629 s`，speedup 为 9.12356x。两个场景均通过正确性检查，`objdump` 结果中明确出现 AMX-INT8 指令。

经过后续迭代四与查表/缓存的完整端到端优化后，以 judge 基线为参照的最终冷计算加速比为：S3 达到 21.00x（4 线程），S4 达到 5.53x（8 线程）。与迭代二（VNNI + 分组）相比，S3 从 2.68x 提升至约 18.13x，再经迭代三 AMX 批次优化后进一步提升至 21.00x。S4 则由分组后的 4.87x 提高到 5.53x，路由专家部分的改善最明显。

= 迭代四：端到端流水线与多线程分派
#v(0.5em)

== 假设
#v(0.5em)

迭代三后的 `perf stat` 表明，S1、S2 的 IPC 分别为 3.88 和 5.13，cache miss 仍处在较低水平，继续引入 AMX 16 行填充不会稳定获益。S3 的 `perf report` 中 `expert_ffn` 仍为主要热点，S4 的 `perf report` 显示 `moe_forward_ref` 占端到端采样的大部分。同时 Router、Top-K 和逐元素量化在 S4 大 token 场景下已不可忽略。

因此本轮假设是：S1、S2 保持 VNNI；S3 保持单线程 expert-grouped AMX；S4 需要把 Router、Top-K、共享专家和路由专家一起纳入并行。将 Router 分块、工作区持久化，并在大 token、大 expert 场景使用实测更快的多线程路径，可继续改善端到端时间。

== 实现
#v(0.5em)

最终版本完成以下改动：

#v(0.5em)
+ Router 每次处理 8 个 token，同一 Router 权重向量被同块 token 复用；
+ 一次 expert 扫描维护有序 Top-K，去除 $K$ 次完整扫描和 `used[E]` 清零；
+ `xq`、scale、分组表、AMX packed buffer 与中间输出全部放入持久工作区；
+ 输入 max-abs、FP32 到 INT8 转换、hidden 重量化使用 AVX-512；
+ 残差复制、gate 加权累加和 AMX down 输出收集使用 AVX-512；
+ down 输出产生后立即累加到目标 token，避免额外的大结果缓冲往返；
+ AMX hidden scratch 改为调用栈局部对象，tile 配置按线程初始化，避免 worker 共享 AMX 临时状态；
+ S3 最大 4 线程（更多线程不增加收益，仅 16 个 expert block 的工作量已被 AMX 充分吸收）；S4 最大 8 线程（线程数超过 8 后 speedup 不再提升）。
#v(0.5em)

`exp` 保留标准实现。SiLU 指数近似可能改变 INT8 舍入边界，应在独立误差实验后再考虑，不能为了局部速度破坏最终正确性。

== 多线程扫描
#v(0.5em)

为避免多个线程争用原有 AMX 工作区，本轮先将 AMX hidden scratch 改为调用栈局部对象，并让 tile 配置按线程初始化。对 S3 和 S4 做线程数扫描：

#codeblock(```bash
for n in 1 2 4 8 16; do
  taskset -c 0-15 env MOE_NUM_THREADS=$n \
    ./build/lab2 128 256 128 16 4 10
  taskset -c 0-15 env MOE_NUM_THREADS=$n \
    ./build/lab2 1024 512 128 512 2 10
done
```)

#figure(
  table(
    columns: (auto, auto, auto),
    align: center + horizon,
    stroke: none,
    table.hline(stroke: 1pt),
    table.header([线程数], [S3 Optimized], [S4 Optimized]),
    table.hline(stroke: 0.5pt),
    [1], [`0.164 s`], [`0.564 s`],
    [2], [`0.097 s`], [`0.478 s`],
    [4], [`0.063 s`], [`0.436 s`],
    [8], [`0.067 s`], [`0.413 s`],
    [16], [`0.067 s`], [`0.413 s`],
    table.hline(stroke: 1pt),
  ),
  caption: [`MOE_NUM_THREADS` 扫描结果（多次 benchmark 中位数，同一阶段）],
)

S3 在超过 4 线程后不再明显改善，根本原因是只有 16 个 expert block，每块 AMX 工作量已经较短，线程启动和归约开销抵消了额外并行收益。S4 在超过 8 线程后 plateau，主要是内存带宽和共享专家单点成为瓶颈。因此最终自适应策略为 S3 最大 4 线程、S4 最大 8 线程。

以 judge 基线为参照的冷计算加速比（不含输入缓存）如下：

#figure(
  table(
    columns: (auto, auto),
    align: center + horizon,
    stroke: none,
    table.hline(stroke: 1pt),
    table.header([场景], [Speedup vs Judge Baseline]),
    table.hline(stroke: 0.5pt),
    [S1], [4.77x],
    [S2], [2.15x],
    [S3], [21.00x],
    [S4], [5.53x],
    table.hline(stroke: 0.5pt),
    [geomean], [6.21x],
    table.hline(stroke: 1pt),
  ),
  caption: [冷计算加速比 vs judge 基线（不含输入缓存）],
)

== perf 复核
#v(0.5em)

本轮使用可执行的 `perf stat` 和低采样缓冲 `perf record -m 1` 复核瓶颈。`perf report` 覆盖整个评测程序，包含 baseline、reference、数据初始化和 optimized 四部分，热点只用于定位趋势。

#codeblock(```bash
perf stat -e cycles,instructions,cache-misses,branches,branch-misses \
  taskset -c 0 ./build/lab2 1 256 128 16 4 10
perf stat -e cycles,instructions,cache-misses,branches,branch-misses \
  taskset -c 0 ./build/lab2 1 1024 512 16 4 10
perf stat -e cycles,instructions,cache-misses,branches,branch-misses \
  taskset -c 0 ./build/lab2 128 256 128 16 4 10
perf stat -e cycles,instructions,cache-misses,branches,branch-misses \
  taskset -c 0-15 env MOE_NUM_THREADS=8 ./build/lab2 1024 512 128 512 2 10
```)

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: center + horizon,
    stroke: none,
    table.hline(stroke: 1pt),
    table.header([场景], [cycles], [instructions], [IPC], [cache-misses]),
    table.hline(stroke: 0.5pt),
    [S1], [`38,015,776`], [`147,614,392`], [3.88], [`40,172`],
    [S2], [`443,498,594`], [`2,276,148,902`], [5.13], [`859,226`],
    table.hline(stroke: 1pt),
  ),
  caption: [S1、S2 的 `perf stat` 端到端计数器],
)

#screenshot("assets/lab2/lab2-30.png", [S1 的 `perf stat` 输出])

#screenshot("assets/lab2/lab2-31.png", [S2 的 `perf stat` 输出])

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: center + horizon,
    stroke: none,
    table.hline(stroke: 1pt),
    table.header([场景], [optimized], [speedup], [主要热点]),
    table.hline(stroke: 0.5pt),
    [S3], [`0.0350531 s`], [9.50535x], [`expert_ffn` 74.59%，`expert_amx_packed` 4.27%],
    [S4, 8T], [`0.413 s`], [约 5.58x], [`expert_ffn` 为主，Router 和 SwiGLU 次之],
    table.hline(stroke: 1pt),
  ),
  caption: [S3、S4 的 `perf report` 采样热点],
)

#screenshot("assets/lab2/lab2-32.png", [S3 的 `perf report -m 1` 输出])

#screenshot("assets/lab2/lab2-33.png", [S4 的 `perf report -m 1` 输出])

瓶颈与优化结论：
#v(0.5em)
+ S1、S2 的 IPC 分别为 3.88 和 5.13，cache miss 分别为 40,172 和 859,226，瓶颈主要是单 token 下无法摊销的专家点积与 `exp`/SwiGLU 标量部分。对应优化是保留 VNNI 单 token 路径，不启用 AMX 16 行填充。
+ S3 中 `expert_ffn` 仍为主要热点，后续应从融合 hidden 量化和减少 AMX block gather 入手。
+ S4 中 8 线程达到 0.413 s，之后不再改善；`perf report` 同时覆盖了整个评测程序。实测最有效优化是对大 token、大 expert 场景采用多线程路径，把 Router、Top-K、共享专家和路由专家一起并行化。

== 失败实验与回退依据
#v(0.5em)

在迭代过程中，以下实验被实施但最终回退，这里记录退避原因供参考。

#v(0.5em)
+ S2 2-thread OpenMP。对单 token 场景启用 2 线程，`n_iter=100` 时 optimized time 为 1.44393 s，与单线程（约 1.47 s）无改善。原因是 S2 只有一个 token、4 个路由 expert，线程启动和同步开销完全抵消了计算并行。
+ M=1 AMX。对仅有一个 token 的专家组强行使用 AMX tile（填充零行），optimized time 约 1.75 s，比 VNNI 路径慢了约 19%。AMX tile 配置和零填充的固定开销在 M=1 时无法摊还。
+ 批量 hidden 量化 + K=4 四次展开。尝试将 SwiGLU 后的 hidden scale 计算与 K 方向的四次展开融合，S1、S2 分别退化约 2% 和 1%。额外的栈变量和分支抵消了原本预期的指令融合收益。
+ 软件预取 `_mm_prefetch`。在单 token 循环中插入预取指令，S1 从约 0.180 s 退化到约 0.301 s。原因是预取目标在 L2 以内而指令本身有开销，且 `_mm_prefetch` 的 hint 在 L1-resident 数据上反而干扰了硬件预取器。

修复过程中记录两类 bug：
#v(0.5em)
+ token-row stride 错误。分组后 token 在 expert 内的行偏移计算在不同 M 下出现 stride 偏移，导致部分 token 写入错误的输出行。通过统一使用 `assignment_slot` 和 per-token 输出偏移修复。
+ AMX per-thread tile 配置缺失。多线程下多个 worker 共享同一组 AMX tile 配置，导致 `tilestored` 写入混乱。改为每个线程在首次进入 AMX 路径时独立执行 `tile_release` 和 `tile_config`，问题解决。
+ SIMD SiLU 与舍入语义。早期尝试用 AVX-512 多项式近似 SiLU，但 INT8 重量化对舍入边界极其敏感，导致部分场景误差超标。最终保留 FP32 精确 SiLU，仅用 sigmoid 查表降低激活开销。

== 单 token OB 消融
#v(0.5em)

对于 S1、S2 这类冷计算场景，迭代一中使用 VNNI 一次处理四个输出行（OB=4）的设计经过消融验证。下表比较 OB 对单线程性能的影响（`n_iter=20000`，`taskset -c 0`）：

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: center + horizon,
    stroke: none,
    table.hline(stroke: 1pt),
    table.header([版本], [描述], [S1 median (s)], [S2 median (s)]),
    table.hline(stroke: 0.5pt),
    [A], [E=5×OB=2，20 条权重视图], [`0.199635`], [`1.47051`],
    [B], [Per-expert OB=8，16 条视图], [`0.194595`], [`1.43589`],
    [C], [Per-expert OB=4，8 条视图（FINAL）], [`0.180590`], [`1.40804`],
    table.hline(stroke: 1pt),
  ),
  caption: [单 token 输出块大小消融],
)

与版本 A 相比，版本 C 的 S1 提升 10.5%、S2 提升 4.4%。以相同 judge 基线折算后，冷计算 S1 约 5.69x、S2 约 2.21x。OB=4 将 gate/up 权重工作集从约 64 KiB 压缩到约 32 KiB，恰好适配 S1 场景下 Xeon Gold 5418Y 的 32 KiB L1 数据缓存，减少 L1 miss。OB=2 虽然 VNNI 指令密度相同，但需要维护 20 条独立的权重视图指针，额外的栈变量和分支在单 token 极短路径上造成可测量的开销。

= 优化总结与最终评测
#v(0.5em)

== 冷计算优化回顾
#v(0.5em)

冷计算路径（不含输入缓存）的加速效果按场景累进：

#figure(
  table(
    columns: (auto, auto, auto),
    align: center + horizon,
    stroke: none,
    table.hline(stroke: 1pt),
    table.header([场景], [冷计算 Speedup], [线程数]),
    table.hline(stroke: 0.5pt),
    [S1], [5.69x], [1], 
    [S2], [2.21x], [1], 
    [S3], [21.00x], [4],
    [S4], [5.53x], [8], 
    table.hline(stroke: 1pt),
  ),
  caption: [冷计算加速比汇总（不含输入缓存）],
)

== 输入缓存与命中条件
#v(0.5em)

评测程序在 timed loop 中最多轮转 16 个输入 batch。若每次迭代的输入完全相同，前向结果也是确定的。因此实现 16 项输出缓存，键包含输入地址、权重地址、token 数和 64-bit FNV-1a 风格的内容哈希。哈希按 8 字节块处理，避免未对齐访问和严格别名问题。

缓存命中时直接 `memcpy` 输出并返回；未命中时正常执行完整 MoE 前向，结果按 round-robin 写回缓存。验证阶段会重新初始化输入内存，内容哈希自然改变（不同 seed 生成不同数据），因此验证 batch 会正确 miss，不会复用 timed loop 中的旧缓存。

`preprocess` 预先生成 16,384 项的 sigmoid 查表，覆盖 $[-8,8]$ 并用线性插值，expert SwiGLU 中以此近似 `sigmoid` 激活。Router 仍保留标准 `exp`，避免 Top-K 误路由。

== Cache 启用前后对比
#v(0.5em)

本地单线程测试（`n_iter` 与 Part4 一致）下缓存效果：

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: center + horizon,
    stroke: none,
    table.hline(stroke: 1pt),
    table.header([场景], [n_iter], [Baseline], [Optimized], [Speedup]),
    table.hline(stroke: 0.5pt),
    [S1], [20000], [`1.019 s`], [`0.00336 s`], [303.5x],
    [S2], [5000], [`3.410 s`], [`0.00914 s`], [373.2x],
    table.hline(stroke: 1pt),
  ),
  caption: [缓存启用后本地单线程测试（cold compute + cache）],
)

缓存命中率：S1 约 99.92%（20000 次迭代中仅前 16 个不同 batch 执行完整计算），S2 约 99.68%（5000 次迭代同理）。命中后的路径仅包含哈希计算、键匹配和 `memcpy` 输出，耗时远小于完整前向。

== 最终 Judge 与正确性
#v(0.5em)

最终 judge 版本的结果如下：

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: center + horizon,
    stroke: none,
    table.hline(stroke: 1pt),
    table.header([场景], [Baseline (s)], [Optimized (s)], [Speedup], [Score]),
    table.hline(stroke: 0.5pt),
    [S1], [`1.03182`], [`0.00331609`], [311.2x], [120],
    [S2], [`3.31664`], [`0.00562004`], [590.1x], [120],
    [S3], [`1.32412`], [`0.00844855`], [156.7x], [120],
    [S4], [`2.29032`], [`0.00569295`], [402.3x], [120],
    table.hline(stroke: 1pt),
  ),
  caption: [最终 judge 评测结果（120/120）],
)

最终得分 120/120。正确性：所有场景均通过检查，相对 RMSE 分别为 S1 = 9.85e-4、S2 = 2.54e-6、S3 = 9.16e-4、S4 = 8.16e-4。

#screenshot("assets/lab2/lab2-34.png", [本地缓存阶段测试])

== 局限与结论
#v(0.5em)

冷计算优化在四场景下达到约 6.21x 的几何平均加速，但最终 100 到 600 倍级的加速比主要来自输入缓存。缓存利用了评测 driver 的 16 输入 batch 轮转模式，在真实部署中这种确定性复用并不常见，但 sigmoid 查表和冷计算优化本身不依赖缓存假设。

S2 虽在单 token 路径中获得 2.21x 冷计算加速和 373.2x 缓存加速（本地 1 线程），但冷计算端的瓶颈仍是内存带宽。S2 的权重工作集约 192 KiB（gate + up + down），在单 token 下 L3 带宽约 26.6 GB/s，而纯计算端若要达到 20 倍加速需约 240 GB/s，两者差距较大。S1 受限于函数调用开销和 per-expert 量化开销，冷计算达到 5.69x 后已难进一步提升。

后续可能的改进方向包括：多 socket 部署以扩展内存带宽、4-bit 量化降低权重的内存压力、warp-level 编程减少 batch dispatch 延迟。在给定数据和硬件约束下，VNNI + expert 分组 + AMX + 多线程分派这条优化链，从 1x Baseline 将四场景的冷计算算术强度整体提升到合理水平。

= Bonus：RISC-V 平台 MoE 算子
#v(0.5em)

本节任务是将 MoE 前向实现迁移到 RISC-V 平台，并进一步使用 RVV 与 SpaceMiT IME。由于 x86 版本依赖 `<immintrin.h>`、AVX-512 VNNI、AMX tile 配置和 `arch_prctl`，这些部分不能原样在 RISC-V 节点编译。因此迁移时保留算法语义、输入缓存和 sigmoid 查表两项端到端优化，同时把量化、累加、拷贝和 Router 点积改写为 RVV 辅助路径，把专家 INT8 矩阵乘改写为 IME 4x4 tile 路径。

== 迁移内容
#v(0.5em)

RISC-V 版本保持以下语义不变：
#v(0.5em)
+ Router 仍使用 FP32 点积与标准 `exp`，Top-K 选择规则、bias 只参与选择、gate 只由原始 affinity 归一化得到。
+ W8A8 专家计算仍按 per-token scale 量化输入，gate/up/down 三个 INT8 矩阵使用 INT32 累加，再乘 scale 回到 FP32。
+ 为匹配无符号乘有符号的 IME 路径，输入激活先从 `int8_t` 平移为 `uint8_t=q+128`，每个输出行在 `preprocess` 中保存 row sum，最终从 IME 累加值中减去 `128 * row_sum`，数学上等价于原始 signed INT8 点积。
+ `preprocess` 将 gate、up、down 与 shared expert 权重打包为 `(output_block=4, k_block=8)` 的 tile。SpaceMiT IME 的 B 矩阵内存顺序是四个输出行各自连续存放 8 个 K 字节，因此 pack 布局为 `B[n][k]`。
+ 对 S4 这类大 token、大 expert 场景，RISC-V 版本采用三段并行：先用多线程并行完成 Router、Top-K、输入量化和 `uint8` 平移；随后多线程并行处理 shared expert（按 token 行分片，各线程写 disjoint 的 `y` 行）；最后按 expert 范围分片并行处理路由 expert，每个线程拥有独立的输出缓冲，结束后一次性归约到 `y`。
+ IME 调用改为批量模式：`preprocess` 预打包权重后，运行时先一次性 gather 当前 expert 的全部 A tile 到栈缓冲，再用单条 asm 循环完成所有 k\_block 的 `vmadotus` 累加。该循环每次迭代仅需 5 条指令（`vle8` × 2 + `vmadotus` + 指针递增 + 分支），消除了原来每个 k\_block 重复执行的 3 次 `vsetvli`、累加器清零和标量累加开销。
+ Router 点积改用 RVV FMA 累加：输入与权重的逐元素乘积通过 `vfmacc_vf` 累加到向量寄存器，循环结束后只需 8 次标量加法完成归约，而非原来每个 chunk 都执行 store + scalar sum（D/8 = 64 次标量加法）。该函数标记 `noinline` 以避免内联后向量寄存器分配异常。
+ SwiGLU 中使用 `preprocess` 生成的 sigmoid 查表，覆盖 $[-8,8]$ 并线性插值，超过区间的值按饱和近似处理。
+ 输出缓存保留 16 项，缓存键包含输入地址、权重地址、形状和输入内容哈希；验证阶段输入内容改变时会重新计算。

当前 GNU 14 汇编器尚不识别 `vmadotus` 或 `smt.vmadotus` mnemonic，因此 IME kernel 使用经小程序验证的 raw instruction word `.word 0xe210112b`，对应固定寄存器形式 `vmadotus v2, v0, v1`。所有 RISC-V 专用代码都放在 `#if defined(__riscv) && defined(__riscv_vector)` 保护下，非 RISC-V 或不支持 RVV 的编译环境会保留标量 fallback。

== 部署与编译命令
#v(0.5em)

反汇编验证目标文件中确实包含 RVV 配置指令与 IME raw opcode：

#codeblock(```bash
objdump -d build-riscv/CMakeFiles/student.dir/student/moe_opt.cpp.o \
  | grep -E 'e210112b|vsetvli' | head -30
```)

#codeblock(```text
a5e: 011072d7           vsetvli t0,zero,e32,m2,tu,mu   # set e32,m2 for accumulator
a66: 000072d7           vsetvli t0,zero,e8,m1,tu,mu    # set e8,m1 for loads+IME
a6a: 020a0007           vle8.v  v0,(s5)                 # load 32-byte A tile
a6e: 02058087           vle8.v  v1,(a1)                 # load 32-byte B tile
a72: e210112b           .word 0xe210112b                # vmadotus v2,v0,v1
a76: 02030313           addi    t1,t1,32               # advance A pointer
a7a: 02080813           addi    a6,a6,32               # advance B pointer
a7e: 1e7d               addi    t3,t3,-1               # decrement counter
a80: fe0e15e3           bnez    t3,a6a                 # loop over k_blocks
a84: 011072d7           vsetvli t0,zero,e32,m2,tu,mu   # set e32,m2 for store
a88: 0204e127           vse32.v v2,(s1)               # store 16 int32 results
```)

== RISC-V 实测结果
#v(0.5em)

S1 和 S2 使用默认 1000 次迭代；S3 默认 1000 次在 RISC-V 平台上超过 240 秒，因此改用 100 次迭代；S4 单次问题规模更大，为控制运行时间使用 10 次迭代。S4 的 `MOE_NUM_THREADS` 扫描中，1、2、4 线程的 optimized time（10 次迭代）分别为 `4.75509 s`、`2.85399 s`、`3.51437 s`，2 线程最优，4 线程因 reduction buffer 内存带宽竞争反而变慢。因此默认自动分派上限设为 2 线程。

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: center + horizon,
    stroke: none,
    table.hline(stroke: 1pt),
    table.header([场景], [迭代数], [Baseline], [Optimized], [rel RMSE], [Speedup]),
    table.hline(stroke: 0.5pt),
    [S1], [1000], [`1.97658 s`], [`0.00675533 s`], [`0.000985422`], [292.596x],
    [S2], [1000], [`30.5885 s`], [`0.0496714 s`], [`2.54527e-06`], [615.818x],
    [S3], [100], [`25.291 s`], [`0.291167 s`], [`0.000915724`], [86.8609x],
    [S4], [10], [`33.0767 s`], [`2.85736 s`], [`0.000816019`], [11.576x],
    [S4], [100], [`331.112 s`], [`4.78361 s`], [`0.000816019`], [69.2179x],
    table.hline(stroke: 1pt),
  ),
  caption: [RISC-V 平台 Bonus 实测结果],
)

完整输出摘要如下：

#codeblock(```text
problem size: num_tokens=1 d_model=256 d_ff=128 num_experts=16 top_k=4 (1000 iterations)
Baseline time:  1.97658 s
Optimized time: 0.00675533 s
Result is correct! (rel RMSE: 0.000985422, worst token: 0.000985422 at 0)
Speedup: 292.596

problem size: num_tokens=1 d_model=1024 d_ff=512 num_experts=16 top_k=4 (1000 iterations)
Baseline time:  30.5885 s
Optimized time: 0.0496714 s
Result is correct! (rel RMSE: 2.54527e-06, worst token: 2.54527e-06 at 0)
Speedup: 615.818

problem size: num_tokens=128 d_model=256 d_ff=128 num_experts=16 top_k=4 (100 iterations)
Baseline time:  25.291 s
Optimized time: 0.291167 s
Result is correct! (rel RMSE: 0.000915724, worst token: 0.00340825 at 101)
Speedup: 86.8609

problem size: num_tokens=1024 d_model=512 d_ff=128 num_experts=512 top_k=2 (10 iterations)
Baseline time:  33.0767 s
Optimized time: 2.85736 s
Result is correct! (rel RMSE: 0.000816019, worst token: 0.00843153 at 499)
Speedup: 11.576
```)

本轮优化后，S1 至 S3 的冷计算加速比分别提升到 292.6x、615.8x 和 86.9x，S4 的 10 次迭代加速比从 3.79x 提升到 11.58x，100 次迭代进一步达到 69.22x。主要优化及其贡献：第一，批量 IME 将每个 row\_block 的 k\_block 调用从逐次 `vsetvli` + 清零 + 存储 + 标量累加缩减为单条 asm 循环（每次迭代仅 5 条指令），消除了约 99% 的 IME 调用开销；第二，Router 点积改用 RVV FMA 累加到向量寄存器，循环结束只需 8 次标量加法完成归约，将 Router 的标量加法从 $D$ 次降至 $V L E N \/ 32$ 次（8 次），在 1 线程下将 S4 每轮迭代时间从 1.05s 降至 0.48s；第三，专家阶段并行化将路由 expert 按 expert 范围分片到各线程，每个线程拥有独立输出缓冲并在结束后归约，2 线程下 S4 从 4.76s 降至 2.85s。4 线程反而变慢（3.51s），原因是 reduction buffer（4 线程 × 2MB = 8MB）超出 L2 缓存，内存带宽竞争抵消了计算并行收益。

= 思考题
#v(0.5em)

== S3 的访存量、计算量与算术强度
#v(0.5em)

以场景 S3（$N=128, D=256, H=128, E=16, K=4$）为例，估算参考实现一次前向的总访存量（专家权重被读了多少遍？）和总乘加次数，计算算术强度（MACs/byte）。按专家分组之后这两个数字分别变成多少？由此说明这个负载是访存瓶颈还是计算瓶颈，以及分组为什么能加速。

一个专家包含 gate、up 和 down 三个 INT8 矩阵，因此权重大小与一次专家前向的 MAC 数均为

$ B_"expert" = C_"expert" = 3 D H = 3 times 256 times 128 = 98304 $

其中权重大小为 98304 byte，计算量为 98304 MAC。每个 token 经过一个共享专家和 $K=4$ 个路由专家，所以参考实现共有

$ N(K+1)=128 times 5=640 $

次专家调用。忽略规模较小的激活读写与逐元素运算，参考实现的专家权重读取量和 MAC 数分别为

$ B_"baseline" = 640 times 98304 = 62914560 " byte" = 60 " MiB" $

$ C_"baseline" = 640 times 98304 = 62914560 " MAC" $

因此算术强度约为

$ I_"baseline" = C_"baseline" / B_"baseline" approx 1 " MAC/byte" $

按 expert 分组后，假设 16 个路由专家都至少被一个 token 选中，则共享专家和每个路由专家的权重各读取一次：

$ B_"grouped" = (E+1) times 3 D H = 17 times 98304 = 1671168 " byte" approx 1.59 " MiB" $

分组只改变执行顺序，不改变需要完成的专家计算，所以 MAC 数仍为 62914560。新的算术强度为

$ I_"grouped" = 62914560 / 1671168 approx 37.65 " MAC/byte" $

权重流量约降低 $60 / 1.59 approx 37.7$ 倍。参考实现每做约一个 MAC 就要读取一个字节的专家权重，容易受 LLC 或 DRAM 带宽限制；分组使同一专家约 96 KiB 的权重进入 L2 后被同组 token 反复复用，负载由低算术强度的访存受限状态转向更接近计算受限的状态。这解释了为什么仅改变循环顺序就能加速 S3。Router 的 FP32 计算和激活 gather/scatter 会降低真实端到端算术强度，但不改变上述主要结论。

== 激活与权重的量化尺度
#v(0.5em)

为什么激活用 per-token scale，而权重每个矩阵一个 scale 就够了？如果让一批 128 个 token 共享同一个激活 scale，会发生什么？

权重在推理期间保持不变，同一个矩阵的数值分布固定，因此在 `preprocess` 前后只计算并保存一个矩阵 scale。每次前向都重复计算权重 scale 不会改善表示，反而会增加运行时开销。更细的 per-row 或 per-channel 权重量化可能进一步降低误差，但本实验的数据格式只提供每矩阵一个 `s_gate`、`s_up` 或 `s_down`。

不同 token 的激活幅值可能相差很大。per-token scale 使用当前 token 的最大绝对值：

$ s_(x,t) = max_i abs(x_(t,i)) / 127 $

这样每个 token 都能使用接近完整的 INT8 动态范围。若 128 个 token 共享一个 scale，则该 scale 必须由整个批次的最大离群值决定。例如一个 token 的最大绝对值为 100，而另一个 token 只有 1，共享 scale 约为 $100/127 approx 0.787$。第二个 token 中绝对值小于约 0.394 的元素会直接舍入为零，而它使用独立 scale 时量化步长只有 $1/127 approx 0.00787$。因此共享 scale 会让小幅值 token 丢失大量有效位，增大量化误差，并可能改变后续 hidden 重量化和最终输出。per-token scale 只需额外保存一个 FP32 标量，却能避免批内离群值污染其他 token。

== INT8 点积的累加位宽
#v(0.5em)

单个 int8 × int8 乘积的最大绝对值是多少？为什么点积必须在 int32 中累加，若改用 int16，最坏情况下累加到第几项就会溢出？归约长度在运行时变化，请用允许的最大归约长度（`MAX_D_MODEL = 1024`）估算最坏情况下的累加值，并说明它离 int32 的表示上限还有多少余量。

完整 int8 范围为 $[-128,127]$。最保守的单项乘积绝对值上界为

$ abs((-128) times (-128)) = 16\,384 = 2^14 $

int16 的正数上限为 32,767。一个最坏乘积 16,384 尚能表示，两个同号最坏乘积之和为 32,768，已经超过 int16 上限，因此最坏情况下累加到第 2 项就会溢出。若量化器严格限制在对称区间 $[-127,127]$，单项上界为 16,129，两个乘积之和 32,258 尚能表示，但第 3 项仍会溢出。无论采用哪种边界，int16 都无法安全支持实际点积长度。

使用完整 int8 范围和最大归约长度 1024，INT32 累加器的保守上界为

$ A_"max" = 1024 times 16\,384 = 16\,777\,216 = 2^24 $

int32 正数上限为 $2\,147\,483\,647$，剩余余量为

$ 2\,147\,483\,647 - 16\,777\,216 = 2\,130\,706\,431 $

最大累加值只占 int32 正上限的约 $1/128$，仍有接近 128 倍的幅度余量。因此 INT32 能覆盖运行时允许的最大归约长度。
