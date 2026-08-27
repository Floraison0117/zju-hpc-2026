#import "@preview/tablem:0.3.0": tablem, three-line-table
#import "@preview/cuti:0.4.0": show-cn-fakebold
#import "@preview/algo:0.3.6": algo, code
#show: show-cn-fakebold
#show table: it => align(center, it)
#set text(font: ("Palatino Linotype", "KaiTi"))

#set page(numbering: "1")
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => { counter(math.equation).update(0); it }
#set math.equation(numbering: n => {
  let h-counter = counter(heading).get()
  let h-num = if h-counter.len() > 0 { h-counter.first() } else { 0 }
  numbering("(1.1)", h-num, n)
})
#show enum: it => { set block(spacing: 0.5em); pad(left: 2em, it) }
#let centertitle(body) = [
  #set text(size: 24pt, weight: "bold")
  #align(center)[ #v(1em) #body ]
]
#centertitle[性能分析技术基础]

#let intuition(body) = block(width: 100%, inset: 1em, stroke: (left: 2pt + blue.darken(20%)), fill: blue.lighten(88%))[ #text(weight: "bold", fill: blue.darken(30%))[直觉] #h(0.5em) #body ]
#let example(body) = block(width: 100%, inset: 1em, stroke: (left: 2pt + green.darken(20%)), fill: green.lighten(88%))[ #text(weight: "bold", fill: green.darken(30%))[例] #h(0.5em) #body ]
#let aside(body) = block(width: 100%, inset: 1em, fill: luma(235))[ #emph(body) ]

#set par(first-line-indent: (amount: 2em, all: true), spacing: 1em, leading: 0.8em)
#align(center)[ #text(size: 18pt, weight: "bold")[目 $quad$ 录] ]
#v(1em)
#show outline.entry.where(level: 1): it => { v(1.2em, weak: true); strong(it) }
#outline(title: none, indent: 1.5em)
#pagebreak()

= 引言：为什么需要性能分析

#v(0.5em)
你刚写完一个 HPC 程序,运行起来比预期慢了很多。你的第一反应可能是:"是不是那个三重循环太慢了?我去优化一下。"但改完之后,性能几乎没有提升。为什么?因为你优化的可能根本不是真正的瓶颈。*性能瓶颈*（Performance Bottleneck）往往不在你猜想的位置,直觉告诉你应该优化那个耗时最长的循环,但真正的瓶颈可能在于*缓存未命中*（Cache Miss）、*分支预测失败*（Branch Misprediction）,或者内存带宽耗尽。

#v(0.5em)
*性能分析*（Profiling）是 HPC 中"先测量、再优化"的核心方法论。它提供数据驱动的手段,帮你定位真正的热点代码段、理解内存访问模式、并衡量优化效果。当数据规模或系统变大时,扩展性问题会浮现,只有通过 profiling 才能系统性地诊断。

#intuition[不妨这样想:不做 profiling 的优化就像不看地图就找路,你可能走得很努力,但方向完全错了。profiling 就是那张地图,告诉你当前在哪、瓶颈在哪、该往哪里使劲。]

#v(0.5em)
一个真实的例子:在 *m5C-RNA* 这个 HPC 应用中,开发者通过 profiling 发现,程序的大部分时间并不是花在核心计算上,而是花在了意想不到的 I/O 和内存访问上。如果没有 profiling,他们可能一直在优化错误的代码段。

== Profiling 与猜测的对比

#v(0.5em)
我们来对比两种工作方式:

#v(0.5em)
*不做 profiling 的问题*:
- 只优化"看起来明显"的瓶颈,而真正的热点往往隐藏在深处。
- 在错误的代码段上浪费时间,投入与回报不成正比。
- 对系统行为的理解有限,无法判断优化的天花板在哪。

#v(0.5em)
*数据驱动优化的优势*:
- 聚焦真正的*热点*（Hotspot）,即实际耗时最多的代码段。
- 理解内存访问模式,判断瓶颈是计算受限还是带宽受限。
- 衡量每次优化的实际效果,避免"感觉变快了"的主观判断。

#v(0.5em)
记住核心原则:*先测量、再优化、后验证*。这三步缺一不可。

= Profiling 的两大类别

#v(0.5em)
HPC 中的 profiling 可以分为两大类别,分别从不同角度审视性能。你需要同时掌握两者,才能全面理解程序的性能状况。

== 系统级剖析

#v(0.5em)
*系统级剖析*（System Profiling）关注硬件整体的性能,给出优化的"上限",即这台机器最多能算多快、搬多少数据。它回答的问题是:"这台硬件的理论极限是多少?"

#v(0.5em)
典型关注点包括:
- *峰值浮点性能*（Peak Float Performance）:CPU/GPU 每秒能执行多少次浮点运算。
- *内存带宽*（Memory Bandwidth）:内存子系统每秒能搬运多少字节数据。
- *核间通信延迟*（Core-to-Core Latency）:不同核心之间传递消息的延迟。

#intuition[系统级剖析就像测量一条公路的限速和车道数。它告诉你这条路理论上最多能跑多快、能并行通过多少辆车,但并不关心你具体开了什么车、驾驶技术如何。]

== 程序级剖析

#v(0.5em)
*程序级剖析*（Program Profiling）关注单个程序的行为,回答的问题是:"我的程序离硬件极限还有多远?"它深入到程序内部的:
- *内存访问模式*（Memory Access Patterns）:程序如何读写内存,是否连续、是否有冲突。
- *分支预测*（Branch Prediction）:条件跳转的预测命中率。
- *缓存命中率*（Cache Hit Rate）:数据在缓存中找到的比例。

#v(0.5em)
两者缺一不可:系统级告诉你"天花板在哪",程序级告诉你"离天花板多远"。只有同时了解两者,你才能判断当前的优化空间还有多大,以及瓶颈究竟出在硬件限制还是程序实现上。

#aside[如果系统级测试显示内存带宽为 100 GB/s,而你的程序级剖析显示实际只利用了 20 GB/s,那么瓶颈很可能在程序的内存访问模式上,而非硬件本身。]

== 系统级剖析工具推荐

#v(0.5em)
以下是几款常用的系统级剖析工具,每款聚焦一个硬件维度:

#v(0.5em)
#table(
  columns: 3,
  [*工具*], [*测量目标*], [*用途*],
  [CPUFP], [SIMD 指令峰值性能], [测量各类 SIMD 指令能达到的最大吞吐量],
  [core-to-core-latency], [核间通信延迟], [测量核心之间传递消息的延迟,辅助 NUMA 优化],
  [STREAM], [内存带宽], [测量内存子系统的持续带宽],
  [cpu-micro-benchmark], [CPU 微架构], [逆向工程 CPU 微架构细节],
)

#v(0.5em)
这些工具都是开源的,可以在 GitHub 上找到。它们通常以*微基准测试*（Micro-benchmark）的形式运行:用精心设计的小程序压测硬件的某个特定方面,从而获得硬件的理论上限数据。

= CPU 性能分析工具

#v(0.5em)
有了系统级的"天花板"数据,接下来我们要用程序级工具看看程序离天花板有多远。CPU 端有两款主流工具:Perf 和 Intel VTune。

== Perf：Linux 内核剖析工具

#v(0.5em)
*Perf*（Linux Performance Profiler）是 Linux 内核自带的剖析工具,它的最大优势是开销极低,适合在生产环境中持续使用。Perf 在各类平台上都能运行(包括国产 PAC 处理器),是 HPC 工程师的必备技能。

#v(0.5em)
Perf 的核心能力:
- *CPU 采样与追踪*（Sampling and Tracing）:周期性地记录程序正在执行的指令地址,统计各函数的耗时占比。
- *硬件计数器访问*（Hardware Counter Access）:直接读取 CPU 的性能计数器,获取缓存命中率、分支预测失败率等微架构指标。
- *调用图生成*（Call Graph Generation）:记录函数调用关系,帮你理解热点函数是被谁调用的。
- *内存剖析*（Memory Profiling）:分析内存访问的延迟和命中情况。
- *低开销*（Low Overhead）:采样式工作,对程序运行速度的影响很小。

#v(0.5em)
常用命令:

```bash
perf top          # 实时查看当前系统的热点函数
perf record       # 采样记录程序运行数据,生成 perf.data 文件
perf report       # 读取 perf.data,展示分析报告
perf stat         # 统计硬件计数器,如缓存命中率、分支预测失败率
```

#v(0.5em)
逐行说明:

#v(0.5em)
- `perf top` 像 `top` 命令一样实时刷新,但显示的是各函数的 CPU 占比,适合快速定位热点。
- `perf record` 在程序运行期间持续采样,把结果写入 `perf.data` 文件,之后可以离线分析。
- `perf report` 读取 `perf.data`,以交互式界面展示热点函数及其调用链。
- `perf stat` 不采样,而是统计程序运行期间各类硬件事件的总数,适合快速评估整体特征。

#example[假设你有一个矩阵乘法程序 `gemm`。用 `perf stat ./gemm` 运行后,输出可能显示 `L1-dcache-load-misses` 占比为 15%,说明 L1 数据缓存的未命中率较高,程序可能存在内存访问不连续的问题。再用 `perf record ./gemm` 采样,`perf report` 会告诉你具体哪个函数的缓存未命中最多。]

== Intel VTune Profiler

#v(0.5em)
*Intel VTune Profiler* 是 Intel 提供的综合性能分析工具,功能比 Perf 更丰富,提供图形化界面,适合离线深度分析。如果说 Perf 是命令行的瑞士军刀,VTune 就是图形化的专业工作站。

#v(0.5em)
VTune 的优势:
- *丰富的 GUI 界面*（Rich GUI Interface）:可视化展示热点、调用栈、时间线等,比纯文本报告更直观。
- *高级热点分析*（Advanced Hotspot Analysis）:不仅找出热点函数,还能定位到具体代码行和汇编指令。
- *线程分析*（Threading Analysis）:分析多线程程序的并行效率、锁竞争和负载均衡。
- *内存与缓存分析*（Memory and Cache Analysis）:深入分析各级缓存的命中率和内存带宽利用。
- *Intel 硬件优化*（Intel Hardware Optimization）:针对 Intel CPU 的微架构特性提供优化建议。

#v(0.5em)
VTune 提供多种分析类型:

#v(0.5em)
#table(
  columns: 2,
  [*分析类型*], [*关注问题*],
  [Hotspots（热点）], [定位耗时最多的代码段和指令],
  [Threading（线程）], [分析线程并行效率、锁竞争和等待],
  [MPI], [分析 MPI 通信瓶颈、负载不均和同步开销],
  [Memory Access（内存访问）], [缓存命中率与内存带宽分析],
  [Microarchitecture（微架构）], [CPU 流水线利用率、分支预测、端口占用],
)

#v(0.5em)
Perf 和 VTune 并非互斥。日常快速排查用 Perf,深度调优用 VTune,是 HPC 工程师常见的组合工作流。

#aside[VTune 需要 Intel 许可证才能使用全部功能,但学生和研究者可以申请免费的 Intel oneAPI 教育许可。Perf 则完全免费,随 Linux 内核发布。]

= GPU 性能分析工具

#v(0.5em)
GPU 程序的性能分析比 CPU 更复杂,因为 GPU 涉及成千上万的线程、复杂的内存层次和特殊的调度机制。NVIDIA 提供了两款互补的工具:Nsight Systems 和 Nsight Compute。

== Nsight Systems：系统级 GPU 剖析

#v(0.5em)
*Nsight Systems* 提供*系统级时间线视图*（System-wide Timeline View）,关联 CPU 和 GPU 的活动,帮你找到"瓶颈在哪"。它从宏观角度审视整个应用的执行过程。

#v(0.5em)
核心能力:
- *系统级时间线*（System-wide Timeline）:在时间轴上展示 CPU 线程、CUDA stream、GPU kernel 的执行情况。
- *CPU/GPU 活动关联*（CPU and GPU Activity Correlation）:对比 CPU 提交任务和 GPU 执行任务的时间,找出 CPU 等待 GPU 或 GPU 空闲的时段。
- *数据传输分析*（Memory Transfer Analysis）:分析 host 与 device 之间的数据拷贝,发现不必要的传输。
- *多 GPU 支持*（Multi-GPU Support）:同时分析多个 GPU 的活动。
- *低开销*（Low Overhead）:对程序运行影响小,适合分析真实负载。

#v(0.5em)
适用场景:
- 应用*瓶颈定位*（Application Bottlenecks）:找出程序整体上受限于 CPU 还是 GPU。
- *GPU 利用率分析*（GPU Utilization）:检查 GPU 是否被充分利用,是否存在大量空闲。
- *数据传输优化*（Data Transfer Optimization）:发现冗余的 host-device 拷贝。
- *多线程分析*（Multi-threading Analysis）:分析多 CPU 线程提交 GPU 任务的效率。

#intuition[想象一条流水线工厂,Nsight Systems 就是工厂的整体监控录像。它告诉你哪个工位在排队、哪个工位在空转、物料在哪个环节卡住了。你先看整体,定位问题区域,再深入细节。]

== Nsight Compute：Kernel 级细粒度剖析

#v(0.5em)
*Nsight Compute* 深入到单个 GPU *kernel*（核函数）内部,分析 *warp 效率*、*occupancy*（占用率）、*memory throughput*（内存吞吐量）等,帮你找到"瓶颈的原因"。

#v(0.5em)
分析能力:
- *详细 kernel 指标*（Detailed Kernel Metrics）:每个 kernel 的执行时间、指令吞吐、内存带宽利用率。
- *内存吞吐分析*（Memory Throughput Analysis）:各级缓存（L1、L2、HBM）的命中率和带宽利用。
- *Warp 效率指标*（Warp Efficiency Metrics）:分析 warp 内的线程是否都在有效工作,还是大量线程在等待。
- *指令分析*（Instruction Analysis）:各类指令的执行占比和吞吐。
- *性能限制因素识别*（Performance Limiters Identification）:自动判断 kernel 受限于计算、内存还是延迟。

#v(0.5em)
分析焦点:
- *Kernel 优化*（Kernel Optimization）:找出 kernel 内部的低效代码。
- *内存访问模式*（Memory Access Patterns）:检查是否合并访问（coalesced access）、是否存在 bank conflict。
- *Occupancy 分析*（Occupancy Analysis）:SM 上活跃 warp 数与最大 warp 数的比例。
- *计算吞吐量*（Compute Throughput）:计算单元的利用率。

#v(0.5em)
*关键区别*:Nsight Systems 看"全局时间线"找瓶颈位置,Nsight Compute 看"kernel 细节"找瓶颈原因,两者配合使用,不要混用。典型工作流是:先用 Nsight Systems 定位最耗时的 kernel,再用 Nsight Compute 深入分析该 kernel 的内部瓶颈。

#aside[Occupancy 高不一定意味着性能好,但 Occupancy 低且 warp 大量停滞,通常是性能杀手。理解这一点需要结合具体的 kernel 行为,不能只看单一指标。]

= Roofline 模型

#v(0.5em)
前面介绍了各种 profiling 工具,但如何把测量数据转化为可操作的判断?*Roofline 模型*（Roofline Model）是性能分析的核心理论工具,它结合算力（FLOPS）与带宽,给出某算法在特定硬件上能达到的性能理论上限。

#v(0.5em)
Roofline 模型的核心问题是:

#v(0.5em)
*"一个有 $A$ 次计算、$B$ 次内存访问的模型,运行在算力 $C$、带宽 $D$ 的系统上,可达性能上限 $E$ 是多少?"*

#v(0.5em)
这里涉及两个维度:
- *计算平台*（Computing Platform）:硬件的*计算能力*（FLOPS）和*内存带宽*（Memory Bandwidth）。
- *算法或模型*（Algorithm/Model）:程序的*总计算量*和*总内存访问量*。

== 计算强度

#v(0.5em)
*计算强度*（Operational Intensity,简称 OI）定义为每字节内存访问所执行的运算次数:

#v(0.5em)
$ "OI" = frac{"运算次数"}{"内存访问字节数"} = A / B $

#v(0.5em)
计算强度的单位是 FLOP/Byte,它决定了程序是*计算受限*（Compute-bound）还是*内存受限*（Memory-bound）。

#intuition[计算强度可以理解为"每搬一桶水能洗多少件衣服"。如果洗一件衣服需要很多水（计算强度低），那你大部分时间都在搬水,瓶颈在带宽;如果洗一件衣服只需要一点点水但需要很长时间搓洗（计算强度高），那瓶颈在计算速度。]

#v(0.5em)
根据计算强度,性能上限由两个因素中的较小者决定:

#v(0.5em)
$ E = min(C, D times "OI") $

#v(0.5em)
- 当计算强度*高*（$D times "OI" > C$）时,程序是*计算受限*（Compute-bound）,性能上限 $E = C$,即算力成为瓶颈。
- 当计算强度*低*（$D times "OI" < C$）时,程序是*内存受限*（Memory-bound）,性能上限 $E = D times "OI"$,即带宽成为瓶颈。

#v(0.5em)
形象地说,Roofline 图像是一条"屋檐线":横轴是计算强度（对数尺度）,纵轴是性能（对数尺度）。低计算强度区域是带宽斜线,性能随计算强度线性增长;高计算强度区域是算力水平线,性能不再增长。两条线的交点就是从 memory-bound 到 compute-bound 的*转折点*（Ridge Point）。

== 例子：用 Roofline 分析一个程序

#example[假设一个系统的算力 $C = 10$ TFLOPS,带宽 $D = 100$ GB/s。一个程序每读 1 字节做 2 次运算,即计算强度 $"OI" = 2$ FLOP/Byte。

#v(0.5em)
计算性能上限:

#v(0.5em)
$ E = min(C, D times "OI") = min(10 " TFLOPS", 100 " GB/s" times 2 " FLOP/Byte") $

#v(0.5em)
带宽限制部分:$100 times 2 = 200$ GFLOPS $= 0.2$ TFLOPS。算力限制部分:$10$ TFLOPS。取较小值,实际上限 $E = 0.2$ TFLOPS,程序是 memory-bound。

#v(0.5em)
这意味着,尽管硬件算力高达 10 TFLOPS,但由于计算强度太低,程序只能发挥 2% 的算力。要让程序接近算力上限,需要提高计算强度,例如通过*循环展开*（Loop Unrolling）、*分块*（Tiling）等手段增加数据复用,让每次从内存搬来的数据被多次使用。]

#v(0.5em)
回过头看,这个例子揭示了一个反直觉的事实:即使你花大价钱买了顶级 GPU,如果你的程序计算强度太低,它也只能发挥出很小一部分性能。这就是为什么 profiling 和 Roofline 分析如此重要:它们告诉你,瓶颈不在硬件,而在程序的内存访问模式。

= 实战：性能分析工作流

#v(0.5em)
把前面的工具和理论串联起来,一个典型的 CPU 性能分析工作流如下:

#v(0.5em)
+ *系统级基线*（System Baseline）:用 STREAM 测量内存带宽,用 CPUFP 测量峰值算力,获得硬件的"天花板"。
+ *程序级热点定位*（Hotspot Identification）:用 `perf record` 采样程序运行,`perf report` 找出最耗时的函数。
+ *瓶颈分类*（Bottleneck Classification）:用 `perf stat` 查看缓存命中率等指标,或用 Roofline 模型判断是 compute-bound 还是 memory-bound。
+ *深度分析*（In-depth Analysis）:对关键热点用 VTune 的 Memory Access 或 Microarchitecture 分析,找出具体的微架构瓶颈。
+ *优化与验证*（Optimize and Verify）:实施优化后,重新 profiling,对比性能变化是否达到预期。

#v(0.5em)
对于 GPU 程序,工作流类似,但工具替换为:

#v(0.5em)
+ 用 Nsight Systems 获取整体时间线,找出最耗时的 kernel 和不必要的 CPU-GPU 数据传输。
+ 用 Nsight Compute 深入分析关键 kernel 的 occupancy、内存访问模式和 warp 效率。
+ 用 Roofline 模型判断 kernel 是 compute-bound 还是 memory-bound,指导优化方向。

#aside[优化的黄金法则:每次只改一个变量。不要同时尝试多种优化,否则你无法判断是哪个改动带来了性能变化。]

= 本章你将学会

#v(0.5em)
+ 理解性能分析的必要性,认识到"不做 profiling 的优化就是盲人摸象"。
+ 区分系统级剖析与程序级剖析,知道何时用 STREAM/CPUFP,何时用 Perf/VTune。
+ 掌握 Perf 的四个核心命令(`top`、`record`、`report`、`stat`)及其适用场景。
+ 理解 Nsight Systems 与 Nsight Compute 的分工,知道"找瓶颈位置"与"找瓶颈原因"的区别。
+ 运用 Roofline 模型判断程序是 compute-bound 还是 memory-bound,并据此选择优化方向。

= 要点速查

#v(0.5em)
#table(
  columns: 2,
  [*要点*], [*说明*],
  [直觉不可靠], [瓶颈通常不在你猜想的位置,必须用 profiler 测量],
  [系统 vs 程序], [系统级给硬件上限,程序级给离上限的距离],
  [先测基线], [优化前先测量,优化后也要测量验证效果],
  [计算强度], [运算次数 / 内存访问字节数,决定 compute-bound 或 memory-bound],
  [Perf], [内核级低开销,适合生产环境,命令行使用],
  [VTune], [图形化深度分析,适合离线调优,支持多种分析类型],
  [Nsight Systems], [系统级时间线,看全局,找瓶颈位置],
  [Nsight Compute], [Kernel 级细粒度,看细节,找瓶颈原因],
  [Roofline 转折点], [带宽斜线与算力水平线的交点,跨越它即从 memory-bound 转为 compute-bound],
  [大规模测试], [扩展性问题只在目标规模才显现,小规模测试可能掩盖问题],
)

= 小结

#v(0.5em)
性能分析是 HPC 的核心方法论:不做 profiling 的优化等于盲人摸象。本章从"为什么需要性能分析"出发,介绍了两大类别(系统级 vs 程序级)、系统级工具(CPUFP、core-to-core-latency、STREAM、cpu-micro-benchmark)、CPU 工具(Perf、VTune)、GPU 工具(Nsight Systems、Nsight Compute),以及统一性能理论 Roofline 模型。

#v(0.5em)
记住核心原则:*先测量、再优化、后验证*。先用系统级工具了解硬件上限,再用程序级工具定位瓶颈,用 Roofline 模型判断瓶颈类型,最后针对性地优化并重新测量。下一讲将进入具体的性能优化实践,学习如何把 profiling 的发现转化为实际的代码改进。

#v(2em)
#align(center)[#text(size: 12pt, fill: luma(120))[讲义基于 HPC101-2025 Day9「性能分析技术基础 Introduction to Profiling」课程内容编写]]
