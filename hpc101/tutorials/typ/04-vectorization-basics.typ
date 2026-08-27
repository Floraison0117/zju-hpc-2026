#import "@preview/tablem:0.3.0": tablem, three-line-table
#import "@preview/cuti:0.4.0": show-cn-fakebold
#import "@preview/algo:0.3.6": algo, code
#show: show-cn-fakebold
#show table: it => align(center, it)
#set text(font: ("Palatino Linotype", "KaiTi"))

#set page(numbering: "1")
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => {
  counter(math.equation).update(0)
  it
}
#set math.equation(numbering: n => {
  let h-counter = counter(heading).get()
  let h-num = if h-counter.len() > 0 { h-counter.first() } else { 0 }
  numbering("(1.1)", h-num, n)
})
#show enum: it => {
  set block(spacing: 0.5em)
  pad(left: 2em, it)
}
#let centertitle(body) = [
  #set text(size: 24pt, weight: "bold")
  #align(center)[
    #v(1em)
    #body
  ]
]
#centertitle[向量化并行计算基础 — 入门讲义]

#set par(first-line-indent: (amount: 2em, all: true), spacing: 1em, leading: 0.8em)
#align(center)[
  #text(size: 18pt, weight: "bold")[目 $quad$ 录]
]
#v(1em)
#show outline.entry.where(level: 1): it => {
  v(1.2em, weak: true)
  strong(it)
}
#outline(title: none, indent: 1.5em)
#pagebreak()

= 为什么需要向量化

传统编程模式下，CPU 一次只处理一个数据：取一条指令、取一个操作数、执行运算、存回结果。然而在现代 CPU 中，算术逻辑单元（ALU）完全有能力同时处理多个数据。*向量化计算*的核心思想就是：一次参与运算的是由多个值组成的向量，而非单个标量。

如果能把循环中彼此独立的计算组织成向量操作，就能在相同时间内完成数倍的工作量。这正是从 NumPy 到 AVX-512 等技术背后的统一动机。向量化既可以是软件层面的（如 NumPy 的数组运算），也可以是硬件层面的（如 SIMD 指令集），两者相辅相成。

= 并行的两种形态

== 无依赖关系的并行

考虑两个 $3 times 3$ 矩阵的加法 $C = A + B$。矩阵中 9 个位置的加法彼此完全独立，没有数据依赖。这就像 9 位厨师各自独立做一道菜，互不干扰。这种"令人愉悦的并行"（embarrassingly parallel）是最理想的情况，可以直接向量化，每个元素的计算互不影响。

== 乘法的并行与规约

考虑向量点积 $a dot b = sum_(i=0)^(n-1) a_i b_i$。乘法部分对每个 $i$ 独立，可以并行执行。但最终的求和需要把所有乘积累加起来，这个*规约*（Reduction）步骤本质上是串行的。

规约可以用树形结构优化：先将相邻的乘积两两相加，再对结果两两相加，将 $O(n)$ 的串行步骤降为 $O(log n)$。但即使如此，最后一步仍需串行合并两个部分和。这种"部分并行、部分串行"的模式在数值计算中极为常见。

= NumPy：向量化计算的利器

NumPy 是 Python 生态中最核心的数值计算库，其核心数据结构 `ndarray` 提供了高效的向量化运算，让 Python 程序员无需编写底层循环就能享受向量化带来的性能提升。

== ndarray 内存模型

NumPy 的 `ndarray` 要求所有元素具有相同的数据类型（dtype），并且在内存中连续存储。这使得 CPU 缓存可以高效预取数据，SIMD 指令可以一次加载连续的多个元素。

相比之下，Python 的 `list` 存储的是指向 Python 对象的指针，每个元素需要一次额外的内存寻址，不仅浪费空间，还破坏了缓存局部性：

#table(
  columns: 3,
  align: (left, left, left),
  [*特性*], [*Python list*], [*NumPy ndarray*],
  [存储方式], [对象指针], [连续原始值],
  [元素类型], [可混合], [统一 dtype],
  [缓存友好性], [差], [好],
  [内存开销], [大], [小],
)

对于大规模数值计算，NumPy ndarray 的性能通常比等价的 Python 循环快一到两个数量级。

== dtype 与类型转换

dtype 决定了每个元素的存储大小和解释方式。常见类型包括 `int32`（4 字节整数）、`float64`（8 字节浮点）、`bool`（布尔）、`object`（任意 Python 对象）等。可以使用 `.astype()` 进行类型转换：

```python
import numpy as np
a = np.array([1, 2, 3], dtype=np.int32)
b = a.astype(np.float64)
```

选择合适的 dtype 既能节省内存（`int32` 比 `int64` 省一半空间），又能影响计算精度和性能。在 HPC 中，有时使用 `float32` 牺牲精度换取速度和内存节省是合理的策略。

== 广播机制

当两个数组的形状不匹配时，NumPy 的*广播*（Broadcasting）机制会自动扩展较小的数组，使其与较大的数组兼容。广播遵循三个条件，须同时判断每个维度：

- 两个数组在该维度上的长度相同，*或者*
- 其中一个数组在该维度的长度为 1，*或者*
- 其中一个数组没有该维度（视为长度 1）

例如，一个 $3 times 3$ 的矩阵可以与一个长度为 3 的一维数组相加，后者会被广播为 $3 times 3$：

```python
a = np.ones((3, 3))
b = np.array([1, 2, 3])
c = a + b
```

如果两个数组在任何维度上既不相同、也不为 1，广播将失败并抛出异常。理解广播规则对正确使用 NumPy 至关重要。

== 视图与拷贝

NumPy 的操作有时返回*视图*（View），有时返回*拷贝*（Copy），理解区别至关重要：

- *切片*返回视图：`a[1:5]` 与原数组共享内存，修改切片会影响原数组
- *花式索引*返回拷贝：`a[[0, 2, 4]]` 创建新数组，修改不影响原数组
- *赋值*是引用：`b = a` 不创建副本，`b` 和 `a` 指向同一数组

如果需要独立副本，使用 `.copy()` 方法显式拷贝。常见的陷阱是：对切片做原地修改后，原数组也被改变，导致难以追踪的 bug。

= SIMD：硬件级向量化

== SIMD 原理

SIMD（Single Instruction Multiple Data）是一种并行计算范式：一条指令同时对多个数据执行相同操作。CPU 内部有宽达 256 位或 512 位的寄存器，可以一次容纳多个浮点数并同时运算。例如，一条 AVX-512 加法指令可以同时完成 8 个双精度浮点数的加法。

== 指令集族谱

x86 平台的 SIMD 指令集经历了多代演进：

#table(
  columns: 3,
  align: (left, center, center),
  [*指令集*], [*寄存器宽度*], [*双精度浮点数容量*],
  [MMX], [64 位], [不支持浮点],
  [SSE / SSE2], [128 位], [2 个],
  [AVX], [256 位], [4 个],
  [AVX2], [256 位], [4 个],
  [AVX-512], [512 位], [8 个],
)

ARM 平台使用 NEON 指令集（128 位）。AVX-512 的 512 位寄存器一次可容纳 8 个双精度浮点数，配合 FMA（Fused Multiply-Add）指令，单条指令可以完成乘加运算。

== 加速倍数不等于数据个数

虽然 AVX-512 可以一次处理 8 个双精度浮点数，但实际加速比往往达不到 8 倍，原因包括：

- *内存带宽瓶颈*：数据加载速度跟不上计算速度，CPU 空闲等待数据
- *解码开销*：复杂指令需要更多解码周期，且指令缓存有限
- *散热限制*：AVX-512 高负载可能导致 CPU 降频，反而变慢

在某些 Intel 处理器上，开启 AVX-512 会触发降频机制，导致整体性能下降。因此实际优化中需要实测比较，而非盲目追求更宽的指令集。

== Intrinsic 函数

编译器提供了与 SIMD 指令对应的 C 函数接口，称为 *Intrinsic*。命名规则为"指令类型_操作_数据类型"：

- `_mm256_add_pd`：AVX2（256 位）加法（add），双精度（pd = packed double）
- `_mm512_mul_pd`：AVX-512 乘法，双精度
- `_mm_load_ps`：SSE 加载，单精度（ps = packed single）

这些函数对应一条或少数几条机器指令，编译器会直接生成对应的 SIMD 指令。

== 手写 SIMD 流程

手写 SIMD 代码通常遵循三个步骤：

+ *Load*：从内存加载数据到 SIMD 寄存器
+ *Compute*：执行向量运算
+ *Store*：将结果从寄存器存回内存

```c
__m512d va = _mm512_loadu_pd(&a[i]);
__m512d vb = _mm512_loadu_pd(&b[i]);
__m512d vc = _mm512_add_pd(va, vb);
_mm512_storeu_pd(&c[i], vc);
```

上述代码每次处理 8 个双精度浮点数的加法，循环步长设为 8。需要注意循环边界：当数组长度不是 8 的倍数时，剩余元素需要用标量代码处理。

== 何时手写

在简单场景下，开启编译器优化 `-O3 -mavx2` 后，编译器可以自动将循环向量化，效果往往已经足够好。只有在以下情况才需要手写：

- 代码逻辑复杂，编译器无法自动向量化
- 需要精细控制内存对齐和寄存器分配
- 对性能有极致要求，需要利用 FMA 等高级指令

手写 SIMD 还需注意四大问题：*内存对齐*（未对齐的访问会降低性能）、*循环边界*（剩余迭代需要标量处理）、*分支开销*（SIMD 内的条件分支代价高）、*寄存器超限*（寄存器不够时溢出到内存，性能暴跌）。

= 易错点与陷阱

- `*` 运算符是逐元素乘法，不是点积；点积使用 `@` 或 `np.dot()`
- 深浅拷贝陷阱：视图共享内存，修改会相互影响
- 广播的三个条件必须逐维度判断，不满足则报错
- `reshape` 中 `-1` 表示自动推断该维度大小
- SIMD 加速比不等于数据个数，受内存带宽限制
- AVX-512 可能因发热导致降频，反而比 AVX2 慢
- 手写 SIMD 需注意内存对齐、循环边界、分支开销和寄存器数量

在实际 HPC 项目中，通常会先用 NumPy 或编译器自动向量化获得基线性能，再通过性能分析工具（如 Intel VTune）定位热点，有针对性地手写 SIMD 优化关键路径。这种渐进式优化策略既避免了过早优化的陷阱，又能确保投入产出比。

#v(2em)
#align(center)[#text(size: 12pt, fill: luma(120))[讲义基于 HPC101 2025 课程内容编写]]
