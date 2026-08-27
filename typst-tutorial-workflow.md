# 从 PDF/PPTX 撰写 Typst 讲解文档的工作流与避坑指南

## 一、定位与对比

本工作流产出的是**供新手学习的教学讲义**，而非考前速记提纲。以 HPC101 课程与 Ascend C 课程的原稿（PDF/PPTX）为输入，改写成工科生友好、可独立阅读的学习材料。

| 维度 | 原稿 PPT/PDF | 本工作流产出 |
|------|-------------|-------------|
| 用途 | 课堂演示，配合讲师口述 | 课前预习 / 课后自学，脱离讲师也能读懂 |
| 密度 | 要点式，每页几个关键词 | 适中，展开动机与直觉，代码逐行解释 |
| 代码 | 截图或片段，无解释 | 关键代码配逐行批注，说明每步在干什么 |
| 风格 | 提纲式词条 | 第一人称叙事体（"我们""你""不妨想"） |
| 语义框 | 无 | 三种语义框（intuition / example / aside） |
| 要点速查表 | 无 | 可选附录，放小结前，供复习快速回看 |

## 二、整体工作流

### 1. 环境准备

| 工具 | 用途 | 安装/检查命令 |
|------|------|--------------|
| Typst | 编译 `.typ` -> `.pdf` | `typst --version` |
| python-pptx | 提取 PPTX 文本 | `pip install python-pptx` |
| pymupdf (fitz) | 提取 PDF 文本 | `pip install pymupdf` |
| Poppler | 渲染 PDF 页面供视觉检查 | `pdftoppm -v` |
| CeTZ / Fletcher | 绘制矢量技术图 | 由 Typst `@preview` 包按固定版本加载 |

字体检查（Typst 可用，注意系统字体名是 `KaiTi` 不是 `kaiti`）：

```powershell
typst fonts | Select-String "KaiTi|Palatino|SimSun|SimHei"
```

### 2. 提取源文件内容（fan out 子代理）

对每份 PDF/PPTX，启动一个 `explore` 子代理，用 Python 脚本 dump 全文，然后返回结构化大纲。

**PDF 提取脚本：**

```python
import fitz
doc = fitz.open(r"路径.pdf")
with open("extracted.txt", "w", encoding="utf-8") as f:
    for i, page in enumerate(doc):
        f.write(f"--- PAGE {i+1} ---\n{page.get_text()}\n")
```

注意：Windows 控制台默认 GBK 编码，直接 `print()` 含特殊 Unicode 字符的 PDF 文本会报 `UnicodeEncodeError: 'gbk' codec can't encode character`。必须写入 UTF-8 文件再读取。

**PPTX 提取脚本：**

```python
from pptx import Presentation
prs = Presentation(r"路径.pptx")
with open("extracted.txt", "w", encoding="utf-8") as f:
    for i, slide in enumerate(prs.slides):
        f.write(f"--- SLIDE {i+1} ---\n")
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                f.write(shape.text + "\n")
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame.text.strip():
            f.write("[NOTES] " + slide.notes_slide.notes_text_frame.text + "\n")
        f.write("\n")
```

**子代理返回的素材采集格式（不是提纲，是写作素材）：**

```
1. 新手动机（3-5 句：为什么需要这个概念，解决什么痛点）
2. 核心概念清单（8-20 个，每个需含：直觉白话 + 形式定义 + 一个可算的小例子）
3. 代码片段（如有：源文件中的代码，逐段标注每行作用）
4. 例子与案例（3-8 个可写入讲义的具体例子，尽量取小数值"算给你看"）
5. 易错点/实践提醒（5-10 项，新手常踩的坑）
6. 关键术语对照（英中对照表，首次出现时用 `*英文*（中文）` 格式）
```

### 3. 视觉盘点与配图制作

在写正文前建立 `visuals.json`，逐张记录教学目的、来源页码、制作方式、文件路径和验收状态。先把源 PDF/PPTX 渲染成缩略图并人工查看，再按“真实截图、数据图、矢量概念图、装饰图”四类选择制作路径。技术结构不得交给生成式图片决定。

### 4. 撰写 Typst 文件

每份讲义采用统一的模板头部和结构：

```
头部导入 + 语义函数定义 -> 居中标题 -> 目录 -> 正文 -> 小结 -> 版权尾注
```

正文结构（学习者友好，非提纲式）：

```
引言动机（这章解决什么问题，为什么值得学）
  -> 核心概念（每个按"动机 -> 直觉 -> 形式 -> 回看"四步展开）
  -> 实战/案例（代码逐行批注 + "取小数值算给你看"的例子）
  -> 本章你将学会（3-5 条可检验的学习成果）
  -> 要点速查（可选附录，表格形式，供复习快速回看）
  -> 小结（承上启下，衔接下一章）
```

### 5. 编译验证

```powershell
typst compile --root . "文件名.typ" "文件名.pdf"
```

编译无输出即成功，有错误则根据报错修复后重编。

### 6. 批量编译

```powershell
$dirs = @("hpc101/tutorials", "ascend-c/tutorials")
foreach ($dir in $dirs) {
    Get-ChildItem -Path $dir -Filter "*.typ" | ForEach-Object {
        typst compile --root $dir $_.FullName ($_.FullName -replace '\.typ$', '.pdf')
    }
}
```

---

## 三、高质量配图工作流

### 1. 先判断图片承担什么任务

图片必须解决一个明确的学习问题，不能只为了填空白。按下表选择制作路径：

| 图片类型 | 适用内容 | 首选方法 | 禁止事项 |
|---------|---------|---------|---------|
| 证据截图 | profiler、终端、IDE、真实运行现象 | 从可信环境重新截图，或忠实提取原稿 | 伪造运行结果、补写不存在的 UI |
| 数据图表 | 性能、带宽、占用率、可计算示例 | 从原始数据或明确公式脚本化生成 | 凭印象画趋势、把推导值写成实测值 |
| 技术概念图 | 架构、层次、数据流、流水线 | CeTZ、Fletcher 或独立 SVG 矢量重绘 | 用 AI 生成硬件连接、算法步骤或实验结论 |
| 装饰性插画 | 章节引入、非技术类比、视觉节奏 | AI 生成或有许可的素材 | 放入技术标签、品牌仿冒或暗示具体结构 |

代码一律优先使用 Typst raw block，不把代码截图当作配图。表格能够更清楚表达精确映射时，也不要强行改成插画。

### 2. 建立视觉清单和来源链

每个试点教程在自己的资源目录保存 `visuals.json`，至少包含：

- `id`：稳定且可用于交叉核对的图片标识。
- `kind`：`source-reference`、`vector-redraw`、`derived-chart` 或 `ai-decorative`。
- `purpose`：这张图具体帮助读者理解什么。
- `source_reference`：PDF 页码、PPTX 页码、数据文件或推导过程。
- `file` 或 `implementation_file`：位图文件或 Typst 矢量图实现。
- `technical_claims`：是否承载技术事实。
- `status`：`reference-only`、`draft` 或 `accepted`。

AI 装饰图还必须保存生成工具和最终提示词，并将 `technical_claims` 固定为 `false`。正文图注需要明确写出“AI 生成装饰图，不表达具体硬件结构”。

### 3. 资源目录与命名

资源跟随教程系列放置：

```text
hpc101/tutorials/assets/<tutorial-id>/
ascend-c/tutorials/assets/<tutorial-id>/
assets/lab<N>/tutorials/<tutorial-id>/
```

使用 `概念-用途-v1.svg/png` 形式的可读名称。源页截图以 `source-pageNN-*` 或 `source-slideNN-*` 命名，可以作为 `reference-only` 保留；正式图不要使用 `image-1.png` 这类无法判断用途的名字。

### 4. 源视觉提取与人工检查

PDF 先定位含关键术语的页面，再以足够分辨率渲染：

```powershell
pdftoppm -png -r 144 -f <page> -l <page> .\source.pdf .\assets\source-pageNN
```

PPTX 同时提取文本、媒体和页面结构。`python-pptx` 无法完整表达 SmartArt、组合图形和连接线，所以对图片型或形状密集的页面必须人工查看媒体资源或整页渲染。提取出来的图片只是重绘依据，是否直接进入讲义要根据清晰度、版权和信息密度重新判断。

人工检查至少回答四个问题：原图的箭头表示数据流还是层级关系，颜色是否承载语义，哪些部件被有意省略，正文是否把简化图误称为完整架构图。

### 5. 矢量绘图库的分工

试点固定使用以下版本：

```typst
#import "@preview/cetz:0.5.2"
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
```

- *Fletcher* 适合节点、箭头、层级树和线性流程，节点连接和箭头吸附写法简洁。
- *CeTZ* 适合需要坐标控制的硬件框图、时间线、柱图和自定义几何布局。
- Fletcher 0.5.8 内部依赖 CeTZ 0.3.4，可以与教程直接使用的 CeTZ 0.5.2 共存。两个版本都必须固定，不能使用浮动版本。
- Fletcher 图的边界计算对宽节点较敏感，正式使用时应整体 `align(center, scale(...))`，并在 PDF 渲染后检查左右裁切。
- CeTZ 的 `content` 中优先传入 `text(...)`、`stack(...)` 等内容值，不要把代码表达式错误地写成会被原样渲染的文本。

每个教程根目录可以保留一个 `*-visuals.typ`，集中定义矢量图函数。正文只负责调用、图注、label 和教学解释，避免把大量绘图坐标混入叙事正文。

### 6. 统一视觉与可读性标准

- 技术图优先使用矢量；截图按最终显示尺寸的至少 2 倍准备。
- 采用有限色板，同一种语义在同一教程中保持颜色一致。
- 最小图中文字以最终 PDF 100% 缩放可读为准，不能靠读者放大才能辨认。
- 除颜色外，再用文字、形状或线型表达区别，保证灰度打印仍可理解。
- 每张图必须有图注、正文引用和紧邻的解释，不能出现“图放在这里但正文不讨论”的情况。
- 简化图必须说明省略范围，推导图必须说明数据如何得到，实测图必须链接原始日志或数据。

### 7. 自动检查与视觉验收

先运行资源和来源检查：

```powershell
python .\tools\tutorial-visuals\check_visuals.py
```

再编译教程，并把含新增图片的页面渲染为 PNG：

```powershell
typst compile --root <tutorial-dir> <source.typ> <output.pdf>
pdftoppm -png -r 150 -f <first-page> -l <last-page> <output.pdf> .\tmp\tutorial-visuals\review
```

视觉验收按“主动找问题”的方式进行，至少检查：边缘裁切、文字重叠、节点过小、箭头穿过标签、低对比度、图注与正文不一致、分页割裂和装饰图被误读成技术图。首轮发现问题后必须完成一次修复与复验，不能只看编译是否成功。

### 8. 当前试点范围

首轮只覆盖以下两个教程，不迁移其他存量教程：

- `hpc101/tutorials/typ/11-gpu-programming.typ`
- `ascend-c/tutorials/03-ascend-c-programming-model.typ`

试点完成后保留工具、清单格式、包版本和视觉规范。后续是否迁移存量教程，需要单独评估阅读收益与维护成本。

---

## 四、写作风格规范（操作化）

这是本工作流的核心。以下规则将"新手学习教辅"的定位落地为可执行的写作纪律。

### 1. 动机先行

每节先一段"为什么需要这个概念/它解决了什么痛点"，再给定义。

- 反例（提纲式）："OpenMP 是一套多线程并行 API。"
- 正例（动机式）："我们已经有 `std::thread`，为什么还需要 OpenMP？因为手动管理线程要写一堆 boilerplate：创建、同步、合并、异常处理。OpenMP 用一行 `#pragma omp parallel for` 就把循环自动分给所有核心，让开发者专注算法而不是线程管理。"

### 2. 直觉 -> 形式 -> 再回看

先白话讲一遍，再上公式/代码，再用一句"换句话说"收束。

```
[直觉] 矩阵乘法就是把"行的点积"重复很多次，每次点积互相独立。
[形式] $C = A times B$, $C_(i,j) = sum_k A_(i,k) B_(k,j)$
[回看] 换句话说，每个 $C_(i,j)$ 只依赖 A 的第 i 行和 B 的第 j 列，
       所以不同的 $(i,j)$ 可以并行计算。
```

### 3. 第一人称叙事

用"我们""你""不妨想"拉近距离；允许口语转折（"等等""先别管严格性"）。避免教科书式的第三人称陈述。

### 4. 代码逐行批注（HPC 特色）

代码块不是用来"展示存在"的，而是用来"教读者看懂"的。关键代码块后必须配批注，逐段解释每行干什么、为什么这么写。

```cpp
constexpr int32_t BUFFER_NUM = 2;   // 双缓冲数量，流水线的基础
pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(T));
```

批注可以放在代码注释里（代码块内），也可以紧跟在代码块后用 `#aside` 框展开"这段代码的关键设计"。

### 5. 术语中英对照

专有术语首次出现时写 `*OpenMP*（开放式多处理）` 做一次对照，之后一律用英文。通用概念词用中文（线程、内存、带宽、缓存）。人名、工具名、框架名用原拼写。

### 6. 例子落地

每章至少一个"取小数值算给你看"的小例子，用 `#example` 框包起来。例：

- 存储层次：取 $1 "KB" = 1024 "B"$，算一个 4GB 模型在内存带宽 50GB/s 下的搬运时间。
- 并行切分：取 $N=8$ 个元素、$4$ 个核心，画出每个核心分到哪几个元素。
- 性能估算：取实测延迟，倒推算力利用率。

### 7. 排版间距

- 标题（`= ` / `== `）下若直接跟正文（非语义框或公式），标题与正文之间加 `#v(0.5em)`。
- 有序列表（`+ ...`）与上下内容之间各加 `#v(0.5em)`。

### 8. 严禁使用破折号

中文写作习惯用 `——` 做破折号，但本工作流**明确禁止**。用逗号或冒号替代：

- 反例：`超算的能力超出单机时——比如气候模拟——就必须用集群。`
- 正例：`超算的能力超出单机时，比如气候模拟，就必须用集群。`

写完全文后用以下命令检查，确保零残留：

```powershell
Select-String -Path .\*.typ -Pattern "——"
```

---

## 五、Typst 模板参考

每份 `.typ` 文件自包含，头部含统一的导入、格式设置和三个语义函数定义。

```typst
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
  #align(center)[ #v(1em) #body ]
]
#centertitle[标题]

#let intuition(body) = block(
  width: 100%,
  inset: 1em,
  stroke: (left: 2pt + blue.darken(20%)),
  fill: blue.lighten(88%),
)[
  #text(weight: "bold", fill: blue.darken(30%))[直觉] #h(0.5em) #body
]
#let example(body) = block(
  width: 100%,
  inset: 1em,
  stroke: (left: 2pt + green.darken(20%)),
  fill: green.lighten(88%),
)[
  #text(weight: "bold", fill: green.darken(30%))[例] #h(0.5em) #body
]
#let aside(body) = block(
  width: 100%,
  inset: 1em,
  fill: luma(235),
)[
  #emph(body)
]

#set par(first-line-indent: (amount: 2em, all: true), spacing: 1em, leading: 0.8em)
#align(center)[ #text(size: 18pt, weight: "bold")[目 $quad$ 录] ]
#v(1em)
#show outline.entry.where(level: 1): it => { v(1.2em, weak: true); strong(it) }
#outline(title: none, indent: 1.5em)
#pagebreak()

= 第一章
...
```

### 三个语义函数的用法

| 函数 | 视觉 | 用途 | 典型场景 |
|------|------|------|---------|
| `#intuition[...]` | 蓝色左边框 + 浅蓝底 | 定理/概念前的"这东西想干嘛"动机段 | 引出新概念前，先讲它要解决什么问题 |
| `#example[...]` | 绿色左边框 + 浅绿底 | 工程化/数值例子框 | "取 N=3 算给你看"、存储换算、性能估算 |
| `#aside[...]` | 灰底 + 斜体 | 旁白/易错提醒/吐槽 | "这步在干什么""新手常踩的坑"、代码设计意图 |

---

## 六、需要避免的坑

### 1. Typst 语法陷阱

| 坑 | 现象 | 解决方案 |
|----|------|---------|
| **`<` 被解释为标签** | `获奖难度 <5%` 报 `unclosed label` 错误 | 用 `\lt` 或改写为中文"低于" |
| **`>` 同理** | `大于>某某` 报错 | 用 `\gt` |
| **数学模式中多字母变量** | `$QK^T$` 报 `unknown variable: QK` | 加空格 `$Q K^T$` 或加引号 `$"QK"^T$` |
| **多字母运算符（Var、cov、max 等）** | `$Var(X)$` 被解析为 $V a r$ 乘积 | 用 `op("Var")(X)`、`op("cov")(X,Y)` |
| **`rightarrow` 不是 Typst 语法** | `$rightarrow$` 报 `unknown variable` | 用 `$arrow.r$` |
| **`infty` 不是 Typst 符号** | `$p in [1, infty]$` 报 `unknown variable: infty` | 用 `infinity`：`$p in [1, infinity]$` |
| **`lesssim` 不是 Typst 符号** | 报 `unknown variable: lesssim` | 用 `<~`（小于号加波浪） |
| **`otimes` 不是 Typst 符号** | 报 `unknown variable: otimes` | 用 Unicode 字符 `⊗`（U+2297） |
| **`success.hairspace` / `succeeds.eq` 不可用** | 报 `unknown variable` | 用 Unicode 字符 `⪰`（U+2A7F） |
| **`gray.darker` 不存在** | `fill: gray.darker` 报 `cannot access fields on type color` | 用 `luma(120)` 等数值。注意 `.darken(20%)` 方法可用，`gray.darker` 字段访问不可用 |
| **`#show table: align(center)` 语法错误** | `missing argument: body` | 必须用闭包：`#show table: it => align(center, it)` |
| **`three-line-table` 不能用作 show selector** | `only element functions can be used as selectors` | 删掉该行；`three-line-table` 内部调用 `table()`，`#show table:` 已覆盖 |
| **`#import` 不传播 `#set` / `#show` 规则** | 导入 preamble 后 heading 不编号、字体不应用 | 将 `#set` / `#show` 规则封装进函数，用 `#show: func` 应用。当前每份 `.typ` 自包含无此问题，若未来引入共享 preamble 需注意 |
| **字体名大小写** | `kaiti` 可能匹配不到系统字体 | 用 `KaiTi`（通过 `typst fonts` 确认）。`STKaiti` 也可用 |
| **代码块中的特殊字符** | 反引号或 `$` 在代码块中被解析 | 确保代码块用三反引号包裹 |
| **不要在 `.typ` 中写注释** | `//` 注释虽合法但要求不加注释 | 直接删除所有注释 |
| **`//` 注释会吞掉单行 content 的 `]`** | `rect()[// TODO]` 报 `unclosed delimiter` | 注释放在 `[ ]` 内部换行，或直接不用占位 |
| **公式 label 后缀是 `>` 不是 `}`** | `$ ... $ <eq:foo>` 误写为 `}` 报 `unclosed label` | label 语法是 `<tag>`（尖括号） |
| **居中公式结尾不要加句号** | `$ ... $ <eq:foo>` 后的 `.` 被渲染成单独一行 | 居中公式结尾不加任何标点 |
| **分数含乘积需加括号** | `a b / c` 被解析为 `(a b) / c` 但易混淆 | 分子分母含多因子一律括起：`(a b) / c`、`a / (b c)` |
| **`|X|` 在数学模式中有歧义** | `P{|X| >= t}` 嵌套场景解析错误 | 数学公式中统一用 `abs(X)` 替代 `|X|` |
| **`underset` 在 Typst 中不可用** | 报 `unknown variable: underset` | 改用纯文本描述，避免在公式中做标注 |
| **`norm(chevron.l X, v chevron.r)` 逗号被当参数分隔符** | 报 `unexpected argument` | 给内积整体加一层括号：`norm((chevron.l X, v chevron.r))` |
| **多行公式对齐** | 长公式折行报错或排版混乱 | 用 `&` 标记对齐点，`\` 换行 |
| **严禁使用破折号 `——`** | 中文写作习惯用 `——` 做破折号 | 用逗号或冒号替代，写完用 `Select-String` 检查零残留 |
| **`dot()` 是 math accent 不是二元运算** | `$dot(a, b)$` 报 `unexpected argument` | `dot` 只接受一个参数（上标点），内积用 `$op("dot")(a, b)$` |
| **十六进制/含字母的数字字面量不能进数学模式** | `$0x00821500$` 报 `unknown variable: x00821500` | 含字母的数字字面量用行内代码（反引号）包裹，不要放进 `$ ... $` |
| **`sub` 不是 Typst 函数** | `$T sub 0$` 报错 | 用下标语法 `$T_0$` |
| **`xor` 等非内置运算符** | `$a xor b$` 报 `unknown variable` | 用 `$op("xor")(a, b)$` |
| **数学 label 含连字符** | `$ ... <eq-linear> $` 中连字符被解析为减法 | label 名避免连字符，用 `<eqlinear>` 或下划线 `<eq_linear>`，或直接去掉 label |
| **`...` 在数学模式中无效** | `$a, b, ...$` 报错 | 用 `dots` 或 `dots.c`（居中省略号）：`$a, b, dots.c$` |
| **`tilde` 是 accent 不是符号** | `$x tilde y$` 报错或渲染异常 | `tilde` 是上标运算符，用 `~` 表示近似，或用 `op("tilde")` |
| **代码块中 `\\n` 渲染为双反斜杠** | `PRINTF("...\\n", ...)` 在 Typst raw block 中显示为 `\n` 字面量 | Typst raw block 是字面渲染，用单 `\n` 即可 |

### 2. 内容提取陷阱

| 坑 | 现象 | 解决方案 |
|----|------|---------|
| **PDF 大量页面是图片** | `get_text()` 返回空或只有标题 | 文本提取覆盖概念讲解，代码截图需人工查看原 PDF |
| **PPTX 无 speaker notes** | notes_slide 为空 | 正常现象，以幻灯片正文为准 |
| **PPTX 中表格/SmartArt 无法提取** | python-pptx 只能提取文本框 | 对含表格的页面人工补充 |
| **子代理返回空消息** | general 子代理可能不返回结果 | 检查文件是否实际写入（glob 确认），未写入的需手动补写 |
| **explore 子代理只读** | 不能写文件 | 仅用于提取内容，撰写工作由主代理或 general 子代理完成 |
| **Windows 控制台 GBK 编码报错** | `print()` 输出含特殊 Unicode 的 PDF 文本时报 `UnicodeEncodeError: 'gbk' codec can't encode character` | 提取脚本写入 UTF-8 文件而非直接 `print()`，再用 `read` 工具读取 |
| **提取文件名冲突** | 多个子代理共用 `extracted.txt`，后一个读到前一个的残留内容 | 每个子代理用唯一文件名（如 `extracted-<NN>.txt`），或在提取前先删除旧文件 |

### 3. 编译与批量操作陷阱

| 坑 | 现象 | 解决方案 |
|----|------|---------|
| **中文路径编码** | PowerShell 输出中文乱码 | 不影响功能，用 glob/read 工具确认文件名 |
| **`--root` 参数必须指向正确目录** | 找不到图片等资源 | 设为 `.typ` 文件所在目录 |
| **批量替换破坏文件** | 正则匹配不精确 | 先在单个文件上测试，确认后再批量 |
| **PowerShell `Set-Content` 编码** | 可能写入 BOM | 用 `-Encoding UTF8 -NoNewline` |
| **子代理写文件后未编译** | `.typ` 存在但 `.pdf` 缺失 | 主代理统一批量编译验证 |
| **长文档写入超出工具限制** | `write` 工具 JSON 被截断，文件不完整 | 改用 Python 脚本写入（`with open(path, "w", encoding="utf-8") as f: f.write(content)`），或分段写入后拼接 |

### 4. 模板一致性陷阱

| 坑 | 现象 | 解决方案 |
|----|------|---------|
| **各文件模板不统一** | 子代理可能自创格式 | 给子代理提供完整的模板头部代码，要求原样复制 |
| **字体名不统一** | 部分文件 `kaiti` 部分文件 `KaiTi` | 统一用 `KaiTi`，通过 `typst fonts` 确认。子代理可能从现有旧讲义复制 `kaiti` 而非使用模板中的 `KaiTi`，需在编译后用 `Select-String -Pattern '"kaiti"'` 检查 |
| **表格居中规则遗漏** | 部分文件有部分无 | 用 bash 批量插入 `#show table: it => align(center, it)` |
| **结尾版权行不统一** | 来源描述不一致 | 统一格式：`讲义基于 XXX 课程内容编写` |
| **破折号残留** | 子代理可能在叙事中用 `——` | 写完用 `Select-String -Pattern "——"` 检查，零残留才算完成 |

---

## 七、项目结构

```
HPC/
├── hpc101/
│   ├── source/
│   │   ├── 01-hpc-overview.pdf
│   │   ├── 02-cluster-software-hardware.pdf
│   │   ├── 03-computer-systems-for-hpc.pdf
│   │   ├── 04-vectorization-basics.pdf
│   │   ├── 05-cpp-parallel-programming.pdf
│   │   ├── 06-openmp-mpi-basics.pdf
│   │   ├── 07-profiling-basics.pdf
│   │   ├── 08-ml-fundamentals-1.pdf
│   │   ├── 09-ml-fundamentals-2.pdf
│   │   ├── 10-ml-advanced-topics.pdf
│   │   └── gpu-programming.pdf
│   └── tutorials/
│       ├── assets/
│       │   └── 11-gpu-programming/
│       │       ├── visuals.json
│       │       └── *.png
│       ├── typ/
│       │   ├── 11-gpu-programming.typ
│       │   └── 11-gpu-programming-visuals.typ
│       ├── pdf/
│       │   └── 11-gpu-programming.pdf
│       ├── 01-hpc-overview.typ/.pdf
│       ├── 02-cluster-software-hardware.typ/.pdf
│       ├── 03-computer-systems-for-hpc.typ/.pdf
│       ├── 04-vectorization-basics.typ/.pdf
│       ├── 05-cpp-parallel-programming.typ/.pdf
│       ├── 06-openmp-mpi-basics.typ/.pdf
│       ├── 07-profiling-basics.typ/.pdf
│       ├── 08-ml-fundamentals-1.typ/.pdf
│       ├── 09-ml-fundamentals-2.typ/.pdf
│       └── 10-ml-advanced-topics.typ/.pdf
├── ascend-c/
│   ├── source/
│   │   ├── 02-ascend-c-quickstart.pptx
│   │   ├── 03-ascend-c-programming-model.pptx
│   │   ├── 04-ascend-c-operator-development.pptx
│   │   ├── 05-ascend-c-debugging-tuning.pptx
│   │   └── 06-ascend-c-llm-operator-optimization.pptx
│   └── tutorials/
│       ├── assets/
│       │   └── 03-ascend-c-programming-model/
│       │       ├── visuals.json
│       │       └── *.png
│       ├── 03-ascend-c-programming-model-visuals.typ
│       ├── 02-ascend-c-quickstart.typ/.pdf
│       ├── 03-ascend-c-programming-model.typ/.pdf
│       ├── 04-ascend-c-operator-development.typ/.pdf
│       ├── 05-ascend-c-debugging-tuning.typ/.pdf
│       └── 06-ascend-c-llm-operator-optimization.typ/.pdf
├── labs/
│   ├── lab1.typ
│   └── lab1.pdf
├── assets/
│   └── image-*.png
├── tools/
│   └── tutorial-visuals/
│       ├── check_visuals.py
│       └── README.md
└── typst-tutorial-workflow.md   <- 本文件
```

---

## 八、关键经验总结

1. **定位是新手学习教辅，不是复习提纲**：每个概念都要展开动机、直觉、形式、回看，让读者脱离讲师也能读懂。
2. **动机先行**：先讲"为什么需要这个概念"，再给定义。没有动机的定义只是词典词条。
3. **代码逐行批注**：代码块不是用来展示存在的，是用来教读者看懂的。关键代码后必须配逐段解释。
4. **先验证模板再批量复制**：先写一份 `.typ`，编译通过后再批量生成其余，避免模板错误传播到所有文件。
5. **子代理分工明确**：`explore` 只做内容提取（只读），`general` 负责写文件和编译；主代理负责协调和补缺。
6. **编译是底线，不是视觉验收的终点**：文件写入成功不等于内容正确，必须先通过 `typst compile`，再把相关页面渲染成图片检查裁切、重叠和可读性。
7. **字体名用 `KaiTi`**：不是 `kaiti`，通过 `typst fonts` 确认系统可用字体。
8. **严禁使用破折号 `——`**：用逗号或冒号替代，写完用 `Select-String` 检查零残留。
9. **Typst 数学模式中多字母需加空格**：`$Q K^T$` 而非 `$QK^T$`，`$arrow.r$` 而非 `$rightarrow$`，多字母运算符用 `op("...")`。
10. **`#show` 规则语法严格**：`#show table: align(center)` 不行，必须 `#show table: it => align(center, it)`；非 element function 不能做 selector。
11. **表格居中一行搞定**：`#show table: it => align(center, it)` 同时覆盖 `#table(...)` 和 `#three-line-table[...]`（因为后者内部调用前者）。
12. **三个语义框统一使用**：`#intuition`（动机）、`#example`（例子）、`#aside`（旁白/提醒），让正文层次分明。
13. **提取脚本必须写入 UTF-8 文件**：Windows 控制台 GBK 编码会 crash，PDF/PPTX 提取一律写文件再读取。
14. **子代理提取文件名必须唯一**：多个子代理共用 `extracted.txt` 会导致后一个读到前一个的残留内容。
15. **子代理可能从旧文件复制错误**：子代理可能从现有旧讲义复制 `kaiti`（小写）而非使用模板中的 `KaiTi`，编译后需用 `-CaseSensitive` 检查。
16. **长文档写入用 Python 脚本**：`write` 工具对超长内容可能 JSON 截断，改用 `with open(path, "w", encoding="utf-8") as f: f.write(content)`。
17. **数学模式陷阱汇总**：`dot`/`tilde`/`sub` 是 accent 不是运算符；`...` 用 `dots`；十六进制字面量不进数学模式；label 名避免连字符；`\\n` 在 raw block 中用单 `\n`。
18. **技术图按用途选工具**：节点和箭头优先 Fletcher，坐标控制、硬件框图和定量图优先 CeTZ，不为统一工具而牺牲可读性。
19. **AI 图片只作装饰**：AI 图不得承载硬件结构、算法步骤、数据或实验结果，图注必须明确其装饰性质。
20. **来源链跟图片一起提交**：每张试点图片都在 `visuals.json` 中记录来源、用途、实现方式与验收状态。
21. **视觉修改必须修复再复验**：首轮渲染要主动寻找问题，修复后重新编译并检查受影响页面。
