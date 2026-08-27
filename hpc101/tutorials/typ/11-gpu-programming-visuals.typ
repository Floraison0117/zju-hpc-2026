#import "@preview/cetz:0.5.2"
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

#let visual-source(body) = align(center, text(size: 8pt, fill: luma(105), body))

#let gpu-execution-hierarchy() = figure(
  stack(
    dir: ttb,
    spacing: 6pt,
    align(center, scale(76%, reflow: true, diagram(
      cell-size: 11mm,
      node-stroke: 0.7pt + rgb("315f85"),
      node-inset: 5pt,
      edge-stroke: 0.8pt + luma(80),
      node((0, 1.5), align(center, stack(dir: ttb, spacing: 0pt, [*Grid*], [Kernel])), fill: rgb("dcecff"), corner-radius: 4pt),
      node((2.2, 0.5), align(center, stack(dir: ttb, spacing: 0pt, [*Block 0*], [SM 0])), fill: rgb("e8f3ff"), corner-radius: 4pt),
      node((2.2, 2.5), align(center, stack(dir: ttb, spacing: 0pt, [*Block 1*], [SM 1])), fill: rgb("e8f3ff"), corner-radius: 4pt),
      node((4.5, 0.5), align(center, stack(dir: ttb, spacing: 0pt, [*Warp 0*], [32 threads])), fill: rgb("fff0d9"), corner-radius: 4pt),
      node((4.5, 2.5), align(center, stack(dir: ttb, spacing: 0pt, [*Warp 1*], [32 threads])), fill: rgb("fff0d9"), corner-radius: 4pt),
      node((6.7, 1.5), align(center, stack(dir: ttb, spacing: 0pt, [*Thread*], [Registers])), fill: rgb("e4f5ea"), corner-radius: 4pt),
      edge((0, 1.5), (2.2, 0.5), "-|>"),
      edge((0, 1.5), (2.2, 2.5), "-|>"),
      edge((2.2, 0.5), (4.5, 0.5), "-|>"),
      edge((2.2, 2.5), (4.5, 2.5), "-|>"),
      edge((4.5, 0.5), (6.7, 1.5), "-|>"),
      edge((4.5, 2.5), (6.7, 1.5), "-|>"),
    ))),
    visual-source([依据课程原稿第 4 页重绘；Fletcher 0.5.8]),
  ),
  caption: [CUDA 执行层次与硬件调度关系],
)

#let coalescing-bars() = figure(
  stack(
    dir: ttb,
    spacing: 6pt,
    cetz.canvas({
      import cetz.draw: *
      set-style(stroke: 0.7pt + luma(90))
      line((0.8, 0.5), (9.5, 0.5))
      line((0.8, 0.5), (0.8, 4.8))
      for y in range(0, 5) {
        line((0.65, 0.5 + y), (0.8, 0.5 + y))
        content((0.5, 0.5 + y), [#text(size: 7.5pt)[#(y * 25)%]], anchor: "east")
      }
      rect((2.0, 0.5), (4.2, 4.5), fill: rgb("5aa9e6"), stroke: rgb("2d6f9f"), radius: 3pt)
      rect((6.1, 0.5), (8.3, 1.0), fill: rgb("f2aa4c"), stroke: rgb("a96619"), radius: 3pt)
      content((3.1, 4.18), [#text(fill: white, weight: "bold")[100%]], anchor: "center")
      content((7.2, 1.25), [#text(weight: "bold")[12.5%]], anchor: "south")
      content((3.1, 0.15), [连续访问\4 个 32B sectors], anchor: "north")
      content((7.2, 0.15), [跨步访问\32 个 32B sectors], anchor: "north")
    }),
    visual-source([数值由课程原稿第 10 页示例推导；CeTZ 0.5.2]),
  ),
  caption: [Warp 读取 32 个 `int` 时的有效带宽比例],
)
