#import "@preview/cetz:0.5.2"
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

#let visual-source(body) = align(center, text(size: 8pt, fill: luma(105), body))

#let aicore-data-path() = figure(
  stack(
    dir: ttb,
    spacing: 6pt,
    cetz.canvas({
      import cetz.draw: *
      rect((0, 1.0), (1.8, 3.0), radius: 4pt, fill: rgb("e8eef4"), stroke: rgb("526b7a"))
      content((0.9, 2.0), text(size: 9pt, weight: "bold", [HBM]), anchor: "center")
      rect((2.7, 1.0), (4.5, 3.0), radius: 4pt, fill: rgb("fff0d8"), stroke: rgb("b26b16"))
      content((3.6, 2.0), align(center, stack(dir: ttb, spacing: 1pt, text(size: 9pt, weight: "bold", [DMA]), text(size: 8pt, [搬入 / 搬出]))), anchor: "center")
      rect((5.4, 0.3), (8.1, 3.7), radius: 4pt, fill: rgb("e9f6ef"), stroke: rgb("237a57"))
      content((6.75, 3.35), text(size: 9pt, weight: "bold", [Local Memory]), anchor: "center")
      rect((5.75, 0.65), (7.75, 1.55), radius: 3pt, fill: white, stroke: rgb("4b9b78"))
      content((6.75, 1.1), [UB], anchor: "center")
      rect((5.75, 1.85), (7.75, 3.0), radius: 3pt, fill: white, stroke: rgb("4b9b78"))
      content((6.75, 2.43), align(center, stack(dir: ttb, spacing: 0pt, text(size: 8pt, [L1 / L0A]), text(size: 8pt, [L0B / L0C]))), anchor: "center")
      rect((8.9, 0.3), (11.8, 1.75), radius: 4pt, fill: rgb("e7f0ff"), stroke: rgb("315f85"))
      content((10.35, 1.03), align(center, stack(dir: ttb, spacing: 1pt, text(size: 9pt, weight: "bold", [Vector Unit]), text(size: 8pt, [向量计算]))), anchor: "center")
      rect((8.9, 2.25), (11.8, 3.7), radius: 4pt, fill: rgb("efe8ff"), stroke: rgb("6f4ba0"))
      content((10.35, 2.98), align(center, stack(dir: ttb, spacing: 1pt, text(size: 9pt, weight: "bold", [Cube Unit]), text(size: 8pt, [矩阵计算]))), anchor: "center")
      rect((2.7, 4.35), (8.1, 5.35), radius: 4pt, fill: rgb("f4f4f4"), stroke: luma(110))
      content((5.4, 4.85), text(size: 8pt, [*Scalar Unit*：控制、发射、地址计算]), anchor: "center")
      line((1.8, 2.0), (2.7, 2.0), mark: (end: ">"), stroke: 1pt + rgb("b26b16"))
      line((4.5, 2.0), (5.4, 2.0), mark: (end: ">"), stroke: 1pt + rgb("b26b16"))
      line((8.1, 1.1), (8.9, 1.1), mark: (end: ">"), stroke: 1pt + rgb("315f85"))
      line((8.1, 2.95), (8.9, 2.95), mark: (end: ">"), stroke: 1pt + rgb("6f4ba0"))
      line((8.9, 1.45), (8.1, 1.45), mark: (end: ">"), stroke: 0.8pt + rgb("4b9b78"))
      line((8.9, 3.3), (8.1, 3.3), mark: (end: ">"), stroke: 0.8pt + rgb("4b9b78"))
      line((5.4, 4.35), (5.4, 3.7), mark: (end: ">"), stroke: 0.8pt + luma(90))
    }),
    visual-source([依据课程原稿第 5 页的 910B 架构抽象重绘；CeTZ 0.5.2]),
  ),
  caption: [AI Core 的计算单元、搬运单元与数据通路],
)

#let pipeline-stages() = figure(
  stack(
    dir: ttb,
    spacing: 6pt,
    align(center, scale(60%, reflow: true, diagram(
      cell-size: 12mm,
      node-stroke: 0.7pt + rgb("237a57"),
      node-inset: 6pt,
      edge-stroke: 0.9pt + luma(75),
      node((0, 1), align(center, stack(dir: ttb, spacing: 0pt, [*输入*], [Progress 1..n])), fill: rgb("eef2f5"), corner-radius: 4pt),
      node((2.2, 1), align(center, stack(dir: ttb, spacing: 0pt, [*CopyIn*], [Stage 1])), fill: rgb("dff3ea"), corner-radius: 4pt),
      node((4.4, 1), align(center, stack(dir: ttb, spacing: 0pt, [*Compute*], [Stage 2])), fill: rgb("e7f0ff"), corner-radius: 4pt),
      node((6.6, 1), align(center, stack(dir: ttb, spacing: 0pt, [*CopyOut*], [Stage 3])), fill: rgb("fff0d8"), corner-radius: 4pt),
      node((8.8, 1), align(center, stack(dir: ttb, spacing: 0pt, [*输出*], [Progress 1..n])), fill: rgb("eef2f5"), corner-radius: 4pt),
      edge((0, 1), (2.2, 1), "-|>"),
      edge((2.2, 1), (4.4, 1), "-|>"),
      edge((4.4, 1), (6.6, 1), "-|>"),
      edge((6.6, 1), (8.8, 1), "-|>"),
    ))),
    visual-source([依据课程原稿第 8 至 9 页重绘；Fletcher 0.5.8]),
  ),
  caption: [同一切片串行依赖、不同切片流水并行],
)
