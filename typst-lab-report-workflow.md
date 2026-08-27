# Typst Lab Report Workflow

This workflow is for HPC lab reports in this workspace. It assumes the report mixes code changes, remote benchmarking, command output screenshots, and Typst compilation.

The key lesson from Lab 2 is that report writing has two different phases:

- Step 1: build the report framework before screenshots exist.
- Step 2: after the user finishes screenshots, integrate them, align the text and figures, and make the report read like a real experiment rather than a command log.

Keep these phases separate. Step 1 should make screenshot collection easy. Step 2 should make the final report coherent.

## Step 1: Write the Framework

In Step 1, do not try to polish every paragraph. The goal is to create a complete scaffold that tells the user what evidence to collect.

### 1. Establish the Experiment Story

Start from the causal chain, not from screenshots.

- Keep each optimization iteration in the same shape: hypothesis, implementation, verification, analysis.
- Put profiling evidence in the hypothesis section of the iteration it motivates.
- Avoid a separate profiling chapter unless profiling itself is the experiment target.
- Keep old iterations stable when the user asks for them to remain unchanged.
- Add only the minimum context needed to explain why the next optimization exists.
- Do not invent benchmark output. Use placeholders when evidence is not collected yet.

For a performance lab, a good iteration usually answers:

- What bottleneck did we observe?
- What change targets that bottleneck?
- How do we prove the change really ran?
- What changed in time, speedup, RMSE, IPC, cache misses, or hotspots?
- What limitation remains?

For Lab 2, the final structure became:

- Iteration 1: VNNI single-token kernel.
- Iteration 2: expert grouping.
- Iteration 3: AMX data layout and batched expert kernel.
- Iteration 4: end-to-end pipeline and runtime dispatch.
- Iteration 5: lookup activation and input cache.
- Bonus: RISC-V RVV and SpaceMiT IME implementation.

### 2. Reserve Screenshot Positions

When screenshots are not ready, place explicit screenshot reservations near the text they will support. Do not dump all screenshots at the end.

Use a consistent placeholder style such as:

```typst
#screenshot-placeholder("assets/lab2/lab2-30.png", [S1 的 `perf stat` 输出])
```

or, if the helper is not available yet, leave a visible Typst comment-free marker in prose:

```typst
#screenshot("assets/lab2/lab2-30.png", [S1 的 `perf stat` 输出])
```

The placeholder caption should already explain what the screenshot must show:

- scenario name, for example S1, S2, S3, S4;
- command type, for example correctness, `perf stat`, `perf report`, `objdump`;
- expected evidence, for example AMX instruction, RVV instruction, speedup, thread scan.

Name images sequentially in the lab asset folder:

```text
assets/lab2/lab2-30.png
assets/lab2/lab2-31.png
assets/lab2/lab2-32.png
```

### 3. Write Screenshot Commands

Step 1 must include the exact commands the user should run before taking screenshots. Put the command block immediately before or near the reserved screenshot.

For benchmark screenshots:

```bash
cmake --build build -j "$(nproc)"
taskset -c 0 ./build/lab2 1 256 128 16 4 10
taskset -c 0 ./build/lab2 1 1024 512 16 4 10
taskset -c 0 ./build/lab2 128 256 128 16 4 10
taskset -c 0 ./build/lab2 1024 512 128 512 2 10
```

For threaded experiments, print labels so screenshots are self-explanatory:

```bash
for n in 1 2 4 8 12 16 24; do
  echo THREADS=$n S3
  taskset -c 0-23 env MOE_NUM_THREADS=$n \
    ./build/lab2 128 256 128 16 4 10
  echo THREADS=$n S4
  taskset -c 0-23 env MOE_NUM_THREADS=$n \
    ./build/lab2 1024 512 128 512 2 10
done
```

For `perf stat`:

```bash
perf stat -e cycles,instructions,cache-misses,branches,branch-misses \
  taskset -c 0 ./build/lab2 1 256 128 16 4 10
```

For low-buffer `perf record`:

```bash
perf record -m 1 -F 49 --output /tmp/lab2-s3.data -- \
  taskset -c 0 ./build/lab2 128 256 128 16 4 50
perf report --stdio --no-children --sort symbol \
  --percent-limit 2 --input /tmp/lab2-s3.data | head -n 60
```

For instruction-level evidence:

```bash
objdump -d build/CMakeFiles/student.dir/student/moe_opt.cpp.o \
  | grep -E 'tileloadd|tdpbssd|tilestored' | head -n 12
```

For RISC-V RVV or IME evidence:

```bash
objdump -d build-riscv/CMakeFiles/student.dir/student/moe_opt.cpp.o \
  | grep -E 'e210112b|vmadot|vsetvli' | head -n 30
```

### 4. Mark Unknown Numbers Explicitly

In Step 1, tables may use placeholders, but the placeholders must be easy to replace:

```typst
table.header([场景], [迭代数], [Baseline], [Optimized], [Speedup]),
[S4], [10], [`待截图`], [`待截图`], [待截图],
```

Do not write approximate final claims before the screenshots exist. It is better to write:

```typst
截图完成后在这里比较 S4 单线程与多线程的 optimized time。
```

than to guess a speedup.

## Step 2: Integrate Screenshots and Polish

Step 2 begins only after the user has completed the screenshots. The goal changes: now the report must be internally consistent, visually aligned, and readable.

### 1. Read Every Screenshot

For each screenshot, identify:

- the exact command that produced it;
- scenario and iteration count;
- baseline time;
- optimized time;
- speedup;
- RMSE or correctness line;
- relevant counters such as IPC, cache misses, or hotspot percentages;
- instruction evidence such as `tdpbssd`, `vsetvli`, or `.word 0xe210112b`.

Then update every matching location:

- table cells;
- prose numbers;
- figure captions;
- analysis paragraphs;
- final summary table;
- any earlier or later paragraph that compares against the same result.

If two runs differ, explain why. For example, an uninstrumented run can be faster than a `perf record` run because sampling adds overhead. A cold low-iteration run can show less cache benefit than a long repeated-input run.

### 2. Replace Placeholders With Real Figures

Use the report helper for screenshots:

```typst
#screenshot("assets/lab2/lab2-30.png", [S1 的 `perf stat` 输出])
```

Do not leave visible `#screenshot-placeholder` calls in the final report after screenshots are available.

Check all image references:

- path is relative to the Typst root;
- file exists under `assets/lab<N>/`;
- caption describes what the screenshot proves;
- screenshot appears near the paragraph that discusses it.

### 3. Align Text and Figures

After screenshots are inserted, reread the surrounding section as a reader.

- Put the figure immediately after the command block or after the first paragraph that needs the evidence.
- Do not make the reader scroll far away to verify a number.
- If a figure contains four terminal outputs, summarize the exact four results below it.
- If a table already contains all numbers, the prose should explain the trend, not repeat every cell.
- If an image is too tall or dense, split it into two images or show only the relevant terminal region.
- Keep captions factual: what the figure proves, not a vague label like "运行结果".

Good caption:

```typst
caption: [RISC-V S4 在 1、2、4、8 线程下的正确性与 optimized time]
```

Weak caption:

```typst
caption: [测试截图]
```

### 4. Add Human Readability

After numbers are correct, make the report sound like an experiment done by a person.

Do:

- explain why a result is plausible;
- mention when a result is worse than expected;
- distinguish cold computation from cache-hit speedup;
- say what was tried and why it was kept or rejected;
- connect each iteration to the next bottleneck.

Avoid:

- bare command dumps without interpretation;
- table-only sections;
- claiming a tool proves more than it does;
- overconfident explanations for noisy timing;
- generic phrases such as "效果很好" without evidence.

Example of useful analysis:

```text
S4 在移植多线程 Router 分派后，10 次迭代 optimized time 从上一版 `13.4647 s` 降到 `8.7605 s`。收益仍低于 x86 主线，原因是当前 RISC-V 版本只并行 Router、Top-K 与输入量化准备阶段，专家 grouped IME 阶段仍保持单线程，以避免多线程共享 scratch 和输出累加带来的同步开销。
```

### 5. Remove Stale Text

Screenshots often invalidate earlier draft wording. Search for stale claims before finishing:

```powershell
Select-String -Path .\labs\lab2.typ -Pattern "待截图|placeholder|没有移植|标量实现|旧版|TODO"
```

Also search for old tool names that are no longer part of the story:

```powershell
Select-String -Path .\labs\lab2.typ -Pattern "VTune|perf 失败|临时"
```

If the report changed from "not implemented" to "implemented", update both the implementation section and the analysis section. Do not only change the table.

## Validation Rules

### Benchmark Integrity

Never invent output. Every number must come from:

- a fresh remote run;
- an existing screenshot;
- a verified command log.

If a benchmark was run with a non-default iteration count, state the count in the table and prose.

### Typst Compile

After every meaningful edit, compile from the repository root:

```powershell
typst compile --root . .\labs\lab2.typ .\labs\lab2.pdf
```

If Typst is installed through WinGet in this workspace, this command is often needed:

```powershell
& "$env:LOCALAPPDATA\Microsoft\WinGet\Links\typst.exe" compile --root . .\labs\lab2.typ .\labs\lab2.pdf
```

### Project-Specific Text Check

The lab report must not contain the forbidden em dash:

```powershell
Select-String -Path .\labs\lab2.typ -Pattern "——"
```

Zero output is expected.

### Final Placeholder Check

Before finalizing, check for unfinished placeholders:

```powershell
Select-String -Path .\labs\lab2.typ -Pattern "screenshot-placeholder|待截图|TODO"
```

The final report should have:

- no missing image errors;
- no forbidden em dash;
- no visible screenshot placeholders;
- no stale benchmark numbers;
- captions that match the screenshots;
- prose that explains the results instead of merely listing them.
