# Lab5 Task2 性能热点定位（多尺度 Profiling 归因）

> 目标：用多尺度 profiling 定位 Lab5 task2（端到端 INT4 推理）当前 749.8 s / 0.426 tok/s
> 的性能热点，并以实测证据裁决 README §4 的四条假设，而非沿用推断。
> task1 已 100 分（ΔNLL=0.0935 < 0.16），OOM 已修复（评测能跑完），本工作的范围仅为
> **定位热点**，不包含修复实现。

## 0. 结论（TL;DR）

- **热点已定位**：decode 阶段的 fused dequant-GEMM Triton kernel（`_fused_dequant_gemm_kernel_v2`），
  nsys 测得平均 7.15 ms/launch、最大 24.3 ms、7216 次，占 GPU kernel 时间 **99.2%**；
  torch.profiler 测得 self CUDA **63.8%**。
- **decode 占总时长 189%**（BS2 并行，故 >100%），prefill 仅占 3%。每步 TPOT 中位 **4.494 s**，
  是报告最优配置（0.50 s/step）的 **~9 倍**。
- **ncu 细查该 kernel**：占用率仅 12.5–25%（寄存器+共享内存压力，grid 64 block 未填满 132 SM），
  45% 周期停顿于 L1TEX 记分牌依赖（INT4 解包/反量化加载延迟），50% 周期无可用 warp。
- **次要瓶颈**：CPU↔GPU 同步 `cudaStreamSynchronize` 占 CUDA API 时间 91.7%（55 s / 2462 次，
  单次最大 232 ms），来自 generate 主循环里每步的 `.item()`/`.tolist()`（采样取 token），
  正是 CUDA Graph 可消除的那部分开销。
- **假设裁决**：H1（CUDA Graph 缺失）确认；H2（BS2）为次要杠杆；H3（v2 kernel 未生效）**否决**；
  H4（async offload 不重叠）**否决**。

## 1. 评测环境与复现口径

| 项 | 值 |
|---|---|
| 分区 / 硬件 | `lab5`，H800 PCIe MIG 1g.10gb（10 GiB），8 CPU，24 GiB host，30 min 墙 |
| 镜像 | `harbor.clusters.zjusct.io/public/hpc101-lab5:v0.2`（OJ 等价） |
| Python | `/opt/lab5-venv/bin/python`（torch 2.13.0+cu132, triton 3.7.1） |
| 被测源码 | `~/lab5/src/hpc101_infer/`（OJ 提交件，SHA 已与 dev 一致） |
| 模型 | `~/HPC101/src/lab5/results/gemma-4-12b-gptq-cholesky-w4a16`（预量化 checkpoint） |
| 数据集 | `~/lab05_final/datasets/performance_public.jsonl`（10 条，prompt 250–2000 tok） |
| 配置 | int4_hybrid / flash / rebind_batch / async / ring_indexed / compact_active_slots / BS2 / msl=2048 / seed=0 / synchronize_metrics=False / **cuda_graph=False** |
| Profilers | `nsys 2026.2.1.0`、`ncu`（remote `/usr/local/bin`）、`torch.profiler` |
| 作业 | coarse `125975`、medium `125797`、fine `126161`（首次 `126095` 因过滤语法失败） |

## 2. Scale 1 — Coarse：端到端相位分解

作业 `125975`（`run_profile_coarse.sh`，`--scale coarse`），完整 BS2 public 集。

| 指标 | 实测 | OJ 对照 |
|---|---|---|
| wall_run | **746.621 s** | 749.796 s（误差 <0.5%，复现成功） |
| throughput | **0.4286 tok/s** | 0.426 tok/s（复现 0 分） |
| generated_tokens | 320 | 320 |
| n_decode_steps | 310 | — |
| sum_prefill_TTFT | 22.076 s | **3.0% of wall** |
| sum_decode_time | 1414.370 s | **189.4% of wall**（BS2 并行 → >100%） |
| TPOT min/med/mean/max | 4.484 / **4.494** / 4.562 / 13.061 s | 最优 0.50 → **慢 ~9×** |
| peak_alloc / reserved | 3.081 / 3.785 GiB | <10 GiB，无 OOM |

每请求 TTFT 随 prompt 长度增长（250 tok：1.21 s 冷启 3.96 s；2000 tok：2.6–3.8 s），
但 TTFT 总和仅占 3%。**749.8 s 几乎全部花在 decode**；每步 decode ~4.5 s。

证据：`coarse_phase_breakdown.json`、`jobC_coarse_phase_breakdown.out`。

## 3. Scale 2 — Medium：算子级时间分布与 CPU/GPU 重叠

作业 `125797`（`run_profile_coarse_medium.sh`）。短窗口 = 1 prefill + 12 decode（BS1 small，
torch.profiler）/ 20 decode（medium-nsys，nsys 包裹）。

### 3.1 torch.profiler key_averages（self CUDA time，单位已由 µs 换算为 s）

| 算子 | self CUDA (s) | calls | 占比 | 类别 |
|---|---|---|---|---|
| `_fused_dequant_gemm_kernel_v2` | **28.30** | 3936 | 63.8% | fused dequant-GEMM |
| `aten::copy_` | 5.20 | 19346 | 11.7% | H2D copy |
| `LAB5_WEIGHT_PREFETCH_H2D`(NVTX) | 5.16 | 576 | — | async offload 预取标记 |
| `Memcpy HtoD (Pinned->Device)` | 5.16 | 1152 | — | 实际 H2D（与上同源） |
| `aten::mm` / `nvjet_sm90...` | 0.104 | 12 | <0.3% | prefill cuBLAS fallback |
| `_flash_attn_kernel` | 0.0041 | 576 | 0.0% | attention（可忽略） |

类别合计（GPU self_time_total = 44.33 s）：fused_dequant_gemm 63.8% / other 24.4% /
h2d_copy 11.7% / attention 0.0%。

### 3.2 nsys cuda_gpu_kern_sum（全进程，含 warmup）

| kernel | Total (s) | Instances | Avg | Max |
|---|---|---|---|---|
| `_fused_dequant_gemm_kernel_v2` | **51.60** | 7216 | 7.15 ms | 24.3 ms |
| `nvjet_sm90_...`（cuBLAS prefill） | 0.19 | 22 | 8.68 ms | 8.70 ms |
| 其余 elementwise/reduce | <0.06 各 | — | — | — |

→ v2 kernel 占 GPU kernel 时间 **99.2%**，单次 ~7.2 ms。

### 3.3 nsys cuda_api_sum（CPU↔GPU 同步）

| API | Total (s) | 占比 | calls | Max |
|---|---|---|---|---|
| `cudaStreamSynchronize` | **55.12** | 91.7% | 2462 | 232 ms |
| `cudaHostAlloc` | 2.12 | 3.5% | 97 | — |
| `cudaLaunchKernel` | 1.90 | 3.2% | 99304 | — |
| `cudaMemcpyAsync` | 0.418 | 0.7% | 6866 | — |
| `cudaStreamWaitEvent` | 0.0053 | 0.0% | 2089 | — |

### 3.4 nsys memcpy 统计

`[CUDA memcpy Host-to-Device]` = 9.55 s（2628 次，125,661 MB，avg 47.8 MB，max 2013 MB）。
async offload 的权重预取确实发生（5.23 GiB decoder 权重逐层 H2D）。

### 3.5 重叠判定（H4）

- GPU active（self_time 44.33 s）> 窗口 wall（36.18 s）→ GPU 利用率 ~122%，说明
  **多 stream 并发**（copy stream 与 compute stream 在时间上重叠）。
- `cudaMemcpyAsync`（0.42 s，非阻塞）+ `cudaStreamWaitEvent`（2089 次，stream 侧排序）
  构成 async offload 机制，本身很廉价；`cudaStreamSynchronize`（55 s，**阻塞 CPU**）
  来自 generate 主循环每步的 `.item()`/`.tolist()`（engine.py:232/431/467/519/612/615），
  与 offloader 无关。
- `measure_operation` 在 `synchronize_metrics=False` 时是 no-op（`device.synchronize(enabled=False)`），
  证实 per-step 计时本身不引入同步。

证据：`medium_key_averages.txt`、`jobA_medium_nsys_stats.out`。

## 4. Scale 3 — Fine：kernel 级根因（ncu）

作业 `126161`（`run_profile_fine.sh` v2，`--kernel-name regex:_fused_dequant_gemm_kernel_v2`）。
首次作业 `126095` 因 `--kernel-name` 做**精确匹配**而 `"fused_dequant_gemm"` 不等于
`"_fused_dequant_gemm_kernel_v2"` 导致 `No kernels were profiled`；改用 `regex:` 前缀后成功。
ncu 对 2 次 launch 做 `--set full`（replay-mode kernel，每次 40 passes）。

**被采样的 launch**：grid `(1, 64, 1)` × block `(256, 1, 1)`，即小层（q_proj/k_proj 类，
M=1 decode，N 分 64 tile）。大层（gate_proj/up_proj/down_proj，N 至 ~14336）未被采样，
但 nsys 显示其单次可达 24.3 ms。

### 4.1 GPU Speed Of Light（吞吐画像）

| 指标 | min | max | avg | 判读 |
|---|---|---|---|---|
| Memory Throughput | 33.3% | 57.2% | **45.2%** | 非 DRAM 带宽瓶颈 |
| DRAM Throughput | 7.7% | 19.2% | **13.5%** | DRAM 远未饱和 → 不是访存带宽受限 |
| L1/TEX Cache Throughput | 37.1% | 65.5% | 51.3% | L1/TEX 为主 |
| L2 Cache Throughput | 6.7% | 10.5% | 8.6% | L2 低 |
| Compute (SM) Throughput | 28.3% | 44.0% | **36.1%** | SM 计算未饱和 |
| Duration (replay) | 181 µs | 220 µs | 201 µs | 单次纯 kernel 时间（小层） |

### 4.2 Occupancy

| 指标 | launch1 | launch2 | 硬件上限 |
|---|---|---|---|
| Theoretical Occupancy | 25% | **12.5%** | 100% |
| Achieved Occupancy | 22.93% | **12.49%** | — |
| Achieved Active Warps/SM | 14.68 | 7.99 | 64 |
| Block Limit Registers | 2 | **1** | — ← **寄存器压力** |
| Block Limit Shared Mem | 2 | **1** | — ← **共享内存压力** |
| Block Limit Warps | 8 | 8 | — |
| grid | (1,64,1) | (1,64,1) | H800 有 132 SM → **64 block 仅填半数 SM** |

ncu OPT 提示：**Est. Speedup 42.83%（launch1）/ 66.71%（launch2）**，限据为
"theoretical occupancy limited by required registers and shared memory"。

### 4.3 WarpStateStats（停顿根因）

| 指标 | launch1 | launch2 |
|---|---|---|
| Warp Cycles / Issued Instruction | 7.40 | 6.35 |（理想 ~1–2） |
| Avg Active Threads / Warp | 32 | 32 |

ncu OPT（**Est. Speedup 44.98%**）："each warp spends **2.9 cycles stalled waiting
for a scoreboard dependency on a L1TEX (local/global/surface/texture) operation**…
this stall type represents about **45.0% of the total average of 6.3 cycles** between
issuing two instructions."

→ 主停顿是 **L1TEX scoreboard 依赖**（等待 INT4 解包/反量化的寄存器/缓存加载完成），
属**延迟受限**而非带宽受限（与 4.1 的 DRAM 13.5% 一致）。

### 4.4 SchedulerStats

| 指标 | launch1 | 判读 |
|---|---|---|
| No Eligible | **50.43%** | 半数周期无可用 warp → 发射槽浪费 |
| Issued Warp / Scheduler | 0.50 | 每 2 周期才发射 1 条（理想 1/周期） |
| Active Warps / Scheduler | 3.67 | 上限 16 → 占用率低 |
| Eligible Warps / Scheduler | 0.96 | <1 → 几乎无备选 warp |

ncu OPT（**Est. Local Speedup 42.83%**）：occupancy 低 + stall 导致 issue slot 闲置。

### 4.5 7.2 ms vs 200 µs 的口径差异（nsys vs ncu）

- ncu `Duration` 是**单次 kernel 在隔离 replay 下的纯执行时间**（小层 ~200 µs）；
  ncu 仅采样了前 2 次 launch（小 q_proj 类层）。
- nsys `cuda_gpu_kern_sum` 的 7.15 ms 平均跨 **7216 次 launch / 328 层 × 22 step**，
  含大层（down_proj/gate_proj，N≈14336，grid 远大于 64，单次达 max 24.3 ms）。
- 故 749.8 s 的 decode 时间 = `Σ(每层每步 kernel 时间)`，大层主导；小层的低占用/高停顿
  机制同样适用于大层（同一 v2 kernel、同一寄存器/smem 限制）。
- 综上：**根因是 v2 fused kernel 自身低效**（低占用 + L1TEX 延迟停顿 + 小 grid 未填满 SM），
  而非 v1/v2 选择错误或 offload 未重叠。

## 5. 假设裁决（README §4）

| # | 假设 | 裁决 | 证据 |
|---|---|---|---|
| H1 | CUDA Graph 缺失（`capture_decode_graph` 死代码 / 无 `cuda_graph` 键） | **确认（机制修正）** | 静态：`config.py:86 cuda_graph: bool = False`，`config.yaml` 无该键；`engine.py:596 use_graph=self.config.cuda_graph` 传给 `decode_step`，因 `False` 永不取 graph 路径。`capture_decode_graph`(engine.py:296) 实为完整实现（warmup+`torch.cuda.graph(g)`+replay），非"死代码"，而是"被配置关闭"。后果即 §3.3 的 2462 次 `cudaStreamSynchronize`（55 s）。 |
| H2 | BS2 限制吞吐 | **次要杠杆** | coarse 复现 BS2=746.6 s。decode 主导且 kernel 延迟受限，BS2 仅把 328 次 launch 摊到 2 序列；BS4/8 已验证显存可行（§README 7.5 GiB），可线性提升 tokens/step，但不改变单 kernel 低占用/高停顿的本质。 |
| H3 | v2 fused kernel 未生效 | **否决** | 静态：`triton_linear.py:14` 的 `_fused_dequant_gemm_kernel`(v1) **从未被调用**（死代码），v1/v2 两个类的 forward 均 dispatch `_fused_dequant_gemm_kernel_v2`（行 126、302）；`HybridQuantizedLinear` 继承 v1 类 → decode(M≤64) 走 v2。实测：nsys/torch.profiler 均只见 `_fused_dequant_gemm_kernel_v2`；ncu 按名 profile 该 kernel 成功（§4）。**v2 确为当前 decode kernel**。遗留问题是 v2 自身低效（§4.2–4.4），而非版本选错。 |
| H4 | async offload copy/compute 不重叠 | **否决** | §3.5：GPU 利用率 ~122%、`cudaMemcpyAsync` 仅 0.42 s 非阻塞、`cudaStreamWaitEvent` stream 侧排序。55 s 同步来自采样循环 `.item()`，非 offloader。offload 重叠正常。 |

## 6. 热点归因（一句话）

> **749.8 s 中 ~189% 花在 decode；decode 的 GPU kernel 时间 99.2% 是 `_fused_dequant_gemm_kernel_v2`
>（nsys 7.2 ms/launch × 328/step，大层单次达 24 ms）。ncu 细查该 kernel：占用率仅 12.5–25%
>（寄存器+共享内存压力，grid 64 block 未填满 132 SM），45% 周期停顿于 L1TEX 记分牌依赖
>（INT4 解包/反量化加载延迟），50% 周期无可用 warp。即 v2 kernel **自身低效**（延迟受限），
> 并非 README §4 推断的 "v1/v2 选错"；叠加 55 s 的 CPU↔GPU 同步（采样循环 `.item()`，
> CUDA Graph 缺失所致）。**

## 7. 可复现命令

```bash
# 远端（zju-hpc-lab2，Windows OpenSSH）
cd ~/lab5-fix-dev
# coarse（~12.5 min）
hpc submit -p lab5 -g 1 -c 8 -m 24Gi -t 30m -n lab5-prof-coarse \
  --chdir ~/lab5-fix-dev bash run_profile_coarse.sh
# medium（torch.profiler + nsys，~5 min）
hpc submit -p lab5 -g 1 -c 8 -m 24Gi -t 30m -n lab5-prof-A \
  --chdir ~/lab5-fix-dev bash run_profile_coarse_medium.sh
# fine（ncu，~8 min；--kernel-name 必须用 regex: 前缀，精确匹配会失败）
hpc submit -p lab5 -g 1 -c 8 -m 24Gi -t 30m -n lab5-prof-fine \
  --chdir ~/lab5-fix-dev bash run_profile_fine.sh
# 产物 ~/lab5-fix-dev/profile-out/，日志 ~/lab5-fix-dev/j<id>.out
```

## 8. 产物索引

| 文件 | 内容 |
|---|---|
| `profile_multiscale.py` | 多尺度 profiling 主脚本（coarse/medium/medium-nsys/fine） |
| `run_profile_coarse.sh` / `run_profile_coarse_medium.sh` / `run_profile_fine.sh` | hpc 作业包装 |
| `coarse_phase_breakdown.json` | Scale 1 相位分解（含每请求 TTFT/TPOT/总时） |
| `medium_key_averages.txt` | Scale 2 torch.profiler key_averages 表 |
| `jobA_medium_nsys_stats.out` | Scale 2 nsys `--stats` 文本报告（cuda_gpu_kern_sum / cuda_api_sum / memcpy） |
| `jobC_coarse_phase_breakdown.out` | Scale 1 作业日志 |
| `fine_ncu_full_report.txt` | Scale 3 ncu 全文报告（SpeedOfLight/Occupancy/WarpState/Scheduler） |
| `jobB_fine_ncu.out` | Scale 3 作业日志（v2，成功） |
| `jobB_fine_ncu_attempt1.out` | Scale 3 首次作业（exact-name 过滤失败，保留作弯路记录） |
| 远端大文件（不入库） | `~/lab5-fix-dev/profile-out/{medium.nsys-rep, medium.sqlite, medium_chrome_trace.json, fine.ncu-rep}` |
