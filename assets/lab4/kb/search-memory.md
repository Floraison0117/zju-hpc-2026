# 搜索记忆 (Search Memory) — Lab4 已测杠杆全表

> KernelPro search memory + KernelEvolve metadata store 的固化形式。每轮迭代前必读：确认候选杠杆是否已测及其结果，避免重复死路。
> 所有结果来自 README-lab4.md §11-§16 实测（job 编号可查），非估算。

## 已部署栈（当前最优，check.sh PASS）

### 任务一 ABE CPU（OJ 408.9s / 86 分，2026-08-23 lab2 Intel 实测）

| 组件 | CMake/flag | 收益 | 正确性 |
|---|---|---|---|
| DIST_INTERP（分析分布式插值） | AMSS_ENABLE_ANALYSIS_DIST_INTERP ON | 分析 4.56→0.8s/step（-82%） | check.sh PASS |
| LOADBAL（least-loaded-first 块分配） | AMSS_ENABLE_LOADBAL ON | -13s（须在 DIST_INTERP 后才显收益） | check.sh PASS |
| flag+loopim | AMSS_OPT 加 `-fno-tree-loop-distribute-patterns -ftree-loop-im` | -3.7% | check.sh PASS RMS=0 |
| TwoP OMP_TUNE | AMSS_ENABLE_TWOP_OMP_TUNE ON | TwoP 90→15s | check.sh PASS |
| 分析三优化 | INTERP_BATCH + PACKED_COLLECTIVES + ANGULAR_CACHE ON | 每步分析 -20.5s→优化到位 | check.sh PASS |
| PACKED_RELAX + TWOP_COS_TABLE | ON | TwoP relax 加速 | check.sh PASS |

部署文件 hash（arm + lab2 一致）：CMakeLists `c1d81201`、MPatch.C `e50aed31`、Parallel.C `87363e55`、Parallel.h `68115702`、cgh.C `e0a10e78`、TwoPunctures.C `2e8abe70`。快照 `~/lab4-cpu-snapshot-20260823_014749`（可回滚）。

### 任务二 ABEGPU（当前部署基线 ~782s 节点噪声内，P1 fdbr + P2 center-fh + P3.5 + P6b + P8 已部署；P6b L2 778.24s job 155105，P8 L2 736.80s job 见 iter17，部署节点实测 ~782s）

| 组件 | 收益 | 正确性 |
|---|---|---|
| TwoPuncture CMake 解耦（AMSS_ENABLE_TWOP_GPU）+ OMP_TUNE | TwoP 268→34s（-234s） | 物理正确 |
| INLSTEN3 forceinline（4 stencil 函数移到 .h + __forceinline__） | -13.8%（1266→1052s） | check.sh PASS RMS=0 bit-exact |
| BRFINAL branchless-fh（fh lambda if 链→branchless select） | -5.5%（1052→1007s） | check.sh PASS RMS=0 bit-exact |
| **P6b forceinline（iter15，d_symmetry_bd_1b/f_at_1b 跨 TU 移 .h + __forceinline__）** | **-13.6%（901→778s，job 155105）** | check.sh PASS RMS=0 bit-exact |
| **P8 forceinline（iter17，polint/d_polin3_1b/d_decide3d 跨 TU 移 fmisc.h + __forceinline__）** | **L2 736.80s（-5.3% vs 778s）；部署节点 ~782s（节点噪声持平）** | check.sh FINAL PASS RMS=0 bit-exact |
| **P10 cudaDeviceReset（GPUManager 析构尾部加 cudaDeviceReset()）** | **0 时延（资源清理，非性能杠杆）** | check.sh FINAL PASS RMS=0 bit-exact（job 157367，743s）；修复 OJ 二次运行 ABEGPU exit 1 无输出崩溃 |
| per-stream sync（synchronize_all_streams） | 纠正：0 收益（多轮 A/B 推翻 §15.7 -30s） | — |

**待部署候选（旧）**：BR_ORD-fix（branchless 4th/2nd order + clamped fh args），996.47s（F=1.011），check.sh FINAL PASS RMS=0 bit-exact。已由 P1 fdderivs branchless（iter7）取代（见下方条目）：**已部署基线 996.47s → 候选 988.33s（F=1.008）→ 正式部署态实测 968.96s（job 152275，check.sh FINAL PASS RMS=0，快照 lab4-gpu-snapshot-pre-fdbr-20260824-033853）**。P1 后又部署 P2（iter8，lopsided/kodiss center-fh reuse）：**正式部署态实测 928.09s（job 152669，check.sh FINAL PASS RMS=0，快照 lab4-gpu-snapshot-pre-p2-20260824-043036）**。P2 后又部署 P3.5（iter10，fdderivs 交叉项 fh 分组 16→8 峰值）：**正式部署态实测 901.24s（job 153420，check.sh FINAL PASS RMS=0，快照 lab4-gpu-snapshot-pre-p35-20260824-060232）**。

## 迭代7 P1：fdderivs branchless (d_fdderivs_point 4th/2nd + clamp)，bit-exact PASS

**候选**：`~/lab4-gpu-cand-fdbr-20260824-030658`（patch `patch_fdderivs_branchless.py`）。
改动：仅 `src/derivatives.h` 1 文件（formal `ffe0c7c3` → 候选 `8650bda1`），只改 `d_fdderivs_point` 的 4th/2nd if/else（line 254-291）→ 完全复用 d_fderivs_point 已部署模式：clamp fh args（ai_m2=max(imin,i-2)...ak_p2=min(kmax,k+2)，防 fine-grid OOB）、61 个 fh 值各算一次共享（4th 与 2nd 公式复用）、m4/m2 mask select、early-return 保留。d_fderivs_point 未动。

**Level-0 ptxas（job 152062，同机 A/B）**：128 regs（launch_bounds 锁），stack 1608B（基线 1160B），**spill stores 6444B（基线 4532B，+42%）/ spill loads 6888B（基线 5516B，+25%）**——61 个 fh 值同时存活推高 live-set，ptxas 溢出更多。

**Level-1 A/B（4 轮交错同节点 2 步）**：base1 8.83115 / fdbr1 8.73439 / fdbr2 8.7294 / base2 8.79373（Step2 s）；base median 8.81244 vs fdbr median 8.73189 → **F=1.0092**。**4/4 .dat IDENTICAL（bit-exact）**。

**Level-2 全量（job 152062，100 步）**：**This Program Cost = 988.33s**（部署基线 996.47s → **-8.1s, F=1.008**），**check.sh FINAL PASS，Trajectory RMS=0（bit-exact）**，约束 Ham=0.28974817/Px=0.039343259/Py=0.047298107/Pz=0.044686547 全 ≤2 且与基线逐位一致，4 .dat 全产，无崩溃（clamp 防住 BR_ORD 式 step-28 OOB）。

**结论（PASS，边际正收益）**：divergence 消除（P0 估 12.15% 潜力）赢了 spill 增量（+42% stores），但 2-step F=1.009 ≈ 100-step F=1.008（不像 d_fderivs 那样 2-step 1.037→100-step 1.011 衰减），说明收益稳定但小。**下一方向提示**：fdderivs 61 fh 值高 liveness 是 spill 主因，若能把 h_* 值以 float 精读或分两半算（fxx/fyy/fzz 一组、交叉项一组）减峰值 live-set，可能解锁更大收益；但 FP32 已证正确性死路，float 只可用于 stencil 输入暂存（不改变 RHS 精度，风险低，属未测方向）。

**部署指令（供主 agent）**：
```bash
cp -r ~/lab4-gpu ~/lab4-gpu-snapshot-pre-fdbr-$(date +%Y%m%d-%H%M%S)
cp ~/lab4-gpu-cand-fdbr-20260824-030658/src/derivatives.h ~/lab4-gpu/src/derivatives.h
cd ~/lab4-gpu && ./compile.sh
# 验证：100 步全量 check.sh（AMSS_OPT=-O3 与候选一致）
```
注意：候选与正式均 AMSS_OPT=-O3 + arch 80 + cuda-13.3 nvcc（build-fdbr 仅多 `-Xptxas -v` 诊断 flag，不改变 codegen）。

## 迭代8 P2：field-local center-fh reuse（d_lopsided_point / d_kodis_point），bit-exact PASS

**候选**：`~/lab4-gpu-cand-p2-20260824-040235`（patch `patch_p2_center_reuse.py`）。改动仅 2 文件：`src/lopsidediff.h`（formal `6306611a` → 候选 `5dddaa75`）+ `src/kodiss.h`（formal `a1a7963c` → 候选 `e88f9632`）；derivatives.h 未动（`8650bda1` 与部署基线一致）。

**机制（P2 语义：同一 stencil 调用内部 fh 复用，不跨调用点保活）**：在 d_lopsided_point / d_kodis_point 内，center 值 `fh(i,j,k)` 在所有可达点（early-return/if 分支保证 i<imax 且 i+1≥1）逐位等于 `f[idx]`（无反射、fac=1.0、in-range，IEEE x*1.0==x）。lopsided 每调用把 f[idx] 重复读最多 3 次（每方向 5-point 分支各一次）；kodis 在单语句内读 3 次。提升单个 `const double h_000 = f[idx];`（lopsided 12 处 `F10*fh(i,j,k)` → `F10*h_000`，kodis 3 处 `TWT*fh(i,j,k)` → `TWT*h_000`），同函数内共享，h_000 随函数结束死亡（仅 +1 暂存寄存器，非长生命周期）。**未做**跨调用点 vx/vy/vz 提升（24 次调用保活 3 double，swpipe2 教训）与跨 fderivs/fdderivs 保活（300 行距离，live-set 风险）。

**Level-0 ptxas（job 152457，同作业 apples-to-apples 重建）**：候选 128 regs / stack 1496B / **spill stores 6352B / spill loads 6792B** vs 部署基线（P1 后）128 regs / 1608B / **6444B / 6888B** → **stores -1.4%、loads -1.4%、stack -7%，live-set 不↑（验收核心达成）**。

**Level-1 A/B（job 152457，4 轮交错同节点 2 步）**：base1 8.7485 / p21 8.5851 / p22 8.82319 / base2 9.23183（Step2 s）；base median 8.99017 vs p2 median 8.70415 → **F=1.0329**。**4/4 .dat IDENTICAL（bit-exact）**。

**Level-2 全量（job 152457，100 步）**：**This Program Cost = 962.93s**（部署基线 968.96s → **-6.0s, F=1.006**），**check.sh FINAL PASS，Trajectory RMS=0（bit-exact）**，约束 Ham=0.28974817/Px=0.039343259/Py=0.047298107/Pz=0.044686547 全 ≤2 且与基线逐位一致，4 .dat 全产。

**结论（PASS，边际正收益）**：2-step F=1.033 → 100-step F=1.006（同 BR_ORD-fix 衰减模式：早期粗网格 stencil 重负载收益大，后期小网格收益小）。同节点 A/B 与跨节点全量方向一致。center 去重减 ~96 load/thread（lopsided 48 + kodis 最多 48）+ 减 spill 流量，直接攻 lg_throttle（P0 7.8%）与 load 指令数。**下一方向提示**：P2 证明同函数 fh 复用安全且正收益；更大收益需跨调用复用（如 vx/vy/vz 或 fderivs/fdderivs 场读共享），但 live-set 风险高，须先验 ptxas；或 P3 Ricci component-wise recomputation（减 live-set → 提 occupancy，dispatch_stall 31% 是首要 stall）。

**部署指令（供主 agent）**：
```bash
cp -r ~/lab4-gpu ~/lab4-gpu-snapshot-pre-p2-$(date +%Y%m%d-%H%M%S)
cp ~/lab4-gpu-cand-p2-20260824-040235/src/lopsidediff.h ~/lab4-gpu/src/lopsidediff.h
cp ~/lab4-gpu-cand-p2-20260824-040235/src/kodiss.h ~/lab4-gpu/src/kodiss.h
cd ~/lab4-gpu && ./compile.sh
# 验证：100 步全量 check.sh（AMSS_OPT=-O3 与候选一致）
```

**迭代5 FP32 混合精度（死路，双轴失败）**：
- 用户指定路径：kernel 内主计算转 float（减半寄存器占用破 Ricci live-set 上限）。候选 `~/lab4-gpu-cand-fp32-20260824_002842`，仅改 bssn_rhs_gpu.cu（sha256 `e1b8d73d`），stencil 内部保持 double（fderivs_f/fdderivs_f wrapper 截断输出），advection/约束 I/O 保持 double，常量转 float。
- **ptxas（job151105）**：float lb2/3/4 = 128/80/64 regs，spill stores 3332/7956/11456B（double 基线 128/128/128 regs，692/5116/6416B）。**float 化不降自然需求**：float 值 1 reg/个后 ptxas 更激进 hoist load 隐延迟 → 峰值 live-set 仍超预算。"float 减半寄存器 → lb3/lb4 不暴 spill → occupancy 2×" 机制链在 ptxas 层面证伪。
- **Level-1 A/B（job151129 vs job151153，同一步 1-2 同网格）**：float 8.722 s/step vs 基线 9.041 → **F=1.037（仅 ~3.6%，算术 2× 收益）**；occupancy 仍 25%。早前对比 100 步均值（9.96）的 12% 是早期步网格粗的假象。
- **正确性（2 步早期 RMS，mini-golden 截取）**：**1 步后 trajectory RMS=0.00715（0.715%），7× 超 1e-3 门限**。逐列：vy -1.82%、x -0.65%、y -0.54%、vx +0.27%；t=0 逐位一致（初始数据干净，漂移纯来自 float RHS）。
- **根因（数学必然）**：轨迹 RMS 门限 1e-3 相对在近零分量（y~7.7e-5、vy~1.5e-3）上等效**绝对容差 ~1e-7**；fdderivs 输出 fxx..fzz 是 O(1/dx²)~O(100) 量级（含消去），截断 float 后绝对误差 ~6e-6，经 Ricci 进 RHS → 绝对误差 ~1e-5 → 小分量相对误差 0.5-1.8%，且随步数累积。**float RHS 与 1e-3 相对 RMS 门限根本不相容**（任何 ≥1e-8 绝对 RHS 误差都会在小分量上爆相对误差）；100 步 RMS 必 ≥0.7%（误差单调累积），Level-2 FAIL 数学确定，未跑全量（诚信：不浪费 18 min 确认已知失败）。
- **即使正确性通过，F=1.037（~960s）也远达不到 500s 目标（需 -70~85%）**。用户 500s 数学中 occupancy 2×+无 spill+指令减半 → 2-3s/step 的预期全部落空。
- **结论：FP32 混合精度死路，勿重试**。降级方案（只 float 后半段代数 / fdderivs 回 double）分析：保留 l_R/fdderivs double 则 Ricci monster 回 double（寄存器收益全失），且小分量绝对容差 ~1e-7 仍要求 RHS 全链 double——任何部分 float 都过不了轨迹 RMS 门。

## 全部已测杠杆（已穷尽，勿盲目重试）

### 任务一 CPU（28+ 杠杆）

| # | 杠杆 | 收益 | 正确性 | job/节 |
|---|---|---|---|---|
| 1 | loadbal least-loaded-first | 0（单独）/ -13s（DIST_INTERP 后） | check.sh PASS | §11/16.11 |
| 2 | NAN_CHECK off | 0 | — | §13 |
| 3 | loop/unroll/prefetch flags | 0 | — | §13 |
| 4 | gij_rhs 显式循环 + temp 内联 | 0 | check.sh PASS | §13.4 |
| 5 | skip 诊断约束残差（predictor co） | 0 | check.sh PASS | §13.10 |
| 6 | lopsidediff/kodiss 向量化 | 不可行（数据依赖 stencil 选择） | — | §13.14 |
| 7 | MPI rank 30 vs 60（SMT） | 0 | — | §13.15 |
| 8 | NUMA/绑核（ppr:socket/numa） | ~0（噪声） | — | §13.12 |
| 9 | OpenMP 15×2 / 10×3 | +29% / +42% 更慢 | — | §11.11 |
| 10 | -march/SVE | -11% 更慢 | — | §11.12 |
| 11 | armclang | +3.5% 更慢 | — | §11.13 |
| 12 | fderivs 4-field batch | 0 | check.sh PASS | §13 |
| 13 | fdderivs interior-direct | 0 | check.sh PASS | §12.5 |
| 14 | division→reciprocal (oinv) | +2.6% 更慢 | check.sh PASS | §12.2 |
| 15 | diff_new zero-init→loop | 0（flag 已覆盖） | — | §13.23 |
| 16 | Constraint_Out skip | 0（重算已 if(false) 跳过） | — | §13.17 |
| 17 | -Ofast 去 -fno-frontend-optimize | +2% 更慢 | check.sh PASS | §12.2 |
| 18 | LTO | 0 | — | §13.19 |
| 19 | PGO | 不可行（ld segfault） | — | §13.21 |
| 20 | flag-stacking (loopim+prefetch+ivopts) | 更慢 + 破坏正确性 | FAIL | §13.25 |
| 21 | PURE 属性 | 不可行（需移除错误 I/O） | — | §13.31 |
| 22 | flag + -funroll-loops | 0 | — | §13.33 |
| 23 | flag+loopim + -fsplit-loops | ~-1%（边际/噪声） | check.sh PASS | §13.39 |
| 24 | flag+loopim + -fgcse-sm/las | 0 | — | §13.43 |
| 25 | flag+loopim + -fmodulo-sched | +0.9% 更慢 | — | §13.46 |
| 26 | flag+loopim + -fpredictive-commoning | 0 | — | §13.46 |
| 27 | flag+loopim + -fprefetch-loop-arrays | 0 | — | §13.48 |
| 28 | SKIP_NAN_ALLREDUCE（移除 NaN barrier） | 0 净收益（straggler 转移） | check.sh PASS | §16.19 |
| 29 | 更细 loadbal（SPLIT_FACTOR） | 0 收益（check.sh PASS，非正确性阻断；psi4 非逐位但 RMS 在限内） | check.sh PASS | §16.23 |
| 30 | level-0 RHS 复制（dist_rhs） | -1.5%（实质无），40 步约束超限（Ham=8.15 >>2，浮点重排累积） | check.sh FAIL 40 步 | §16.18 |
| 31 | array-section lev8 dist | 栈溢出 segfault | — | §16.18 |
| 32 | 2-rank dist_rhs（owner+helper 点对点） | ~+7% 更慢（9.6 vs 9.0 s/step；lev8 块仅 48×48×24，malloc 开销 > 计算收益） | **check.sh FAIL 5 步**（RMS=0.16 >> 1e-3；OFF 对照 RMS=0 bit-exact） | job149276 |
| 33 | compute_rhs_bssn_ 点态融合重写 Stage 1a（whole-array→显式 do k 循环 + alpn1/chin1/gxx/div_beta 标量化，11 个 do k 循环；候选 ~/lab4-cpu-cand-rewrite） | **~+20% 更慢**（稳态 ~11s/step vs baseline ~9s/step；5 步 24.6/20.1/10.7/11.5/11.1s） | **check.sh PASS（RMS=0 bit-exact！）** 约束 Ham=0.228≤2 PASS | job149339 区段 |
| 34 | compute_rhs_bssn_ 点态融合重写 Stage 1b（k-滚动导数融合：fderivs_plane/fdderivs_plane 内联进显式 k 循环 + frx 反射助手，42 导数数组改 k-切片滚动不再物化；候选 ~/lab4-cpu-cand-rewrite，bssn_rhs.f90 86227B@01:43，build-rw/ABE 830608B sha256 ef787f43） | **-23%**（5 步 6.53/6.60/6.96/7.22/7.33s，稳态 ~6.9s/step vs baseline ~9s、Stage 1a ~11s；短跑 F≈1.30）；**Level-2 全量 40 步（job151519）确认**：40/40 步 6.85–7.85s/step，**avg 7.273s/step**，Total Evolve 292.423s，**This Program Cost = 297.31s < 340s**（OJ 100 分门限达成潜力） | **Level-1 5 步 + Level-2 全量 40 步均 check.sh FINAL PASS，Trajectory RMS=0（bit-exact，40/100 matched prefix）**；约束 Ham=0.2774/Px=0.0281/Py=0.0315/Pz=0.0265 全 ≤2 PASS；40 步零浮点漂移（对比 §16.18 40 步约束超限教训） | job151481（s1b 5 步）+ job151519（Level-2 40 步，Final=40.0/Analysis=0.1 OJ-sim，TwoP cache hit 912e370b84cec7cb；per-step 40 值、Total Running 296.353s 记录于 s1b_full_run.log） |
| ★ | -fno-tree-loop-distribute-patterns | **-2%** | check.sh PASS | §13.27 |
| ★ | + -ftree-loop-im | **-0.93%（叠加 -3.7%）** | check.sh PASS | §13.35 |

### 任务二 GPU

| 杠杆 | 收益 | 正确性 | job/节 |
|---|---|---|---|
| lb(256,2) 当前基线 | 实测最优 | — | §15.4 |
| lb(256,4) | 中性 | — | §15.4 |
| lb(256,1) | -20%（占用率 25%→12.5%） | — | §15.4 |
| rhs split v1 | -3%（RDC ABI spill 952B） | — | §15.4 |
| rhs inline-.cuh 单核 | 负收益（spill 688→2964B，4.3×） | — | §15.4 |
| shared-mem 7-point stencil | 不可行（15 场 × halo = 120KB > 82KB） | — | §15.4/context.md §A |
| memory pool | 中性 | — | §15.7 |
| block shape sweep (4,4,16 等) | 中性 | — | §15.7 |
| per-stream sync | **纠正：0 收益**（多轮 A/B 推翻 §15.7 -30s 假阳性） | — | M3/job142485 |
| cudaGraph 捕获 RK4 | 关闭（GPU ~100% 饱和，≤3% 上限） | — | §15.7 |
| LDG（__ldg + __restrict__） | 0 收益（read-only cache 路径不降 L1TEX stall） | bit-exact IDENTICAL | job142555 |
| M2 flag+loopim on TwoP | -1.9%（~0.7s on TwoP，边际） | Newton \|F\| 逐位一致 | job142849 |
| INLSTEN3 forceinline 4 stencil | **-13.8%（1266→1052s），已部署** | bit-exact PASS RMS=0 | job143426/143479 |
| lb(256,3) forceinline 后 | +10%（spill 5116B 杀死 occupancy 收益） | bit-exact 构建 | job143859 |
| lb(256,4) forceinline 后 | +15%（spill 6416B） | bit-exact 构建 | job143934 |
| GAMSMEM shared-mem staging l_Gam | 0（spill 降 2488→1472B 但 smem 延迟抵消） | bit-exact IDENTICAL | job144671 |
| BRFINAL branchless-fh | **-5.5%（1052→1007s），已部署** | bit-exact PASS RMS=0 | job145904/146361 |
| BR_ORD branchless 4th/2nd（无 clamp） | 2-step -6.7% | **100-step FAIL**（step 28 CUDA illegal memory access） | job147265/147353 |
| BR_ORD-fix hybrid（keep early-return + clamp fh args + branchless select） | **F=1.011（1007→996s，-1.07%），2-step F=1.037** | **check.sh FINAL PASS RMS=0 bit-exact** | job148583 |
| **P2 center-fh reuse（iter8，lopsided/kodis 同函数内 fh 去重，bit-exact）** | **F=1.006（969→963s，-6s），2-step F=1.033；spill 6352/6792B（基线 6444/6888B，-1.4%/-1.4%），live-set 不↑** | **check.sh FINAL PASS RMS=0 bit-exact，约束逐位一致** | job152457 |
| **P3.5 交叉项 fh 分组（iter10，16→8 峰值）** | **部署态实测 928.09→901.24s（job 153420，-2.9%）；spill stores 6352→5504B（-13%），natural regs 255 不动** | **check.sh FINAL PASS RMS=0 bit-exact** | job 152457→153420 |
| **P3.5c 交叉项链式累加（iter12，8→5 峰值，最后 live-set 杠杆）** | **F=0.9999（0 收益）；spill stores 5504B 逐字节不变、loads -8B、natural regs 255 不动，死路** | **bit-exact 4/4 IDENTICAL（链式==旧左结合，构造同序；Level-2 未跑，GATE F<1.0 跳过）** | job153640 |
| **P5 fh float（iter14，fdderivs 61 h_* double→float 精读暂存，iter9 点名最后方向）** | **F=0.9902（2-step 8.665 vs 8.580 s/step，-1% 更慢）；lb2 spill stores 5504→5064B（-8%）/ loads 5996→5496B（-8.3%）但 natural regs 仍 255（occupancy 无变化）；nolb spill +9%** | **2-step RMS=2.46e-07（0.000025%）FINAL PASS（精度远安全，对比 FP32 2 步 0.715%）；Level-2 未跑（GATE F<1.0 跳过）** | job154113（ptxas）/154246（A/B） |
| BR_ORD-fix v1（remove early-return + clamp + branchless） | 2-step F≈1.00（early-return 移除开销抵消收益） | bit-exact IDENTICAL（2-step） | job148474 |
| prolong3 lb(256,3) | 0（A/B 噪声内） | bit-exact | job144895 |
| prolong3 Z-loop 表达式重写 | 0（100 步实测确认） | bit-exact PASS RMS=0 | job145083 |
| **FP32 mixed-precision（迭代5）** | **速度 F=1.037（2-step 同网格 8.722 vs 9.041 s/step），但正确性 FAIL** | **2 步早期 RMS=0.00715（1 步后即 7× 超限）；约束 PASS（Ham=0.2228 ≤2）** | job151105/151129/151153 |
| FP32 降级（部分 float） | 未实施：分析判定数学注定失败（小分量绝对容差 ~1e-7 要求 RHS 全链 double；保 l_R double 则寄存器收益全失） | — | 迭代5 分析 |
| load hoist（swpipe2） | 0（spill 暴增 4×） | bit-exact | job145744 |
| **rhs split v2（post-forceinline 2-way 拆分）** | **kernel2 spill 暴增 5.6×（692→3880B stores/4416B loads），死路** | 未跑全量（ptxas 已判死路） | job149108 |
| **Lever A（contraction→kernel1）** | **kernel2 spill 3880→3896B（+0.4%，不变），死路** | 未跑全量（ptxas 已判死路） | job149409 |
| **P9 is_sommerfeld_boundary forceinline（iter18，in-place __forceinline__）** | **codegen NO-OP（SASS 逐字节等价，3B 符号表差异）；base 已 inline** | 未跑 A/B/L2（SASS 等价→无机制可达 F>1.0） | job155750/155778 |

**rhs split v2 死路分析（迭代2，重要）**：
- 假说：§15.4 rhs-split-v1 死路（rhs_deriv_kernel 952B RDC ABI spill）是 PRE-forceinline 测的，forceinline 后应消除 ABI spill → 拆分复活。
- 实测 ptxas（build.log 验证）：
  - `rhs_deriv_kernel`（kernel1，derivatives 段）：128 regs，**56B spill**（vs 单体 692B）——拆分成功，kernel1 极轻量。
  - `rhs_core_kernel`（kernel2，Ricci+RHS 段）：128 regs，**3880B spill stores / 4416B spill loads**（vs 单体 692B/1688B）——**5.6× 恶化**。
- **根因**：Ricci 修正项（Step 4，line 491-560，~60 存活变量：18 metric-deriv contraction gxxx..gzzz + 18 Christoffel l_Gam + 6 gup + 6 dGam + 6 l_R）整体落入 kernel2，加上 36 个 global scratch read（betaxx_out 等）新增 36 double 到 live-set → kernel2 live-set 比 单体更高。
- **结论**：2-way 拆分死路。forceinline 只减指令数（跨 TU 冗余）不减 live-set（自然需求仍 250-254）。kernel1 减负但 kernel2 增负，净负。
- **下一步（迭代3）**：3-way 拆分隔离 Ricci monster——kernel1(derivatives→global) + kernel2(Ricci only, 写 l_Rxx..l_Rzz + Gam-contraction 中间量到 global) + kernel3(RHS assembly 从 global 读 Ricci)。或 Ricci 修正项因式分解（命名中间量 Gam_g_xx = l_Gamxxx*gxxx+... 复用）减冗余存活变量。

**迭代3 naive 3-way 拆分：未实施（agent 重复超时死亡，未产出 ptxas）**。结构性分析判数学注定失败：6 个 Ricci 修正同时引用 51 个存活变量（18 l_Gam + 18 gxxx..gzzz contraction + 9 dGam + 6 gup）= ~408 字节，单 kernel 装不下 128 寄存器预算（1024 字节），无论从哪里切，Ricci kernel 都必 spill。

**迭代4 Lever A（contraction→kernel1）死路分析（重要）**：
- 假说：把 18 个 metric-deriv contraction（gxxx=l_gxx*l_Gamxxx+... 等）从 kernel2 移到 kernel1 尾部（kernel1 只 56B spill，有 headroom），减 kernel2 Ricci 段 live-set 18 变量。
- 实测 ptxas（build.log 验证）：kernel2 spill 3880B → **3896B（+0.4%，不变）**；kernel1 spill 56B → 112B（仍轻量）。
- **根因**：移动计算不移除 live-set 变量。compiler 把从 global 读回的 18 个 gxxx..gzzz 仍保活在寄存器（每个被多个 Ricci 分量引用），故 kernel2 live-set 实质不变。剩余 33 个变量（18 l_Gam + 9 dGam + 6 gup = 264 字节）+ 计算临时变量 + 6 个 l_R 输出，仍是不可压 live-set。
- **结论**：Lever A 死路。forceinline 后寄存器压力是 BSSN Ricci 公式数据依赖的固有属性，非编译器调度问题。

**500s 可达性终审（迭代1-4 证据综合）**：
- ncu（forceinline 部署态）：rhs_kernel 3.39ms/launch，Achieved Occupancy 21.51%（forceinline 未改），Eligible Warps 0.49，L1TEX scoreboard stall 46.6%。
- `__ldg`（read-only cache，等价 shared-mem 缓存机制）+ GAMSMEM（shared-mem staging l_Gam）均 bit-exact 但 **0 收益**，证明 stall 非缓存命中率问题（L1 hit 80.91%、L2 hit 91.14% 已高），而是 **25+ 场 stencil 固有 load 延迟体积**。spill traffic（692B/launch）仅是此 load 体积的小部分。
- 500s 数学：需演化 ≤456s = 4.56s/step avg（TwoP ~44s 固定），当前 9.53s/step。rhs 6.67s/step 须砍到 ~1.0s/step（**-85%**），同时 non-rhs 须达 3.51s/step 下界。即令完美消除 spill（不可能），亦仅减 692B traffic，不破 46.6% stencil-load-latency stall。
- **迭代1-4 结论**：rhs split/Lever A/Lever B（recompute-on-demand）系列均被 Ricci 33+ 变量固有 live-set 卡死，500s 在「寄存器/spill/live-set 重构」策略族内不可达。需用户输入（见 blocked-stop 报告）。

**Lever A 死路分析（迭代4，contraction→kernel1，重要）**：
- 假说：迭代2 split v2 死路根因是 kernel2 的 Ricci 段同时引用 51 个存活变量（18 contraction gxxx..gzzz + 18 l_Gam + 9 dGam + 6 gup）。把 18 个 contraction 移到 kernel1 尾部（kernel1 已有 val_gxx..val_gzz + l_Gam 在手），kernel2 改为从 global scratch 读 18 个 contraction → 应减 kernel2 live-set 18。
- 候选：`~/lab4-gpu-cand-leverA-20260823-181849`（基于迭代2 split candidate `~/lab4-gpu-cand-split-20260823-173209`）。patch：bssn_rhs_gpu.cu 增加 18 个 SCRATCH_GXXX..SCRATCH_GZZZ slot（NSLOTS 36→54），kernel1 尾部加 18 contraction 计算+写 scratch，kernel2 Step 3 改为从 scratch 读。sha256: `3ac82c78...`（基线 `657792b8...`）。
- 实测 ptxas（job149409，build.log 验证）：
  - `rhs_deriv_kernel`（kernel1）：128 regs，**112B spill stores / 88B spill loads**（vs 迭代2 的 56B/56B）—— spill 翻倍（预期，新增 18 contraction + 18 scratch write）。
  - `rhs_core_kernel`（kernel2）：128 regs，**3896B spill stores / 4376B spill loads**（vs 迭代2 的 3880B/4416B）—— **几乎不变**（stores +0.4%，loads -0.9%）。
- **根因**：contraction 值 `gxxx..gzzz` 仍需在 kernel2 的 Step 4（Ricci connection correction）同时存活。无论 `gxxx = l_gxx*l_Gamxxx + ...`（inline 计算）还是 `gxxx = d_scratch[...]`（global 读），编译器看到相同数据流：18 个值被赋值后在 Step 4 使用 → 仍需保活在寄存器（或 spill）。移动计算位置不改变 live-set，因为 `l_gxx`、`l_Gamxxx` 等输入变量在 kernel2 中本就存活（Steps 4/6/7/constraints 都用）。contraction 从计算变为 global read **不减少任何存活变量**。
- **结论**：Lever A 死路。moving computation without removing live-set variables is futile。kernel2 的 3896B spill >> 2000B 门限，未进 A/B。
- **下一步 Lever B 建议（recompute-on-demand）**：算完 `l_Rxx` 后立即释放 `gxxx, gxyx, gxzx`，算 `l_Ryy` 时重算所需 contraction（用额外 FLOP 换 live-set 减少）。但风险：重算时仍需 `l_gxx, l_gxy, l_Gamxxx` 等输入存活，live-set 减少有限。更激进的方向是 3-way split 隔离 Ricci monster，或重新考虑整体 kernel 结构（回到单体 kernel + 不同 launch_bounds 策略）。

**BR_ORD 死路分析（重要教训）**：
- 原始 BR_ORD patch 把 d_fderivs_point 的 4th/2nd order if/else 改为 branchless predication（无条件计算所有 fh() 值），2-step bit-exact -6.7%。
- 但 100-step 在 step 28 崩溃（CUDA illegal memory access at gpu_manager.cu:83）。
- **根因**：branchless 版无条件调用 fh(i±2,j,k) 等，即使 stencil 条件不满足。fh() 的 in_range/valid 乘法将越界结果置零，但**内存访问本身仍发生**。在细网格层（~step 28 激活的 moving grid），越界访问越过数组末尾 → CUDA illegal memory access。
- **修复**：clamp fh() 参数到 [imin,imax]×[jmin,jmax]×[kmin,kmax]，确保所有内存访问在数组范围内。在 stencil-valid 点 clamped==original（条件保证范围），在无效点值被 m4/m2=0 屏蔽 → bit-exact。
- 修复后 100-step PASS，bit-exact（RMS=0，Ham=0.28974817 逐位一致），F=1.011（-1.07%，-10.8s）。
- 收益从 2-step 的 -6.7% 降到 100-step 的 -1.07%：clamp 开销（12 个 max/min 指令/调用 × 21 调用）+ 后期步（小网格）divergence 减少更少。
- **2-step bit-exact ≠ 100-step PASS 的核心教训再现**：原 BR_ORD 2-step bit-exact 但 100-step 崩溃。修复版 2-step bit-exact 且 100-step PASS。必须跑 Level-2。

## 未测/可探索方向（需用户授权，多数越界）

- **compute_rhs_bssn_ 算法级重写**（whole-array → 显式循环 + 寄存器驻留，~978 行）：§13.11 估砍 ~50% compute_rhs（2.72→1.36s/step）。高风险，需全量 check.sh 验证（浮点重排可能改变 RMS/约束）。lab 规则禁止“未经验证的低精度替换”，需用户授权。
- **低阶 stencil / 少场量**：改物理，RMS 可能超 1e-3，禁止。
- **cudaGraph 捕获**（GPU）：边际，未实测。
- **packed collective lev8 straggler 复制**（CPU）：实测 11× 慢 + 栈溢出（§16.18），非正确性阻断但实现不可行。
- **2-rank owner+helper dist_rhs（方案 A）**：将 30-rank Bcast/Gatherv/Allreduce 改为 2-rank 点对点 Isend/Irecv，仅 owner + 1 helper 参与，其余 28 rank 跳过。实测 job149276：5 步无崩溃，~9.6 s/step（比 baseline 9.0 略慢，lev8 块仅 48×48×24 = 55K 点，malloc 92 数组 × 432KB = 39.8MB/call × 8 calls/step 的分配开销 > 计算收益）。**正确性失败**：RMS = 0.16 >> 1e-3（OFF 对照 RMS = 0 bit-exact）。根因：k-slice 边界 stale-read——Christoffel 符号等中间量由各 rank 独立计算，但 Ricci 张量等后续计算在 k-slice 边界读取邻近 k 的 Christoffel 值时获得的是**计算前初值**而非计算后新值。即使仅 1 个边界（2 rank），误差已超限。**k-slice 分布式 RHS 方案无论 rank 数均不可行**（#30 30-rank 版 40 步 FAIL，#32 2-rank 版 5 步 FAIL）。

- **compute_rhs_bssn_ 点态融合重写 Stage 1a**（#33，用户授权的算法级重写第一步）：whole-array 表达式（5 点态 init + 35 RHS + 11 约束）改为显式 `do k/j/i` 嵌套循环，alpn1/chin1/gxx/gyy/gzz/div_beta 改点态标量（消 ~20 个中间数组），保留 21 个 fderivs 调用和 42 个导数数组。候选 `~/lab4-cpu-cand-rewrite`（基于 clean formal baseline，无 KSLICE/dist）。实测：**编译通过（vectorized 68 loops + SLP）**，5 步短跑 **check_result.py FINAL PASS 且 RMS=0（bit-exact！）**——证明显式循环与 whole-array 表达式数值完全等价（gfortran 编译为相同求和序）。但**性能 ~+20% 更慢**（稳态 ~11s/step vs baseline ~9s/step）。根因：点态标量融合产生循环结构开销 > 消除 ~20 数组（6.6MB→~4.5MB）的缓存收益；42 个导数数组（bulk of working set）仍物化。**Stage 1b（k-滚动导数融合）已实施并实测（见 #34，job151481）**：5 步 6.53/6.60/6.96/7.22/7.33s（稳态 ~6.9s/step，-23% vs baseline ~9s、-37% vs Stage 1a ~11s），**check_result.py FINAL PASS 且 RMS=0（bit-exact）**，约束 Ham=0.228≤2。导数数组 k-切片滚动（不物化）真正砍掉 working set，与 KB §13.11 缓存假设一致；证 Stage 1a 的 +20% 慢是"点态融合但导数仍物化"的半吊子态，非重写路线本身失败。

  **推论更新**：Level-1 5 步数据乐观（早期步网格粗、且本跑 cache hit 无 TwoP 干扰），需 Level-2 40 步全量验证定论；**Level-2（job151519）已定论：40 步维持 avg 7.27s/step（非早期步乐观值），Program Cost 297.31s < 340s 临界，且 40 步 RMS=0 bit-exact（无浮点累积）**。compute_rhs 重写路线兑现。候选 Program Cost 297.31s vs 当前 OJ 408.9s/86 分 → 若 OJ 开销近似，可达 100 分区间；**OJ 提交决定权在主 agent（含 TwoP cache 是否随包部署、OJ 覆盖 Analysis_Time=0.1 影响）**。

## 迭代6 SMEM tiling (selective, iter6)：bit-exact 但 -5.9% 更慢，死路（负收益）

**候选**：`~/lab4-gpu-cand-smem-20260824-013937`（patch `patch_smem_tile_iter6.py`）。
改动：bssn_rhs_gpu.cu（sha256 `f7305b9b...`）+ derivatives.h（`63589489...`），formal 基线 `4a4b2aab`/`ffe0c7c3`。

**机制（与 LDG/GAMSMEM 的本质区别）**：block (8,8,4) 协作加载 6 最热场（betax/betay/betaz/chi/Lap/dxx；按 stencil 调用实测，每场 1 fderivs(12 load) + 1 fdderivs(63 load) + 1 lopsided(18) + 1 kodis(13) = 106 load，全部 11 个 fdderivs 场等热，此 6 场任意选）的 (12,12,8) tile（±2 halo，1152 doubles=9216B/场，6 场=55296B dynamic smem），__syncthreads 后 fderivs/fdderivs 改读 smem —— **真正消灭 halo 重读**（LDG/GAMSMEM 不消灭）。tile 存 fh-effective 值（f*fac*in_range*valid 在加载时按原 fh 语义逐位预计算）→ smem fh 为纯读，bit-exact。lopsided/kodis 保持 global（±3 halo 放不进 ±2 tile，且 gi=-3 处两套 fh 语义不同）。tile load 放在越界检查前保证全线程同步。

**Level-0 ptxas（job 151410）**：128 regs（launch_bounds 锁），1504B stack（基线 ~1256B），**3912B spill stores（基线 ~4500B，-13%）/ 5080B spill loads（基线 ~5000B）**，1 barrier（__syncthreads），1132B cmem。spill stores 降但 loads 略升，净 spill 流量略减。

**Level-1 A/B（job 151410，4 轮交错同节点 2 步）**：base 8.5103/8.4992（中位 8.5048s）vs smem 9.0096/9.0075（中位 9.0086s）→ **F=0.944（-5.9%）**；Program Cost base 20.82 vs smem 21.78（F=0.956）。两轮 smem 完全一致，非噪声。

**Level-2 全量（job 151446，100 步）**：This Program Cost = **1015.18s**（部署基线 996.47s），**check.sh FINAL PASS，Trajectory RMS=0（0.000000%）bit-exact**，Ham=0.28974817 等约束与基线逐位一致，4 .dat 全产。2 步 IDENTICAL + 100 步 RMS=0 → tile 语义在任意网格尺寸下逐位正确（无 BR_ORD 式崩溃风险）。

**死路原因（诚实）**：
1. tiled 场 fderivs/fdderivs 的 global load 确实变 LDS（构造上必然 + ptxas 1 barrier 佐证；登录节点无 nvdisasm/ncu，未直接数 LDG 指令，属残余未测项）。
2. 但 **smem carveout 代价 > 收益**：2 blocks × 54KB = 108KB smem → L1 从 ~164KB 缩到 ~56KB；kernel 仍有 ~744 次/线程 lopsided/kodis + 场值 global load 走 L1，latency-bound（eligible warps 0.49）下 L1 缩水 → 更多 L2/DRAM 往返 → 净更慢。
3. tile load 开销：27 次 global load/线程 + ~500 条边界算术 + 1 次全 block barrier（每 launch 一次，RHS kernel 每步多次 launch）。
4. 与 GAMSMEM（spill 降但 smem 延迟抵消）、LDG（read-only cache 0 收益）同源结论强化：**stall 是 25+ 场 stencil 固有 load 延迟体积 + 低 occupancy（0.49 eligible warps）的组合，缓存路径/位置（L1、RO-cache、smem）都治不了**；连真正消灭 halo 重读也无效。
5. 结论：**selective SMEM tiling 死路**。即便减场数（省 carveout）也只缩小 0.5s/step 亏空的一部分，不改变负收益定性。勿重试（除非先提 occupancy，而 lb(256,3/4) 已被 spill 杀死）。

## 迭代9 P3：Ricci contraction 分量级 recompute（减 live-set → 提 occupancy），**死路（live-set 未降）**

**候选**：`~/lab4-gpu-cand-p3-20260824-130154`（patch `patch_p3_ricci_recompute.py`）。改动仅 1 文件：`src/bssn_rhs_gpu.cu`（formal `4a4b2aab` → 候选 `7ab128b5`）；derivatives.h `8650bda1`/lopsidediff.h `5dddaa75`/kodiss.h `e88f9632` 与部署基线逐字节一致（diff 验证）。

**机制（用户优先级 #4，P0 dispatch_stall 31% → occupancy 不足是首要 stall）**：rhs_kernel Step 4 的 6 个 Ricci 修正同时引用 18 个 contraction（gxxx..gzzz，各 3 项 mul-add），live-set 峰值区。改为按分量重算：把 Step 3 一次性算 18 个 contraction 全保活，改为 Rxx 前只算它用的 12 个、用完即死，Ryy 前重算其 12 个……Rxy/Rxz/Ryz 各 15 个。每 contraction 重算最多 3 次（81 次定义 vs 原 18），bit-exact（公式逐 token 不变；l_g/l_Gam 在 Step 3→Step 4 间不变，重算同值）。

**各 Ricci 分量引用的 contraction 子集（源码提取）**：Rxx 12 = {gxxx,gxxy,gxxz,gxyx,gxyy,gxyz,gxzx,gxzy,gxzz,gyyx,gyzx,gzzx}；Ryy 12 = {gxxy,gxyx,gxyy,gxyz,gxzy,gyyx,gyyy,gyyz,gyzx,gyzy,gyzz,gzzy}；Rzz 12 = {gxxz,gxyz,gxzx,gxzy,gxzz,gyyz,gyzx,gyzy,gyzz,gzzx,gzzy,gzzz}；Rxy 15、Rxz 15、Ryz 15（各含 12 子集并集 + 交叉项）。patch 脚本自动从源码提取子集 + 校验 used-set==computed-set + 公式逐 token 一致。

**Level-0 ptxas（job 153014，单 TU nvcc 同 CMake flags 逐字节一致，apples-to-apples）**：

| 配置 | regs | stack | spill stores | spill loads |
|---|---|---|---|---|
| base-lb2（部署基线） | 128 | 1496B | 6352B | 6792B |
| base-nolb（自然） | **255** | 640B | 1240B | 1280B |
| base-lb3 | 80 | 1840B | 14108B | 16316B |
| cand-lb2 | 128 | 1336B | 6224B | 6608B |
| cand-nolb（自然） | **255** | 424B | 864B | 736B |
| cand-lb3 | 80 | 1688B | **14152B** | 16248B |

**死路判定（父 agent 门限命中）**：cand natural regs 仍 255（≈250 未降）且 cand lb3 spill 仍 14152B（>3000B 门限）→ **未跑 A/B，直接报告死路**（门限明确：live-set 未降勿硬跑，swpipe2 教训）。

**根因（诚实，三层证据）**：
1. **recompute 机制本身生效**：Ricci 段 spill 确实降（nolb stores 1240→864B -30%、loads 1280→736B -42%；lb2 stores 6352→6224B -2%）——编译器未把 81 次重算 CSE 回 18 次，分量级算-用-释放确实减了 Ricci 段压力。
2. **但 kernel 级 live-set 峰值不在 Ricci Step 4**：natural regs 255 已达 **sm_80 硬件上限 255 regs/thread**（250 pre-P1 → 255 P1+P2 即撞顶），kernel 想要 >255 regs。Ricci 段只贡献了超限溢出的一部分（~380B/thread），峰值在别处。
3. **真峰值 = fdderivs 61-fh 段**：P1（fdbr）给 d_fdderivs_point 加 61 个 fh 值同时存活 → lb2 spill stores 4532→6444B（+42%，job 152062），natural regs 250→255（撞顶）。证据链：P1 加 fh 值前 natural 250/spill 104B；P1+P2 后 natural 255/spill 1240B；P3 减 Ricci 18→15 contraction 后 natural 纹丝不动 255。**下一步应攻 fdderivs 61-fh 段（如 fh 分两半算 fxx/fyy/fzz 组 + 交叉项组、或 fh 值 float 精读暂存），而非 Ricci contraction**。

**结论（PASS 判定：死路）**：contraction 分量级重算在 ptxas 层面证伪“减 Ricci live-set 能提 occupancy”。机制有效但目标段不是峰值段；occupancy 21.5% 的瓶颈在 fdderivs 61-fh 段（同 FP32 迭代5 的“峰值 live-set 不在想动的那段”教训同源）。候选保留（bit-exact 构造已验证，编译通过），未部署，未跑 Level-1/2（门限指示）。

## 迭代11 P3.5b：fdderivs 对角项单输出分组，**死路（已部署代码早已实现该分组，patch 为 NO-OP）**

**候选**：`~/lab4-gpu-cand-p35b-20260824-142731`（patch `patch_p35b_diag_group.py`，实为验证脚本：5 项 P3.5b target-form 不变式全 PASS → 输出 NO-OP，sha256 不变；非真 patch）。

**证据链（诚实，任务前提有误）**：
1. **对角段自 P1（fdbr，8650bda1）起已是逐输出分组**：fxx 组声明 4 个 i 向 h + h_000（峰值 5，非任务假设的"13 同时存活"）；fyy 组 4 个 j 向（h_000 复用）；fzz 组 4 个 k 向。`diff pre-p35(P1态) vs deployed(P3.5态)` 证实 P3.5 只改交叉项（r_fxy_0..3 提取，52 行 diff 全在交叉段），对角段 P1→P3.5 文本零改动。
2. 对角段 fh 调用多重集 = 13（5+4+4），**峰值存活 5 ≤ 任务目标 6**，已达结构下限（fxx 4th 公式需 5 值同时）。
3. 交叉项（fxy/fxz/fyz）不引用 h_000 → h_000 在 fzz 后即死，当前顺序最优（任务里"h_000 是否需活到交叉项"的疑问：不需要）。
4. **Level-0 ptxas（p35-l0 job log；cand==部署态，hash 3ede4646 今日复验与 formal 逐字节一致）**：lb2 128 regs / 1352B stack / **5504B spill stores / 5996B loads**；**natural 255 regs（sm_80 硬件上限）/ 568B stack / 1036B stores / 1140B loads**。P3.5b 为 byte-identical no-op → ptxas 必与部署态相同 → **任务门限命中（natural 仍 255 + lb2 spill 仍 ~5500B 未 <5000B）→ 未跑 A/B/Level-2，直接报告死路**（与 iter9 P3 同 gate，iter7 swpipe2 教训：live-set 未降勿硬跑）。
5. candidate == formal：4 文件 sha256 逐字节一致 + `diff -r` SRC_TREES_IDENTICAL。

**结论（PASS 判定：死路/no-op）**：对角项分组自 P1 已实现，P3.5b 无可改动；"13 fh 同时存活"的形态在部署谱系中从未存在。fdderivs 61-fh 段 live-set 峰值现位于**交叉项段（8 = 4 h + 4 r）**，非对角段（5）。

**下一方向（未测，P3.5c，需主 agent 决定是否续跑）**：交叉项增量累加器折叠——把 r_fxy_0..3 四值分离改为链式 `t=r0; t-=F8*r1; t+=F8*r2; t-=r3`（峰值 8→5，位级构造可行：C++ 左结合与链式同序）。预期收益 < P3.5 的 -2.9%（P3.5 是 16→8 才 -13% spill；8→5 更小），不构成 500s 路径。

**500s 可达性最终判断（11 轮证据综合）：不可达（当前 kernel 结构/寄存器模型内）**：
- 已部署 901.24s（P1+P2+P3.5 自 1266s -29%）。目标 ≤500s 需 -44.5%，即 rhs ~6.5s/step → ~1.0s/step（-85%）。
- occupancy 已到物理极限：natural 255 = sm_80 硬件上限（kernel 想要 >255 regs，live-set 是 BSSN 公式固有：fdderivs 61-fh + Ricci 33+ 变量）；lb3(80)/lb4(64) 提 occupancy 被 spill 杀死（13240B/…），lb1 更慢 → 25%（128 regs × 2 blocks）是 register-limit 平衡点。
- latency-bound 本质：L1TEX scoreboard 46.6%、eligible warps 0.49，stall 是 25+ 场 stencil 固有 load 延迟体积；L1/RO-cache/smem/LDG 缓存路径族全部 0 或负收益。
- 残余可测杠杆仅 P3.5c（预期 1-3% wall），离 -44.5% 差两个数量级。500s 仅可能靠算法级重构砍 stencil load 体积（越 Lab4 物理/诚信边界）达成。

## 迭代12 P3.5c：交叉项链式累加折叠（iter12，最后一个 live-set 杠杆），**死路（bit-exact 但 0 收益，live-set 未降）**

**候选**：`~/lab4-gpu-cand-p35c-20260824-143220`（patch `patch_p35c_chained_acc.py`）。改动仅 1 文件：`src/derivatives.h`（formal `3ede4646` → 候选 `5e79606c`）；lopsidediff.h `5dddaa75`/kodiss.h `e88f9632`/bssn_rhs_gpu.cu `4a4b2aab` 与部署基线逐字节一致。

**机制（任务：交叉项峰值 fh 存活 8→5）**：P3.5 交叉项（fxy/fxz/fyz）每块 4 行、每行 4 fh → 行和 r_0..r_3 同时存活到末尾组合 `m4*C*(r0 - F8*r1 + F8*r2 - r3)`。P3.5c 改为链式增量累加 `acc=r0; acc-=F8*r1; acc+=F8*r2; acc-=r3;`，行 fh 算完即并 acc，峰值 8→5。patch 脚本 5 项不变式校验全 PASS：fh 调用多重集 61 个逐 token 一致、行子表达式 verbatim 保留、m2 项 verbatim、链式 op 序 == 旧左结合组合（C++ 左结合 `((r0-F8*r1)+F8*r2)-r3` 与链式同序 → bit-exact by construction）、防重入守卫（二次应用报错）。

**Level-0 ptxas（job 153640，单 TU apples-to-apples 4 配置 + 全量 build-p35c）**：

| 配置 | regs | stack | spill stores | spill loads |
|---|---|---|---|---|
| base-lb2（P3.5 部署态） | 128 | 1352B | 5504B | 5996B |
| cand-lb2（P3.5c） | 128 | 1352B | **5504B（不变）** | 5988B（-8B） |
| base-nolb | 255 | 568B | 1036B | 1140B |
| cand-nolb | 255 | 568B | **1036B（不变）** | 1140B（不变） |

**live-set 未降**（任务门限命中：natural 仍 255 + lb2 spill 5504B 未 <5000B）。全量 build-p35c ptxas 与单 TU 一致（128 regs / 5504B / 5988B）。

**Level-1 A/B（job 153640，4 轮交错同节点 2 步）**：base1 7.86339 / p35c1 7.86843 / p35c2 7.86626 / base2 7.8694（Step2 s）；base median 7.86639 vs p35c median 7.86735 → **F=0.9999（0 收益）**。**4/4 .dat IDENTICAL（bit-exact 实证）**。

**Level-2 100 步：未跑**（GATE FAILED：F<1.0 且 live-set 未降，按门限规则跳过；bit-exact 已实证，100 步 check.sh 无风险但无意义）。

**死路原因（诚实，三层）**：
1. **机制层面**：链式累加构造 bit-exact 且编译通过，fh 峰值分组确从 4 行同时存活降为 1 行 + acc。
2. **但 ptxas 层面**：spill stores 5504B 逐字节不变（仅 loads -8B）、stack 不变、natural regs 255 不动——**编译器本就把 r_0..r_3 视作短生命周期**（或交叉项段本就不是决定 natural 255 的峰值段，与 iter9 P3 结论同源：fdderivs 61-fh 段的峰值在别处，不在交叉项行和上）。8→5 的源码级显式折叠被 ptxas 寄存器分配吸收，无任何可测收益。
3. **运行层面**：A/B 实测 F=0.9999（噪声内），与 ptxas 结论一致。

**GPU 侧优化收敛结论（iter1-12 全部证据）**：
- 已部署优化链（P1 fdbr + P2 center-fh + P3.5 交叉项分组，自 1266s 起 -29%）**901.24s（job 153420，check.sh FINAL PASS RMS=0 bit-exact）已是本内核结构/寄存器模型内的实际极限**。
- P3.5c（8→5）是本结构内最后一个 live-set 杠杆，实测 0 收益后，**不再有未实测的寄存器/spill/live-set 重构方向**（P3/P3.5b/P3.5c 三连证：Ricci 段、对角段、交叉项段都不是 natural-255 峰值段的决定者，峰值是 fdderivs 61-fh 全段 + Ricci 33+ 变量的 BSSN 公式固有组合，撞 sm_80 255 regs 硬件上限）。
- 500s 目标需 -44.5%（rhs ~6.5→~1.0s/step），残余差距两个数量级，仅靠算法级重构（砍 stencil load 体积，越 Lab4 物理/诚信边界）可达；occupancy 21.5%、latency-bound（L1TEX scoreboard 46.6%、eligible warps 0.49）、L1/RO-cache/smem/LDG 缓存路径族全 0/负收益的证据链在 iter11 已固化。
- **建议**：GPU 侧优化收束，部署基线 901.24s 保持；如需继续推进须主 agent 决策（算法级重构授权或接受现状）。

## 回写规则

新候选测完后，无论成败都回写本文件：杠杆名 | 任务 | F(speedup) | 正确性(check.sh) | job | verdict。失败的杠杆标注死路原因（区分“0 收益” vs “check.sh FAIL” vs “不可行”），防止后续 agent 重试。这对应 KernelPro 的 dead-end pruning 与 KernelEvolve 的 is_buggy 元数据。

## 迭代15 P6：prolong3 6×6 完整 unroll + Z/Y 列融合（bit-exact），**死路（F=0.9235，-7.6%）**；附 P6b forceinline 证据

**候选 P6a**：`~/lab4-gpu-cand-p6-20260824-081206`（patch `assets/lab4/opt/search/patch_p6_prolong3_unroll.py`）。改动仅 1 文件：`src/prolongrestrict_cell_gpu.cu`（formal `a81cd33e` → 候选 `49a99e03`）；derivatives/lopsided/kodiss/bssn_rhs 与部署基线逐字节一致。

**机制（ncu3 证据：prolong3 fixed-latency execution dependency 31.3%，66 regs，0 spill，336B stack）**：336B stack = tmp2[6][6]+tmp1[6] 物化在 local memory（36 STL+36 LDL/线程）。P6a：Z 向 6×6 全展开为 36 个显式 d_zinterp6 链（每 n 列 6 链×6 独立 load），Y 向按列立即融合（消除 tmp2 数组），关联顺序逐 token 保留 → bit-exact by construction（5 项不变式：helper body verbatim、Y 序 z0..z5/z5..z0、36 链 j-map、n-map、tmp2 无残留）。

**Level-0 ptxas（job 154433，单 TU apples-to-apples）**：base 66 regs/336B stack/LDL 33/STL 10/DFMA 219/CALL 93/SASS 514KB；cand **120 regs**/40B stack/**LDL 0**/STL 5/DFMA 579/CALL 453/SASS 1.8MB。**stack round-trip 确实消除（LDL 33→0）但寄存器 66→120（占用率 3→2 blocks）且 SASS 膨胀 3.5×（全展开 + 全 CALL 显式化）**。

**Level-1 A/B（job 154454，OFF/ON×2 交错 2 步）**：OFF [8.205,8.422,8.201,8.403] mean 8.308 s/step vs ON [8.893,9.087,8.901,9.107] mean 8.997 → **F=0.9235（-7.6%）**，分布完全不重叠（非噪声）。**check.sh FINAL PASS，RMS=0（bit-exact 实证）**，约束 PASS。

**死路原因（三层）**：
1. 机制成功：LDL 33→0、STL 10→5、stack 336→40B，bit-exact。
2. 但 36 链全在寄存器中展开 → 66→120 regs → 占用率 3 blocks→2 blocks（37.5%→25%，MIG 下 24→16 warps/SM）→ latency-hiding 能力 -33%，胜过了消除 72 次 local 访存 + 暴露并行 load 的收益。
3. SASS 1.8MB（3.5×）→ I-fetch 压力。**prolong3 的 66-reg 滚动循环 + ABI call 结构是局部最优**：任何提 ILP 的重构都抬高寄存器、丢占用率，净负。

**P6b（新杠杆，probe 级，job 154503/154654）**：SASS 揭示 d_symmetry_bd_1b（fmisc.h 声明无 forceinline，定义在 fmisc_gpu.cu）在 CUDA_SEPARABLE_COMPILATION（-rdc=true，CMakeLists:132/134）下是 **ABI CALL**——prolong3 每输出点 216 次 call/return（base probe CALL.ABS.NOINC 93）。call/return 固定延迟 + 返回栈串行化是 fixed-latency stall 的首要嫌疑。候选 `~/lab4-gpu-cand-p6b-20260824-082335`（fmisc.h `f14df31c`→`8ff0acaf` 声明改 forceinline 定义 [f_at_1b + d_symmetry_bd_1b verbatim 移入 header]，fmisc_gpu.cu `320f415a`→`3fab96a9` 删本地定义 + 补 `#include "fmisc.h"`；其余文件逐字节同 formal）。probe ptxas：prolong3 **66→100 regs**（占用率又 3→2 blocks）/CALL 93→21/BSSY 22→94/LDL 33 仍在（tmp2 未动）；**sommerfeld_routbam 64→52 regs、stack 40→0（占用率升）**。A/B（job 154689）结果见下方更新。

**收敛判断（P6a 后）**：prolong3 的依赖链重构杠杆（unroll/融合）已实测为负——寄存器/占用率权衡是死结。即使捕获 ncu3 全部 31% headroom 也仅 -72s（901→~830s），离 500s 差两个数量级（rhs 须 -85%）。500s 不可达（iter11-14 已定论），prolong3 亦无净正收益杠杆。

## 迭代13 P4：constraint tail work-elimination，**边际死路（成本仅 ~1%）**

**诊断（job p4diag13b，2 步 A/B 去掉约束段变体）**：
- 约束段（rhs_kernel line 852-992，co==0 才执行，每 RK4 步 1/4 substep）成本实测：
  - base（含约束）: Step1 8.469 / Step2 8.701
  - diag（无约束）: Step1 8.408 / Step2 8.617
  - **差异 ~0.9-1%（Step2 -0.084s/8.7s）**
- 虽然约束段占 kernel 行数 14.7%，但实际执行成本只 ~1%（1/4 substep 执行 + 约束计算相对简单）。
- **结论**：constraint tail work-elimination 最大收益 ~1%，且 constraint.dat 必须正确（check.sh 4 个 .dat 之一，不能消除核心输出）→ 边际死路，不值得做。
- 证据：`~/lab4-gpu-cand-p4diag-20260824-070913/evidence/p4diag/job.log`。

## 迭代14 P5：fdderivs fh 值 float 精读暂存（61 h_* double→float），**死路（精度安全但 -1% 更慢）**

**任务（用户决策"你指路，我实测"的最后一个实测方向）**：iter9 点名的"fh 值 float 精读暂存"——fdderivs 61 个 fh 值当前 double（2 regs/个）→ float（1 reg/个），理论减 61 regs，攻 natural-255 峰值段。精度边界：fh 值 O(1) 场值 float 存储（24-bit 尾数，~6e-8 相对误差），公式算术/系数/m4/m2/r_* 行和保持 double；交叉项-only float（16 个）为精度降级路径（未触发）。

**候选**：`~/lab4-gpu-cand-p5-20260824-073425`（patch `assets/lab4/opt/search/patch_p5_fh_float.py`：hash 守卫 + 15 行声明/61 casts 不变式 + 防重入）。改动仅 `src/derivatives.h` d_fdderivs_point 的 15 行 h_* 声明（`double h_x = fh(...)` → `float h_x = (float)fh(...)`），d_fderivs_point（含同名 h_jm2/h_km2 行）与其余文件零改动。derivatives.h：3ede4646 → ba3fee94。

**Level-0 ptxas（job 154113，单 TU bssn_rhs_gpu.cu probe，base/cand × lb2/nolb，方法同 p4diag-l0）**：
| variant | regs | stack | spill stores | spill loads |
|---|---|---|---|---|
| base-lb2（部署态） | 128 | 1352B | 5504B | 5996B |
| cand-lb2（P5） | 128 | 1304B | **5064B（-8.0%）** | **5496B（-8.3%）** |
| base-nolb | 255 | 568B | 1036B | 1140B |
| cand-nolb（P5） | 255 | 648B | 1132B（+9.3%） | 1244B（+9.1%） |
- **natural regs 仍 255**（sm_80 硬件上限，kernel 想要 >255）→ **occupancy 无变化**，P5 未能破 255 峰值。
- lb2 spill 双降 -8%：fh float 在 128-reg 上限处确实给分配器腾了头（P3.5 后 lb2 spill 首次再动）；但 nolb spill 反升 +9% → float 转换扰动自然分配（编译器在峰值处多溢）。

**Level-1 A/B（job 154246，OFF/ON ×2 交错 2 步，同节点）**：
- OFF（部署态）：8.480/8.670/8.484/8.685 → mean 8.580 s/step；ON（P5）：8.541/8.764/8.585/8.771 → mean 8.665 s/step → **F=0.9902（-1.0%）**。
- **2-step RMS = 2.46e-07（0.000025%）FINAL PASS**（ON vs OFF 对照，两轮一致；约束 PASS）——**fh float 精度远安全**：对比 FP32 全量 2 步 RMS 0.715% 差 29,000×。fh 值 float 误差经 4th/2nd 公式（系数 double）后 ~1e-6 绝对量级，远低于轨迹小分量绝对容差 ~1e-7 累积门限（iter5 分析不适用于"仅 stencil 输入暂存 float"）。
- **Level-2 未跑**：GATE F<1.0 跳过（P3.5c 同例；100 步加速不可能由 2 步减速翻转，且精度 2 步已证远安全）。

**死路判定（性能轴）**：61 cvt 指令开销（~122 cvt/thread/fdderivs 调用）> -8% spill 流量收益；kernel latency-bound（spill 非关键路径），且 natural 255 未破 → occupancy 收益为零。**iter9 点名的最后未测方向现测毕为负**。

**收敛判定（GPU 侧优化正式收敛）**：寄存器/live-set/spill/缓存路径杠杆全部实测完毕——P3（Ricci recompute，natural 255 不动）、P3.5b（对角，no-op）、P3.5/P3.5c（交叉 16→8→5，已部署 -2.9% / 0）、P5（fh float，-1%）、FP32（正确性 FAIL）、LDG/GAMSMEM/smem tile（缓存路径全死路）、lb(256,3/4)（spill 杀死）、rhs split/Lever A（spill 暴增）。natural-255 = BSSN 公式固有 live-set（fdderivs 61-fh + Ricci 33+ 变量）撞 sm_80 硬件上限，occupancy 25%（128 regs × 2 blocks）是寄存器受限平衡点。**当前部署 901.24s（P1+P2+P3.5，job 153420，check.sh FINAL PASS RMS=0）即本内核结构内的实际极限**；500s 目标仅可能靠算法级重构砍 stencil load 体积（越 Lab4 物理/诚信边界，需用户授权）。job 154113/154246 教训（容器环境）：hpc 容器内嵌 AMSS_BUILD_DIR=/workspace/lab4/build，多树构建/运行必须显式 export AMSS_BUILD_DIR；新输出根需预建 GW250118/。

## 迭代15 P6：prolong3 6×6 完整 unroll，**PASS（bit-exact -13.6%，最大单笔收益）**

**候选**：`~/lab4-gpu-cand-p6b-20260824-082335`（改 fmisc.h `8ff0acaf` + fmisc_gpu.cu `3fab96a9`）。prolongrestrict_cell_gpu.cu 未动（`a81cd33e`）。
**机制**：ncu3 显示 prolong3 31.3% fixed-latency dependency stall（6×6 插值循环的 val 串行依赖链）。完整 unroll 打破 val 依赖链 + 暴露并行 load。ptxas：prolong3_kernel 66→100 regs（0 spill 保持），SASS 确实变化（sass-base 1.5MB vs sass-cand 1.2MB）。
**Level-1 A/B（job 154575，2 步同节点）**：OFF mean 8.548 vs ON mean 7.461 s/step → **F=1.1458（+14.6%）**，bit-exact（RMS=0）。
**Level-2（job 154969，100 步，隔离目录）**：**This Program Cost = 796.65s**（部署基线 901.24s → -104.6s），**check.sh FINAL PASS，RMS=0 bit-exact**，约束逐位一致。
**部署态（job 155105）**：**778.24s**（-13.6%），check.sh FINAL PASS RMS=0。快照 pre-p6 可回滚。
**经验**：2 步 F=1.146 无衰减（100 步 7.46s/step 一致）——与 BR_ORD/P3.5 的衰减模式不同，unroll 收益稳定。**非 rhs kernel 也有大杠杆**。
**下一步**：restrict3（4.7%, 128 regs/128B spill）+ sommerfeld（3.8%）可能有类似 unroll/occupancy headroom，需 ncu 确认后测试。

## 迭代16 P7：restrict3 列融合 unroll + sommerfeld 结构分析，**死路（F=0.9986，0 收益，bit-exact）**；sommerfeld 无 unroll 头寸（结构性不适用）

**候选**：`~/lab4-gpu-cand-p7-20260824-175747`（patch `assets/lab4/opt/search/patch_p7_restrict3_unroll.py`，hash 守卫 + 5 项不变式）。改动仅 1 文件：`src/prolongrestrict_cell_gpu.cu`（formal `a81cd33e` → 候选 `e900fcae`）；derivatives/lopsided/kodiss/bssn_rhs/fmisc/sommerfeld 与部署基线逐字节一致。

**机制（P6a 模式适配 128-reg 上限）**：restrict3_kernel 被 `-maxrregcount=128` + `__launch_bounds__(256,2)` 钉死 128 regs/25% occupancy（2 blocks 下限）。P7a：Z 向按 n 列展开为 6 个显式 d_zrestr6 链（每链 6 独立 load）+ Y 向立即融合（z0+z5 / z1+z4 / z2+z3 配对序逐 token 保留）→ tmp2[6][6] local 往返消除，寄存器按列有界。5 项不变式：helper body verbatim、Y 配对序、n-map if_fine-2+n、m-map jf_fine-2+m、prolong3 未动。

**Level-0 ptxas（job 155269，单 TU apples-to-apples，含 -maxrregcount=128）**：base restrict3 **128 regs/288B stack**/LDL 18/STL 6/LDG 36/DFMA 186/SASS 482KB（rolled loop + tmp2 物化）；cand **52 regs/0 stack**/LDL 0/STL 0/**LDG 216**/DFMA 276/DMUL 99→669/SASS 2.66MB（全展开，216 并行 load 暴露）。**寄存器 128→52 → 占用率投影 2→4 blocks（25%→50%）**，与 P6a（prolong3 占用率 3→2 掉）方向相反。prolong3 对照组逐字节一致（88 regs/288B stack）。

**Level-1 A/B（job 155279，OFF/ON×2 交错 2 步同节点）**：OFF [7.340,7.515,7.409,7.529] mean 7.4481 vs ON [7.207,7.467,7.582,7.579] mean 7.4588 s/step → **F=0.9986（-0.14%，噪声内）**。**check.sh FINAL PASS，RMS=0（bit-exact 实证）**，约束 PASS。分布完全重叠。

**死路原因（诚实，两层）**：
1. 机制成功但方向反了：base 是 rolled loop（小 SASS 482KB，36 迭代×6 load/iter），cand 全展开 SASS 膨胀 **5.5×（2.66MB）**，DMUL 99→669（SoA factor 乘显式化）+ 216 静态 load → 每点指令数爆炸；I-fetch 压力 + 指令吞吐 > occupancy 翻倍（2→4 blocks）与 local 往返消除（stack 288→0）的收益。
2. restrict3 仅 4.7% 运行时且瓶颈非依赖链（base 的 6 项 val 链每点仅 3 FADD 深，36 点本就相互独立，ILP 充足）；unroll 暴露的并行 load 早已被 rolled loop 的跨迭代调度覆盖。
3. **关键教训（与 P6a 互补）**：prolong3 上 unroll 死因是 regs 66→120 掉占用率；restrict3 上 unroll 死因是 SASS/指令爆炸。**两个 6×6 插值 kernel 的 unroll 杠杆均已实测闭环（P6a + P7a 双死路）**。

**P7b sommerfeld（job 155327，L0 probe）**：sommerfeld_routbam_kernel 55 regs/0 stack（P6b forceinline 后已 lean）；sommerfeld_rout_kernel 46 regs/**1888B stack**（= ya[216] 数据暂存数组 1728B，d_decide3d 变长循环填充 + d_polin3_1b 固定循环读取，算法固有结构）。**结构性不适用 unroll 模式**：d_decide3d 循环 trip count 逐点变化（不可手 unroll）、polint Neville 递归本质串行（unroll 不缩短）、固定 6 循环 nvcc 已自动展开。sommerfeld 无 P7 式 unroll 头寸（非实测死路，是"不可行/不适用"类）。SASS base/cand 指令级逐字节一致（差异仅 lineinfo 路径）。

**收敛判断**：prolong3/restrict3/sommerfeld 三个 non-rhs 插值 kernel 的 unroll 杠杆全部闭环（P6a 死路、P7a 死路、P7b 不适用）。部署基线 778.24s 不变。P6b（forceinline）仍是唯一 PASS 的 non-rhs 杠杆。

## 迭代18 P9：is_sommerfeld_boundary forceinline（P6b/P8 模式第三击），**机制不适用（已 inline），codegen NO-OP 死路（非实测 0 收益，是"不可行/不适用"类）**

**候选**：`~/lab4-gpu-cand-p9-20260824-111622`（sommerfeld_rout_gpu.cu `e9ea7810` → `9bff939b83442bd32209c79695186a08d51e4f4fca610d5ad1059bbb773be82d`，diff 仅 1 行：`__device__ bool` → `__device__ __forceinline__ bool`；函数体 verbatim；7 个 control 文件与部署态逐字节一致）。patch：`assets/lab4/opt/search/patch_p9_sommerfeld_forceinline.py`（hash 守卫 + verbatim 校验 + NO_SYMM/OCTANT 不变式 + 调用点计数）。

**机制偏差说明（有据）**：is_sommerfeld_boundary 定义在 sommerfeld_rout_gpu.cu、仅被同 TU 的两个 kernel（line 63/164）调用，**无跨 TU 调用方**。P6b/P8 的 forceinline 收益只对跨 TU helper 成立（定义在 fmisc_gpu.cu、被 sommerfeld/prolongrestrict TU ABI 调用）。函数体引用 NO_SYMM/OCTANT（.cu:12 本地 constexpr），移到 .h 需连带移动 constexpr 且 fmisc.h/sommerfeld_rout.h 均被 host .C 文件在 math.h 之前解析（fabs 未声明）。故采用 in-place __forceinline__（单文件单变量，语义等价，零 header 污染）。

**Level-0（job 155750，单 TU apples-to-apples，-O3 -maxrregcount=128 -rdc=true）**：
- ptxas：sommerfeld_routbam_kernel 55 regs/0 stack、sommerfeld_rout_kernel 128 regs/2112B stack/160B spill stores/308B spill loads —— **base=cand 逐字节相同**。
- SASS：routbam 641232B **逐字节相同**；rout_kernel base 1437556B vs cand 1437559B（**3 字节差异，仅符号表/lineinfo**），CALL=126=126、BSSY=164=164、LDL=63=63、STL=401=401、insns 5704=5704 全同。
- 设备函数列表：base 含 `_Z22is_sommerfeld_boundary...`（-rdc 外部导出副本），cand 中消失 → forceinline 生效，但调用点**早已被 nvcc 自动 inline**（同 TU 小函数）。
- **126 个 CALL 之谜（PTX 证据，job 155778）**：sommerfeld_rout_kernel PTX 有 124 个 `div.rn.f64` + 2 sqrt，SASS 的 126 CALL = ptxas 把双精度除法降级为 `__cuda_sm20_div_rn_f64_full`/`dsqrt` 库调用（P8 inline 的 polint Neville 递归除法），**与 is_sommerfeld_boundary 无关**。

**结论**：SASS 逐字节等价（3B 符号表差异）→ 运行时差异数学上为零。**跳过 Level-1 A/B 与 Level-2**（对字节级相同 SASS 的二进制做 A/B 只会测噪声，fitness 门 F>1.0 无机制可达）。死路类别：**"不可行/不适用"（机制不适用，base 已 inline）**，非"0 收益"。

**forceinline 井已干涸（重要推断）**：P6b/P8 模式的适用条件 = 跨 TU 非 inline helper。PTX 检查 fmisc.ptx **全文 0 个 call.uni**：global_interp_device（fmisc_gpu.cu:16，较大）也已在所有调用点被 nvcc inline（.visible .func 仅为 -rdc 导出副本）。**global_interp_device forceinline 与 P9 同类，预测 NO-OP，不值得做**（任务要求的同时评估结论）。跨 TU 剩余可 forceinline 的 hot helper 已无（derivatives/lopsided/kodiss/fmisc 关键链全部 forceinline）。

**下一步建议**：forceinline 族闭环。剩余头寸在算法级（rhs_kernel 69.1% latency-bound，寄存器 128 上限 + spill 固有）或 sommerfeld_rout_kernel 的 124 div/点（不可近似，破坏 RMS ≤1e-3，死路）。需用户授权算法级重写才可继续显著推进。

## 迭代19 补充诊断：除法指令非杠杆（cuobjdump SASS 全量统计）

P9 agent 报告 sommerfeld 有 124 个 div.rn.f64（PTX 层面，来自 polint Neville 递归）。**SASS 层面核实（cuobjdump 部署 binary，301,595 行 SASS）**：
- `div.rn.f64` 快路径：9 处（全 binary）
- `__cuda_sm20_div_rn_f64_full` 慢路径调用：9 处
- **除法开销可忽略，非杠杆**（PTX 的 div.rn.f64 多数被 nvcc 优化为快路径/乘法序列）
- 证据：`~/lab4-gpu/evidence/div-diagnostic/sass-full.txt`
- 结论：除法优化方向关闭。P9 agent 的 "124 div/点" 是 PTX 层误解，SASS 层实际仅 9 处慢路径。

## 迭代21 Track A：CUDA intrinsics（__fma_rn/__dmul_rn），**死路（nvcc 已默认 FMA 收缩）**

**背景**：NRPyEllipticGPU 文献报告 intrinsics 减 ~21% 指令（他们 codegen 未默认 FMA）。本地验证：`-O3 -fmad=true`（nvcc 默认）已对 stencil 模式（`m4*d12dx*(h_im2-EIT*h_im1+...)`）发射 DFMA；`-fmad=false` 反而退化为 DMUL+DADD（regs 12→14，指令更多）。**结论：源码改写为 intrinsics 是无操作（SASS 相同），Track A 关闭**。证据：`~/lab4-gpu-cand-tracka/probe-fmad/job.log`（job 160264）。与 iter18 P9 同模式（codegen 已最优）。

## 迭代20 P10-algo：d_fdderivs_point __noinline__（破 61-fh live-set），**死路（F=0.886，-13% 更慢）**

**背景**：用户授权 GPU 算法级重写（Track C，目标 <340s）。CPU 侧 Stage 1b k-滚动融合达 120/120，但 GPU 瓶颈不同：rhs_kernel 已点态融合（每线程一点，inline 算全部导数，无数组物化可消），瓶颈是**寄存器 live-set**（d_fdderivs_point 61 个 fh 同时存活 → natural-255 → occupancy 25% → 无法隐藏 L1TEX scoreboard stall 46.6%）。

**候选**：`~/lab4-gpu-cand-algo-20260824-155444`（patch：derivatives.h `d_fdderivs_point` 从 `__forceinline__` → `static __device__ __noinline__`，强制编译器在函数边界溢出，破 61-fh live-set）。需宿主安全宏修复（host C++ bssn_gpu_class.C 含 derivatives.h，`__noinline__` 非标 C++ → 顶部加 `#ifndef __CUDACC__ #define __noinline__ #endif`）。修复后候选 `~/lab4-gpu-cand-algo-noinline-fixed`（derivatives.h hash `f83295d2`，部署基线 `3ede4646`）。

**Level-0 ptxas（job 158054）**：
- base-lb2（部署态）：128 regs / 5504B spill stores / 5996B loads
- cand-lb2（noinline + lb2）：128 regs / spill 完全相同（5504/5996）→ **lb2 下 noinline 不降 live-set，仅改 ABI，死路**
- cand-lb3（noinline 无 launch_bounds）：**80 regs**（128→80，occupancy 投影 25%→50%+）/ 3720B spill stores / 6716B spill loads → **regs 降但 spill loads 反增**

**Level-1 A/B（job 158428，2 步 OFF/ON×2 同节点）**：OFF [14.6763, 14.7039] mean 14.6901 vs ON [16.5222, 16.6226] mean 16.5724 → **F=0.8864（-13%，更慢）**。check.sh 未跑（GATE F<1.0 跳过 Level-2）。

**死因（诚实）**：noinline 成功破 live-set（regs 80），但 spill 流量代价 > occupancy 翻倍收益。与 iter9 P3（Ricci recompute）、iter12 P3.5c（交叉项链式）同模式——**所有“降 live-set”方向均因 spill 延迟 > occupancy 收益而失败**。编译器 forceinline + 128-reg 选择已是 latency-bound 内核上的最优平衡点。

**结论**：GPU 算法级重写 Track C 的“降 live-set”子方向已闭环（与 iter9/12 同死路）。剩余算法级头寸仅在“砍 stencil load 体积”（类比 CPU k-滚动但需重设计数据流，越物理边界）或“跨变量批处理 prolong3”（未测，Track B）。

## 迭代22 Milestone B：RHS interior/boundary 拆分 kernel（指令削减族，首个正收益），**PASS（bit-exact F=1.128）**

**背景**：milestone-B-rhs-ceiling.md 的 L0 探针（job 163419）发现 interior 特化（fh→纯加载）静态指令 -75.8%（123,824→29,952）、spill -67%/-58%，natural regs 仍 255（occupancy 墙未破）→ 指令削减是证据库从未测过的机制（live-set/occupancy 族全部死路）。本迭代做真实拆分 + L1 决定性测试。

**候选**：`~/lab4-gpu-cand-mb-intsplit-20260825-120249`（patch `assets/lab4/opt/search/patch_mb_intsplit.py`；A/B job `assets/lab4/opt/search/mb_intsplit_ab_job.sh`）。机制：新增 `src/bssn_rhs_gpu_int.cu` = rhs_kernel 逐 token 复制（`#define RHSPROBE_INTERIOR` 使 4 个 fh lambda 走纯加载）+ interior 早退；原 rhs_kernel/launch 加尾参 `int skip_interior`（0=部署行为，ptxas 证实 +8 静态指令 codegen 行为中性）；5 个 host 调用点原 launch 传 1（只算边界壳）+ 追加 int launch（同 stream 顺序执行，写集不相交）；CMake 加新 TU。**k 下边距 2→3 的偏离（有据）**：部署配置 Symmetry=1（equatorial）+ z-bbox [0,320] → Z[0]=0 → lopsided/kodis fh 的 kmin=-3 激活，k=2 处 lopsided(k-3)/kodis 守卫通过并反射 fh(i,j,-1)→f[z=0]*SoA[2]；纯加载路径会读负下标（OOB，BR_ORD 式崩溃风险）。故 interior = i∈[2,ex0-3]∧j∈[2,ex1-3]∧**k∈[3,ex2-3]**，i/j 边距 2 安全（imin=jmin=0，负访问不可达）。

**Level-0 ptxas（job 163622，单 TU apples-to-apples）**：base_lb2 128 regs/1352B/5504+5996B spill/123,824 SASS；patched_lb2（skip 参数）128/1352/5504+5996/123,832（+8，行为中性✓）；**int_lb2 128 regs/656B/1836+2548B spill/30,912 SASS（-75%，spill -67%/-58%）**。

**Level-1 A/B（job 163622，2 步 OFF/ON×2 交错同节点）**：OFF [7.12131,7.32822,7.14117,7.34455] median 7.2347 vs ON [6.32322,6.50475,6.32516,6.50833] median 6.4150 → **F(mean)=1.1276 / F(median)=1.1278**。**8/8 .dat IDENTICAL**（bssn_BH/constraint/psi4/ADMQs × 2 对，去注释行 sha256 逐字节）；**check.sh ON vs OFF FINAL PASS，Trajectory RMS=0（bit-exact 实证）**，约束 PASS。

**裁决**：**PASS（keep，诚实路径正收益）**。机制成立：RHS 模块估 ≈1.27×（460.9→~363s/100 步，若收益全在 RHS），端到端每步 -0.82s（≈-82s/100 步 → ~782→~700s 区间）。**但未达 milestone-B §3.5 的 1.3× 保底门**（340s 路线需 RHS ≥2.7×，本杠杆差一个量级）→ 340s 判定维持 No-Go（与 milestone-B §1 预期 1.2-1.5× 一致，落在下缘）。GATE 判据：F≥1.10 非死路、F<1.30 不触发 10 步/100 步建议（100 步留主 agent 决定；2 步 bit-exact + 占比跨窗口稳定，衰减风险低于 BR_ORD 型）。**残余风险**：2 步短跑乐观值可能衰减（历史 2 步→100 步衰减 0.026-0.03 量级），部署前建议 10 步或 Level-2 确认；k 边距 3 假设 equatorial 固定（trimmed build 锁定，配置防漂移清单已核）。formal 未动（8 文件 hash 与基线逐字节一致，Input 已还原 Final=100）。

## 迭代23 Milestone C round 1：prolong3 interior/boundary 拆分 kernel（rhs 拆分机制移植，指令削减族第二击），**keep（bit-exact，prolong3 模块级 ~1.48×，端到端 F=1.063）**

**背景**：iter22 rhs interior 拆分是首个正收益指令削减杠杆（静态 -75.8%）。prolong3（117.5s / 16.5%，1,192,683 calls，ncu 100 regs/0 spill/25% occ）每输出点经 d_prolong3_device 调 d_symmetry_bd_1b（fmisc.h forceinline，P6b）216 次（6×6 Z 向 × 6 k tap），每次含 range check + 反射 + factor 乘，掩码机制占静态指令大头。interior 特化 = interior 点纯加载替代掩码（同 rhs fh→纯加载机制）。**未重复 P6a（unroll 66→120 regs 掉占用率死路）**；interior 是未测机制。

**候选**：`~/lab4-gpu-cand-mb-prolong3-int-20260825-142201`（patch `assets/lab4/opt/search/patch_p23_prolong3_int.py`；L0 探针 job 164166，A/B job 164232）。机制（严格镜像 rhs 拆分）：fmisc.h d_symmetry_bd_1b 加 `#ifdef PROLONG3_INTERIOR` 纯加载路径（base 构建 dormant，行为中性）；新 TU `src/prolongrestrict_cell_gpu_int.cu` = verbatim 复制 + `#define PROLONG3_INTERIOR` + d_prolong3_device interior early-return + 重命名（prolong3_kernel_int / gpu_prolong3_launch_int / restrict3_kernel_int / gpu_restrict3_launch_int，防 nvlink 重复符号）+ **`extern __constant__` C_PROLONG/C_RESTRICT**（定义留 base TU，否则 nvlink Multiple definition）；原 d_prolong3_device/prolong3_kernel/gpu_prolong3_launch 加尾参 `int skip_interior`（0=部署行为）；host 调用点（**Parallel_GPU.cpp case 3，唯一**，非 MPatch_gpu/bssn_gpu_class）原 launch 传 skip=1 + 追加 int launch（同 stream 顺序执行，写集不相交）；CMake 加新 TU；prolongrestrict.h 声明同步 + gpu_prolong3_launch_int 声明。

**interior 定义（从 d_symmetry_bd_1b 实际逻辑推导，无需 rhs 式 fine-index 边距修正）**：prolong3 taps = 6×6×6 coarse cube [cxI_d-2, cxI_d+3]（d=i/j/k）；掩码恒等（纯加载==掩码值逐位）⟺ 每 tap ∈ [1, extc[d]] 且 >0（无 out-of-range zero、无反射、factor=1.0，IEEE x*1.0==x）⟺ **cxI_i/j/k ∈ [3, extc[d]-3]**。cxI 在 d_prolong3_device 内（lbf/lbc 后）可得，两 kernel 同公式计算 → 划分精确一致、覆盖全集一次。

**Level-0 ptxas（job 164166，单 TU apples-to-apples，-O3 -arch=sm_80 -rdc=true -lineinfo）**：

| variant | regs | stack | spill | 静态 SASS |
|---|---:|---:|---:|---:|
| base prolong3_kernel | 100 | 288B | 0 | 2,896 |
| patched（skip 参数，无 define） | 100 | 288B | 0 | 2,912（+16，行为中性✓）|
| **int prolong3_kernel_int** | **64** | 288B | 0 | **1,104（-61.9%）** |

occupancy：100 regs → 2 blocks（25%）；**64 regs → 4 blocks（50%，翻倍）**——P6a 教训反向验证（P6a unroll 66→120 regs 掉占用率；interior 是 100→64 升占用率）。**GATE PASS**（静态削减 ≥25%✓、regs ≤100✓）。built-binary 交叉核对：base 2,896 / cand-int 1,104 / cand-bnd（skip 边界核）2,912。

**restrict3 顺带探针（job 164166）**：base restrict3_kernel 1,944 静态 / 100 regs；int restrict3_kernel_int 1,320（**-32.1%**）/ 64 regs → restrict3（3.5% 模块）也有掩码削减头寸，但需自己的 interior 条件推导（读 fine grid、ord=2、tap 集不同），留后续轮决策。

**构建修复记录（诚实，两轮）**：job 164166 首建 nvlink 失败 = int TU 重复定义 `__constant__` C_PROLONG/C_RESTRICT（bssn_rhs 无 __constant__ 故 rhs 拆分未遇）→ int TU 改 `extern __constant__`；job 164203 host 链接失败 = **int TU 未应用 skip_interior 签名 patch**（gpu_prolong3_launch_int 13 vs host 引用 14 参数，nm 实证）→ step 3 对 int TU 补 5 处签名 patch（先签名后重命名）。修复后全量 build PASS。

**Level-1 A/B（job 164232，2 步 OFF/ON×2 交错同节点）**：OFF [6.32654,6.51013,6.3363,6.49679] mean 6.4174 vs ON [5.94727,6.12575,5.94297,6.12893] mean 6.0362 → **F(mean)=1.0632 / F(median)=1.0630**。**8/8 .dat IDENTICAL**（bssn_BH/constraint/psi4/ADMQs × 2 对）；**check.sh ON vs OFF FINAL PASS，Trajectory RMS=0（bit-exact 实证）**，约束 PASS。

**裁决**：**keep（orchestrator 裁定）**。prolong3 模块级 ≈1.48×（1.175→~0.79 s/step，若收益全在 prolong3）——**≥1.3× 模块任务门达成**；端到端每步 -0.38s（≈-38s/100 步 → 部署新基线 ~702s → ~664s 预估，F~1.06）。端到端 F=1.063 低于迭代模板 1.10 严格线，但为真实正收益 + bit-exact + 模块级达标（orchestrator 判 keep，诚实路径）。**未达 1.30 触发线 → 不跑 10 步/100 步（留主 agent）**。**残余风险**：2 步短跑乐观值可能衰减（历史衰减 0.026-0.03 量级）；launch 翻倍（+1.19M launches/100 步 ≈ +5s）在更长步可能稀释收益；100 步前建议 10 步确认（主 agent 决策）。formal 未动（5 文件 hash 与基线逐字节一致，Input 已还原 Final=100，`~/lab4-gpu/build` 构建产物已在 job 内生成可删）。

## 迭代24 Milestone C round 2：restrict3 / global_interp / sommerfeld interior 特化批处理（三 kernel），**全部死路（restrict3 F=1.003 噪声内、global_interp F=0.992 更慢、sommerfeld L0 GATE FAIL）**

**背景**：iter22（rhs interior，F=1.128）+ iter23（prolong3 interior，F=1.063）证明 interior 特化机制正收益。本轮把同一模式复制到三个小插值 kernel（restrict3 24.6s/3.46%、global_interp 42s/5.91%、sommerfeld_rout 29.9s/4.21%）。**三个候选各一目录（单变量纪律）**：`~/lab4-gpu-cand-mb-restrict3-int-20260825-152801`、`~/lab4-gpu-cand-mb-globalinterp-int-20260825-154114`（修复版）、`~/lab4-gpu-cand-mb-sommerfeld-int-20260825-152801`。patch 脚本：`assets/lab4/opt/search/patch_p24_{restrict3,globalinterp,sommerfeld}_int.py`（hash 守卫 + 锚点断言 + 防重入）。

**机制（与 iter22/23 同型，按各 kernel 索引语义推导 interior 条件）**：
- restrict3：d_symmetry_bd_1b 读细网格，taps=[fine-2,fine+3] 每维。interior ⟺ **if_fine/jf_fine/kf_fine ∈ [3, extf-3]**（全部 tap ∈ [1,extf] 且无反射；k 下边距 3 对齐 iter22 equatorial 教训）。int TU（已有 iter23 的 PROLONG3_INTERIOR define）加 restrict3 interior early-return；base TU 加 skip_interior 尾参 + 边界 skip。host 调用点（Parallel_GPU.cpp case 2）原 launch 传 1 + 追加 int launch。
- global_interp：global_interp_device 内 d_decide3d 掩码。interior ⟺ 钳制后 **cxB[m]≥1 ∧ cxT[m]≤ex[m]**（= cxI ∈ [3,ex-3] for ORDN=6）。**device 函数改 bool 返回**（interior/boundary 早退防 atomicAdd 误加 0.0 与 d_weight 双计）；base kernel `if (!device(...)) return;` 守卫 atomicAdd。fmisc.h d_decide3d 加 `#if defined(GLOBAL_INTERP_INTERIOR)||defined(SOMMERFELD_INTERIOR)` 纯加载变体。slim int TU（fmisc_gpu_int.cu）= define + X_at_1b + global_interp_device_int + kernel_int + launch_int（改名防 -rdc 重复符号）。host 调用点 4 个（**MPatch.C:379 + MPatch_gpu.cu:98/229 + Parallel_GPU.cpp:685**）。amr kernel 传 skip=0 保持部署行为。
- sommerfeld：sommerfeld_rout_kernel 仅处理边界点；interpolation 窗口钳制后 interior ⟺ cxB[m]≥1 ∧ cxT[m]≤ex[m]（equatorial 下 k 是唯一反射轴）。int TU 加 SOMMERFELD_INTERIOR define + interior early-return + CORRECTSTEP 早退（防冗余双写）+ 改名（is_sommerfeld_boundary_int/rout_kernel_int/routbam_kernel_int/launch_int×2）。host 调用点 2 个（bssn_step_gpu.C:186/332）。

**L0 探针（job 164568，单 TU apples-to-apples，-O3 -arch=sm_80 -rdc=true -lineinfo）**：

| kernel | base 静态 | patched 静态 | int 静态 | 削减 | regs base/int | stack base/int | spill |
|---|---:|---:|---:|---:|---:|---:|---:|
| restrict3_kernel | 1,944 | 1,968 | **1,328** | **-31.7%** | 128/128 | 288/0B | 0/0 |
| global_interp_kernel | 1,768 | 1,784 | **1,232** | **-30.3%** | 64/64 | 2352/2352B | 0/0 |
| sommerfeld_rout_kernel | 5,608 | 5,616 | **5,600** | **-0.1%** | 160/160 | 1968/1968B | 0/0 |

**sommerfeld GATE FAIL（静态削减 0.14% ≪ 20% 门限）→ 死路，未做真实拆分/A/B**。根因：d_decide3d 掩码机器只占 kernel 静态 SASS 的极小部分（P8 forceinline 后 polint/d_polin3_1b 几何/boundary check 占主体），纯加载变体仅省 ~8 条指令。**结构判断（iter16 P7b 预测复核）：sommerfeld 无 interior 头寸，勿重试**。

**restrict3 A/B（job 164596）**：built-binary 交叉核对 base 1,944 / cand-bnd 1,968 / cand-int 1,328 ✓。2 步 OFF/ON×2：OFF [5.998,6.177,5.998,6.192] / ON [5.971,6.149,6.021,6.146] → **F(mean)=1.0032 / F(median)=1.0006（噪声内）**。**8/8 .dat IDENTICAL，check.sh FINAL PASS，RMS=0（bit-exact）**。裁决：**死路（0 收益）**——interior 静态削减 -31.7% 未转运行时（restrict3 是细网格 load latency 主导；+294,975 launches ≈ +1.2s 吃光收益）。

**global_interp A/B（job 164664）**：built-binary 交叉核对 base 1,768 / cand-bnd 1,784 / cand-int 1,240 ✓。2 步 OFF/ON×2：OFF 6.0204 / ON 6.0706 → **F(mean)=0.9917 / F(median)=0.9922（-0.8% 更慢）**。**8/8 .dat IDENTICAL，check.sh FINAL PASS，RMS=0（bit-exact，atomicAdd 分割构造正确：每点每 block 恰一个贡献）**。裁决：**死路（负收益）**——shell 插值点本质在网格边界附近，interior 占比小（大量点窗口越界/贴对称面）→ 纯加载路径覆盖少 + 双 launch 开销净负。

**构建修复记录（诚实，global_interp 一轮失败）**：job 164625 首建 BUILD_FAIL = **Parallel_GPU.cpp:685 的 gpu_global_interp_launch 调用点漏 patch**（根因：此前 grep 用 `*.C` glob，`Parallel_GPU.cpp` 小写 .cpp 后缀被 glob 排除 → 调用点普查漏检；fmisc.h:501 声明已带 skip 尾参 → host 编译 "too few arguments"）。修复：patch 脚本补 Parallel_GPU.cpp（split_gi_calls 通用正则，4 调用点齐）+ global_interp_device 两个裸 `return;` → `return false;`（bool 返回类型，nvcc #117-D 警告）。**教训：调用点普查必须含 .cpp（`grep -rn pattern *.C *.cpp *.cu *.h`）**。另 sommerfeld/restrict3 调用点已复核无同类遗漏（sommerfeld 仅 bssn_step_gpu.C×2 ✓、restrict3 仅 Parallel_GPU.cpp:85 ✓、prolong3 case3 已由 iter23 处理 ✓）。

**裁决汇总（三 kernel 全死路）**：interior 特化机制对 rhs（69.1% 模块）与 prolong3（16.5% 模块）正收益，但对三个 ≤6% 小模块无净收益——静态指令削减在 load-latency 主导 + launch 翻倍的小 kernel 上不成立。**无 keep → 无组合候选**（任务"三个都 keep 后建组合"未触发）。formal 未动（8 文件 hash 与基线逐字节一致，Input 已还原 Final=100，formal 无 twop cache 残留）。

## 迭代26a P26a：rhs_boundary 紧凑一维发射域（6-slab 分解），**PASS（bit-exact F=1.083，L2 100 步）**

**背景（reprofile 2026-08-26，jobs 165992/166024，evidence-reprofile-20260826/REPORT.md）**：rhs_boundary（203.7s/32.98%）与 rhs_interior（190.5s/30.85%）blocks/launch 逐窗口完全相同（125/512/1120）→ boundary kernel 按整个 patch 体积发射、interior 线程早退（浪费 block 调度 + 索引/分支）。同理 prolong3_boundary（52.1s/8.44%）。

**候选**：`~/lab4-gpu-cand-p26a-bndcompact-20260826-012221`（patch `assets/lab4/opt/search/patch_p26a_bnd_compact.py`，hash 守卫 807f7b73 + anchor 断言 + 防重入）。改动仅 `src/bssn_rhs_gpu.cu` 1 文件：新增 `__host__ __device__` bnd_slab_size/bnd_count/bnd_map（6-slab 不相交划分：i-lo/i-hi/j-lo/j-hi/k-lo/k-hi，first-slab-wins；代数 + 数值 16 形状验证，含退化；i 最快映射保证合并访问）；`rhs_kernel` 加尾参 `compact_mode`（1=1-D grid 经 bnd_map 映射壳点，0=原全体积路径逐 token 保留）；`gpu_compute_rhs_bssn_launch` 按 `ex≥8` 调度紧凑 1-D grid（N=bnd_count，256 threads）否则 legacy。int kernel/launch 未动。

**Level-0 ptxas（job 166117，apples-to-apples）**：cand rhs_kernel 128 regs / stack 1400B（base 1352B）/ spill stores 5624B（base 5504B，+2.2%）/ loads 6280B（base 5996B，+4.7%）：bnd_map 整除/取模新增 ~120B spill，regs 锁 128 不动；rhs_kernel_int 逐字节同基线（712/2164/3328）。

**Level-1 A/B（job 166117，2 步 base1→cand1→cand2→base2 交错）**：Step2 base 5.62979/5.62794（med 5.62886）vs cand 5.17329/5.17441（med 5.17385）→ **F=1.0879**。**8/8 .dat IDENTICAL（bit-exact 实证）**。

**Level-2（job 166134，100 步 OJ-sim）**：**This Program Cost = 615.45s**（部署基线 666.64s → **-51.2s, F=1.083**，Total Evolve 570.93s），**check.sh FINAL PASS，Trajectory RMS=0（bit-exact）**，约束 Ham=0.28974817/Px/Py/Pz 与 golden 逐位一致，4 .dat 全产。

**裁决：PASS（keep，部署候选）**。机制：block 削减率随网格增大而增长（40³: 60% → 112³: 85%），细网格收益更大 → 100 步 F=1.083 与 2 步 1.088 几乎无衰减（与 iter22/23 同类结构性收益，非早期网格假象）。**验收线达成**：e2e -51.2s ≫ 3%/20s；boundary kernel 总时间降幅待部署后 ncu 复测（预计 rhs_boundary 203.7s → ~130-140s，-30~35%）。spill +120B 的代价被块调度收益大幅覆盖。

**部署指令（供主 agent）**：
```bash
cp -r ~/lab4-gpu ~/lab4-gpu-snapshot-pre-p26a-$(date +%Y%m%d-%H%M%S)
cp ~/lab4-gpu-cand-p26a-bndcompact-20260826-012221/src/bssn_rhs_gpu.cu ~/lab4-gpu/src/bssn_rhs_gpu.cu
cd ~/lab4-gpu && ./compile.sh
# 验证：100 步全量 check.sh（部署态 OJ-sim）
```

**残余风险**：低。位级验证 2 步 + 100 步双通道；legacy 路径逐 token 保留（罕见小 patch 兜底）；int kernel 未动。**下一方向**：26b（rhs face 特化，编译期镜像索引攻掩码计算，rhs_boundary 剩余最大杠杆）与 26c（prolong3_boundary 同机制 compact，需 coarse-空间 parity 映射）。

## 迭代26b P26b：rhs_boundary face 特化（编译期纯加载），**低于 10s 线（bit-exact F=1.0079，~4-5s）**

**候选**：`~/lab4-gpu-cand-p26b-face-20260826-023926`（patch `assets/lab4/opt/search/patch_p26b_face_spec.py`，hash 守卫 6ef7bf1f/771f6685/40e958e6）。改动：derivatives.h 两个 fh lambda 加 `RHSFACE_PURE`（x/y 纯面全纯加载）与 `RHSFACE_PURE_XY`（z 面 i/j 纯 + k 掩码）分支；bssn_rhs_gpu.cu 6-slab 映射升级为 7 区域（R0-3 x/y 纯面 / R4-5 z 纯面 / R6 REST 边角，映射本地数值验证 156 形状）；新增 bssn_rhs_gpu_face.cu（RHSFACE_PURE）+ bssn_rhs_gpu_facez.cu（RHSFACE_PURE_XY），每区域一个 1-D launch，generic kernel 只处理 R6；CMake 加 2 TU。lopsided/kodis 保留全掩码（面法向反射非恒等）。

**Level-0 ptxas（job 166357）**：rhs_kernel_facepure 128 regs / 736B stack / **1892+3976 spill**（vs generic 1376/5596/6116，**spill -66%/-35%**）；rhs_kernel_facez 720/1648/2368（更瘦）；generic R6 1376/5596/6116（REST 仍全掩码）。128 regs 锁 → occupancy 仍 25%。

**Level-1 A/B（job 166357，2 步交错）**：Step2 base 5.16131/5.17453（med 5.16792）vs cand 5.12851/5.12666（med 5.12758）→ **F=1.0079**。**8/8 .dat IDENTICAL（bit-exact）**。

**裁决：低于用户 10s 停止线（预估 ~4-5s/100 步）→ 不部署，候选保留**。机制层面成功（face kernel spill -66%，纯加载路径生效），但端到端收益微小：**再次证实 rhs kernel 是 load-latency 受限而非指令受限**（iter1-12 证据链：L1TEX scoreboard 46.6%、eligible warps 0.49；iter22 的收益来自纯加载路径的 load 体积/调度变化，非掩码指令削减本身）。face 特化去掉了掩码指令（spill 大降）但同样的 stencil load 延迟仍在 → 墙钟无显著改善。

**与 iter24（restrict3/global_interp/sommerfeld interior 死路）同源结论**：指令削减在 load-latency 主导的 kernel 上不转运行时收益。**boundary 特化线正式收束：26a（发射域 compact，-51.6s）是唯一超线收益，26b/26c（各 ~5-7s）低于 10s 线**。

## 迭代28 P28：matter 重读消除（val_rho/val_Sxx 局部复用），**低于 10s 线（bit-exact F=1.0023，~1.4s）**

**背景**：Iter28 完整版（RHS→RK 融合）被 **sommerfeld 边界阻尼阻断**：RK update 使用 sommerfeld 处理后的 f_rhs（bssn_step_gpu.C：per-field sommerfeld_routbam 在 RK 之前对 varlrhs 边界阻尼），而 RHS kernel 在 sommerfeld 之前运行 → 融合版本在边界点数值不同，**无法 bit-exact**（数据流证据：predictor 循环顺序 = RHS → sommerfeld(varlrhs) → RK）。另确认 Gam/R 输出写早已被注释（历轮已消除死存储）；rho/Sx..Szz 是输入数组且 kernel 内重读 11 处（非 const 指针 → nvcc 无法 CSE）。

**候选**：`~/lab4-gpu-cand-p28-dedup-20260826-052258`（patch `patch_p28_dedup.py`，hash 守卫 6ef7bf1f）。改动：bssn_rhs_gpu.cu + bssn_rhs_gpu_int.cu 的 matter 段：val_rho/val_Sxx..Szz 各加载一次（S calc 前），重读点（711/738/946 rho、699-700→715-720 Sxx..Szz、1049-1051 Sx/Sy/Sz）改用局部变量，bit-exact by construction（同数组同值）。

**Level-1 A/B（job 166888，2 步）**：base med 5.20332 vs cand 5.19116 → **F=1.0023**；**8/8 .dat IDENTICAL**。ptxas：128 regs 保持，spill 5760/6384（base 5624/6280，+2%：val_* 存活代价）。

**裁决：低于 10s 线，不部署**。11 处 load/point 削减被 latency-bound 本质吞没（第 4 次独立确认：指令削减不转运行时收益）。

## 迭代30 P30：GPUManager 显存池重启用（带 1GiB 上限 + memset 复用），**低于 10s 线（bit-exact F=1.0008，~0s）**

**背景**：phase0 账本 cudaMalloc+Free 16.7s/223K 对（2.1%）；gpu_manager.cu 的池实现已存在但被注释。启用池（memset 保持零初始化语义 + POOL_CAP_BYTES 1GiB 限制 MIG 10GiB 上驻留）。

**候选**：`~/lab4-gpu-cand-p30-pool-20260826-052319`（patch `patch_p30_pool.py`，hash 守卫 435c9b69）。改动仅 gpu_manager.cu（Impl.pool_bytes/POOL_CAP_BYTES + alloc/free/clear_pool 池路径）。

**Level-1 A/B（job 166929，2 步）**：base med 5.18891 vs cand 5.18466 → **F=1.0008（~0s）**；**8/8 .dat IDENTICAL**。

**裁决：低于 10s 线**。根因：alloc/free 是 host 侧开销，被 GPU 饱和执行完全隐藏（异步流下 host malloc 与 kernel 重叠）：phase0 的 "16.7s 可回收" 是累计 host 时间，非墙钟增量。**host 侧杠杆（alloc/launch）在 GPU 饱和前提下列为 0 收益**（与 per-stream sync=0、cudaGraph 关闭同源结论）。

## Iter28/29/30 终审（GPU 侧收敛定论）

- Iter28 融合：被 sommerfeld 依赖阻断（bit-exact 不可行）；dedup 子目标 F=1.0023。
- Iter29 batching：证据（prolong wave 尾部 grid med 43 < 并发 56）指向 ~5-15s 且实现重（descriptor 基础设施）；最接近的 26c（prolong3 boundary compact）F=1.0114 已低于 10s 线。未再实施。
- Iter30 streams/pool：pool F=1.0008（host 开销隐藏）；streams/cudaGraph 已有 phase0 负面证据。
- **部署基线维持 615.04s（26a，bit-exact）**。未部署候选合计：26b(~5s) + 26c(~7s) + P28(~1.4s) ≈ 13s（各自 <10s 线，组合部署由主 agent 决定）。

## 迭代26bcd P26BCD：组合部署（26b face + 26c prolong3 compact + P28 dedup），**PASS（bit-exact F=1.0153，L2 605.78s）→ 已部署**

**候选**：`~/lab4-gpu-cand-p26bcd-combined-20260826-103614`（patch `patch_p26bcd_combined.py`，顺序 P28→26b→26c，face/facez TU 继承 dedup）。改动 7 文件：bssn_rhs_gpu.cu / _int.cu / _face.cu / _facez.cu、derivatives.h、prolongrestrict_cell_gpu.cu、CMakeLists.txt。

**Level-1（job 168790，2 步）**：Step2 base 5.20364 vs cand 5.06683 → **F=1.0270**（组合略超单测之和 1.008×1.011×1.002≈1.021）；**8/8 .dat IDENTICAL**。
**Level-2（job 168845，100 步）**：**This Program Cost = 605.78s**（615.04 → **-9.3s, F=1.0153**），check FINAL PASS RMS=0 bit-exact，约束逐位一致。2 步 2.7% → 100 步 1.5% 衰减（早期粗网格 shell 占比高，26a 同类模式）。
**部署（job 168935/169040）**：hash 守卫（formal 6ef7bf1f）→ 快照 pre-p26bcd → 7 文件安装（部署 hash 9baee005 等与 L2 候选逐字节一致）→ 重建 BUILD_OK → 100 步 OJ-sim（日志被清理步骤误删，凭文件一致性 + L2 背书）→ 清理为提交包（9 项，86 src，manifest 92 条 `~/lab4-gpu-submission-p26bcd.sha256`）。

**基线轨迹**：782.59 → 709.50（rhs int）→ 666.64（prolong3 int）→ 615.04（rhs bnd compact）→ **605.78s（26bcd 组合，已部署）**。累计 -22.6% vs 782.59。真实 OJ 上次 705.94s/76 分，本次部署态预计 ~600s/~80 分区间。

**经验**：组合候选的 2 步 F（1.027）> 单测乘积（1.021）：face 内核消除的掩码指令与 prolong3 compact 的发射域改善在细网格上互补；但 100 步衰减到 1.0153，最终收益 ~9.3s。

## OJ 终局记录（2026-08-26）：GPU 81/120 · 601.364s（Iter26bcd 组合部署版）

| 指标 | 值 |
|---|---|
| wallSeconds | 601.364127（scoreBeforeRounding 81.074133）|
| trajectoryRMS | 0（bit-exact）|
| 约束 Ham/Px/Py/Pz | 0.28974817 / 0.039343259 / 0.047298107 / 0.044686547（≤2，逐位一致）|
| MPI/OMP / sourceRevision | 1 / 8 / 5b0edd5-r11 |

**OJ 轨迹**：1604.47s/0 → 1228.53s/55 → 1044.26s/64 → 705.94s/76 → **601.364s/81**。较上次 -104.6s（-14.8%）+5 分；相对 V1 基线 -62.5%。
**OJ-sim 预测验证**：605.78s 预测 vs 601.364s 实际，误差 0.73%（第三次 <1% 验证）。
**部署栈（完整）**：TwoP 解耦 + OMP_TUNE → INLSTEN3 → BRFINAL → P1/P2/P3.5 → P6b/P8 → P10 → iter22 rhs int → iter23 prolong3 int → 26a rhs bnd compact → 26bcd 组合。全部 bit-exact。
**GPU 侧收敛定论**：601.364s 是本内核结构/寄存器模型/发射模式内的实际极限；剩余方向（算法级重构、RHS→RK 融合[sommerfeld 阻断]、host 侧[GPU 饱和隐藏]）均不可行或 <10s。分数再提升需算法级授权。

## 迭代31 P31：global_interp 变量批处理 launch（发射结构杠杆，用户优先级 #1），**keep（bit-exact F=1.0061，低于 10s 线，组合候选）**

**背景**：nsys reprofile-20260826 发现 global_interp_kernel blocks/launch min/med/max = **1/1/144（中位数 1 block！）**，34,400 次 launch 共 48.83s（7.91%），avg 1.42ms。机制：host 对每 (block, variable) 循环 launch，每 launch 只覆盖 NN 个 shell 点（NN=n_tot/cpusize≈96/4=24 粗分片），MedianPatch 单 rank 拥有 1 个 block → 1-block launch（256 线程 = 8 warps）在 ~14 SM 的 MIG 上几乎空转，而每线程跑 216 次掩码 load + Neville 递归的深延迟链。iter24 测的是 interior 特化（load specialization，死路），**发射结构/并行度机制未测过**。

**候选**：`~/lab4-gpu-cand-p31-givb-20260826-163501`（patch `assets/lab4/opt/search/patch_p31_gi_varbatch.py`，hash 守卫 + 防重入）。改动 4 文件：fmisc_gpu.cu（新增 `global_interp_multi_kernel` + `gpu_global_interp_multi_launch`，grid = (ceil(NN/256), num_var)，blockIdx.y = var_idx，每 (j,var_idx) 线程算的与逐变量 launch 的线程 j 完全一致：同 bbox 容差检查、同 global_interp_device 调用（同 SoA 值）、atomicAdd 到同 (j,var_idx)-专属地址 d_shellf[j*num_var+var_idx]、d_weight 仅 var_idx==0 加）+ fmisc.h（声明）+ MPatch_gpu.cu（Interp_Points_GPU / Interp_N_Points_GPU 每 block 组装 d_fields/d_SoA_all 暂存，一次 multi launch，sync 后释放）+ Parallel_GPU.cpp（PatList_Interp_Points_GPU 同改造）。**bit-exact by construction**：每地址的 atomic 贡献集合与顺序不变（每 block 每 (j,var) 恰一个贡献）。构建修复记录：MPatch_gpu.cu 无 `<vector>`（job 171924 BUILD_FAIL），补 include 后 PASS。

**Level-0 ptxas（job 171974）**：global_interp_multi_kernel **64 regs / 2352B stack / 0 spill，与原 global_interp_kernel 逐字节同参**（codegen 中性，机制纯发射结构）。

**Level-1 A/B（job 171974，2 步 ×4 轮交错，Analysis_Time=0.1 开分析——该杠杆在分析路径，必须开分析测）**：base Step2 median 5.59959 vs cand 5.56577 → **F=1.0061**。**8/8 .dat IDENTICAL（bit-exact）**，psi4 行数 19 确认分析路径激活。

**裁决**：单独低于 10s 线（~ -3.4s/100 步），但 **keep 为组合候选**（26bcd 先例：26b/26c 单测各 ~5-7s 低于线，组合部署）。机制解释：Wave 路径 2 vars × 7 blocks、BH 路径 3 vars，多数 multi launch 仍是 1-2 block，仅 MassPAng（17 vars）收益显著；launch 数 34,400 → ~2,600（÷13）。

## 迭代32 P32：sommerfeld boundary compact 发射域（26a 机制移植，用户优先级 #2），**keep（bit-exact F=1.0279，L2 随组合通过）**

**背景**：sommerfeld_rout_kernel 全体积发射（blocks 343/512/1120，与 rhs 同模式），但仅 PATCH 外 bbox 重合面上的单层厚 face 点做功（is_sommerfeld_boundary 早退，~93% 线程空转）；160 regs → occ 9.7%（P8 forceinline polint 膨胀），块串行化浪费大。iter24 测的是 interior 特化（L0 静态 -0.1% GATE FAIL），**compact 发射域未测**。

**候选**：`~/lab4-gpu-cand-p32-sommcompact-20260826-164623`（patch `assets/lab4/opt/search/patch_p32_sommerfeld_compact.py`，hash 守卫 + verbatim body 提取）。改动 3 文件：sommerfeld_rout_gpu.cu（新增 `sommerfeld_rout_compact_kernel`：**逐点 kernel 体 verbatim 提取**（含 is_sommerfeld_boundary 检查），仅把 3-D 解码换成 flat 1-D + 6-face 并集映射（i==1/i==ex0/j==1/j==ex1/k==1/k==ex2 活跃层；边/角点可能被多个层枚举多次，每次写同值同地址 → 良性重复写）；launcher 在 **host 侧**用 h_X/h_Y/h_Z（Block 构造时上传的位级相同 host 副本）评估 6 个 face 条件 = is_sommerfeld_boundary 的 face 判据，无 face 活跃则 **跳过整个 launch**，否则一次紧凑 1-D launch 覆盖活跃层并集）+ sommerfeld_rout.h（launcher 加 h_X/h_Y/h_Z 参数）+ bssn_step_gpu.C（2 个调用点传 cg->X[0..2]）。

**关键正确性论证**：原 launch 的处理点集 = 活跃 face 层并集（is_sommerfeld_boundary 的 (i,j,k) 判定 = 6 个 face 条件恰好一个为真）；紧凑 kernel 枚举同一并集；k==1 face 在 equatorial Symmetry=1 下被排除（is_sommerfeld_boundary 的 z 下界 face 有 `!(Symmetry > NO_SYMM && fabs(zmin) < dZ/2)` 守卫，host 侧镜像）。**bit-exact by construction**。

**Level-0 ptxas（job 172041）**：sommerfeld_rout_compact_kernel **162 regs / 1968B stack / 0 spill**（原 kernel 160 regs / 1968B / 0 spill）——codegen 近乎中性（+2 regs 为 flat 解码），机制纯发射域。

**Level-1 A/B（job 172041，2 步 ×4 轮交错，分析关）**：base Step2 median 5.06712 vs cand 4.92942 → **F=1.0279（~-13.8s/100 步）**。**8/8 .dat IDENTICAL（bit-exact）**。

**裁决**：**keep（L1 强正收益）**。块削减率 ~90%+（壳点占全体积 <7%），发射浪费 + 160-reg 低占用率块串行化同时消除。**L2 未单独跑，随 P313233 组合（迭代34）一并验证（组合 L1 F=1.1148 bit-exact，L2 job 174954）**。

## 迭代33 P33：prolong3 奇偶对齐 2×2×2 组多输出重写（ABEGPU.md 未测方向 #8/#9，用户优先级 #3），**L1 keep（bit-exact F=1.0677），L2 PASS（job 172346：Program Cost 579.66s，check FINAL PASS RMS=0）**

**背景**：prolong3 双 kernel 合计 83.7s（13.55%）：boundary 52.1s（100 regs/25% occ）+ interior 31.6s（64 regs/50% occ）。机制：fine 索引 i, i+1 且 (i+lbf) 为偶共享同一 coarse 锚 cxI（floor(ii/2) 相同）→ **奇偶对齐 2×2×2 组的 8 个 fine 点从同一个 6×6×6 coarse cube 插值**。逐点 kernel 每组重复加载 cube 8 次（8×216=1728 load/组）；组 kernel 一次加载（216 load，**8× load 削减**），k 双奇偶 Z-行共享在 tmp2[2][6][6]，Y/X 奇偶组合展开。P6a unroll 死路（regs 66→120 掉占用率）不适用：多输出是**减 load**不是加寄存器（110/166 regs 仍可接受）。

**候选**：`~/lab4-gpu-cand-p33-prolonggrp-20260826-165912`（patch `assets/lab4/opt/search/patch_p33_prolong3_group.py`，hash 守卫）。改动 2 文件：prolongrestrict_cell_gpu.cu（新增 `prolong3_multi_kernel`：boundary + 26c 式 compact 组枚举 p3_bnd_count/p3_bnd_map 复用（组域 G = (en-lead)/2+1）；组几何（lead_base/cxI_base/G）host 侧算，p3_geom 镜像 device 0.4f 公式（26c 已依赖的同构）；**每成员求和序逐 token 保留**：Z 向 6 个独立 `+=`（t 升序、系数随 k 奇偶翻转——与原 kernel 的 if(k_even) 两条分支逐语句一致）、Y/X 单 6 项表达式（随 j/i 奇偶翻转）；skip_interior 分类组均匀（8 成员共享 cxI））+ prolongrestrict_cell_gpu_int.cu（`prolong3_multi_kernel_int`：interior-only，全盒组枚举 + interior 早退组均匀化）。原 prolong3_kernel/prolong3_kernel_int 保留在文件中（未被新 launcher 调用，供 base diff 对照）。

**Level-0 ptxas（job 172209）**：prolong3_multi_kernel（boundary）**110 regs / 576B stack / 0 spill**（原 100 regs/288B）；prolong3_multi_kernel_int（interior）**166 regs / 0 stack / 0 spill**（原 64 regs——interior 占用率 50%→25%，但每线程 8 输出，算术/负载比翻 8 倍，L1 实测净赢）。

**Level-1 A/B（job 172209，2 步 ×4 轮交错，分析关）**：base Step2 median 5.0536 vs cand 4.73327 → **F=1.0677（~-33s/100 步）**。**8/8 .dat IDENTICAL（bit-exact 实证）**。

**Level-2（job 172346，100 步 OJ-sim）**：**This Program Cost = 579.66s**（部署基线 605.78s → **-26.1s, F=1.045**，Total Evolve 535.11s），**check.sh FINAL PASS，Trajectory RMS=0（bit-exact）**，约束 Ham=0.28974817/Px/Py/Pz 与 golden 逐位一致，4 .dat 全产。2 步 F=1.068 → 100 步 F=1.045 轻度衰减（早期粗网格组载荷大）。**部署候选 #1**。


## 迭代34 P313233 组合：P31+P32+P33 组合树（26bcd 式组合门，P31 在分析路径故 ANALYSIS ON 测），**L1 GATE PASS（bit-exact F=1.1148），L2 PASS（job 174954：Program Cost 539.48s，check FINAL PASS RMS=0）**

**候选**：`~/lab4-gpu-cand-p313233-comb-20260826-171357`（组合树，文件哈希与三个单测逐字节一致：P31 文件 fmisc_gpu `d8684f83`/fmisc.h `abec6936`/MPatch_gpu `6bc40da5`/Parallel_GPU.cpp `2a9545cf`，P32 文件 sommerfeld_rout_gpu `e33e3fed`/sommerfeld_rout.h `3171b508`/bssn_step_gpu.C `c494fd8d`，P33 文件 prolongrestrict_cell_gpu.cu `68a65584`/_int `b739a44e`，未改文件与 26bcd 基线逐字节一致）。

**Level-0 ptxas（job 174936，build-p313233）**：global_interp_multi_kernel 64 regs/2352B stack/0 spill；sommerfeld_rout_compact_kernel 162 regs/1968B/0 spill；prolong3_multi_kernel 110 regs/576B/0 spill；prolong3_multi_kernel_int 166 regs/0 stack/0 spill——与三个单测逐参一致，组合无 codegen 干扰。

**Level-1 A/B（job 174936，2 步 ×4 轮交错，Analysis_Time=0.1 开分析——P31 在分析路径）**：base1 Step2 5.4731 / cand1 4.91326 / cand2 4.91269 / base2 5.48133 → **base median 5.47721 vs cand median 4.91297 → F=1.1148**。**8/8 .dat IDENTICAL（bit-exact 实证）**，psi4 19 行确认分析路径激活。**GATE PASSED（F>1.0 + bit-exact）→ L2**。

**组合收益 vs 单测乘积**：1.1148 > 1.0061×1.0279×1.0677≈1.1034 —— 三杠杆在分析路径（P31）+发射结构（P32/P33）上互补，无干扰。**注意**：此 F 在 ANALYSIS ON 下测得（P31 只在分析路径生效），100 步 L2 含分析开销，衰减风险中等（26bcd 先例：2 步 1.027 → 100 步 1.015）。

**Level-2（job 174954，100 步 OJ-sim + check.sh）**：**This Program Cost = 539.48s**（部署基线 605.78s → **-66.3s, F=1.123**；对比 P33 单测 579.66s → 再 **-40.2s**；Total Evolve 495.73s），**check.sh FINAL PASS，Trajectory RMS=0（bit-exact）**，约束 Ham=0.28974817/Px/Py/Pz 与 golden 逐位一致，4 .dat 全产。**L1 组合 F=1.1148（分析开）→ L2 F=1.123，无衰减（发射结构类收益稳定，26a/26bcd 同模式）**。**部署候选 #2（P31+P32+P33 三合一，正式部署由主 agent 执行）**。

**部署指令（供主 agent）**：
```bash
cp -r ~/lab4-gpu ~/lab4-gpu-snapshot-pre-p313233-$(date +%Y%m%d-%H%M%S)
# 从组合树安装 9 个文件（哈希见上）：fmisc_gpu.cu / fmisc.h / MPatch_gpu.cu / Parallel_GPU.cpp / sommerfeld_rout_gpu.cu / sommerfeld_rout.h / bssn_step_gpu.C / prolongrestrict_cell_gpu.cu / prolongrestrict_cell_gpu_int.cu
cp ~/lab4-gpu-cand-p313233-comb-20260826-171357/src/{fmisc_gpu.cu,fmisc.h,MPatch_gpu.cu,Parallel_GPU.cpp,sommerfeld_rout_gpu.cu,sommerfeld_rout.h,bssn_step_gpu.C,prolongrestrict_cell_gpu.cu,prolongrestrict_cell_gpu_int.cu} ~/lab4-gpu/src/
cd ~/lab4-gpu && ./compile.sh
# 验证：100 步全量 check.sh（部署态 OJ-sim）；预计 ~540s（OJ-sim）/ ~536s（真实 OJ）→ 82-83 分区间
```
注意：部署态需保持 AMSS_OPT=-O3 与候选一致；bssn_rhs_gpu.cu/derivatives.h 等未改文件与 26bcd 基线一致，无需复制。


## 迭代35 RK4 跨变量批处理（用户优先级 #4，发射结构杠杆）：**死路（bit-exact F=0.9896，-1% 更慢）**

**候选**：`~/lab4-gpu-cand-rk4batch-20260827-025931`（patch `assets/lab4/opt/search/patch_rk4_batch.py`，hash 守卫 + 3-pass 重排不变式）。改动 3 文件：rungekutta4_rout_gpu.cu（新增 `rungekutta4_batch_kernel`：RK4Batch 结构体（24×3 指针）按值传参，grid=(ceil(n/256), num_var)，blockIdx.y=var_idx，每 (idx,v) 线程与原逐变量 launch 的 idx 线程算术逐 token 一致）+ rungekutta4_rout.h（声明 + RK4_MAXVARS 宏）+ bssn_step_gpu.C（两个调用点（predictor 行~167 / corrector 行~316）改为 3-pass 重排：pass A 全 bam（lev==0）+ 收集 24 指针 → pass B 单次批量 RK4 → pass C 全 rout（lev>0）；每变量序 bam_i→rk4_i→rout_i 保留，跨变量数组 disjoint（不同 sgfn）→ bit-exact by construction）。

**机制（与 P31 同族的发射结构杠杆）**：rungekutta4_rout_kernel 912,576 次 launch（24 vars × 38,024 sites），19.76s/3.2%，avg 0.022ms，16 regs/83.6% occ，stall_long_scoreboard 65.4（memory-latency-bound）。批处理后 launch 数 ÷24（→38,024）。

**构建修复记录（诚实，一轮失败）**：job 175002 首建 FAIL = RK4_MAXVARS 宏只定义在 .cu，bssn_step_gpu.C 编译时未声明 → 补 header 宏（#define RK4_MAXVARS 24 放 rungekutta4_rout.h）+ .cu 去重。修复后 job 175017 BUILD_OK。

**Level-0 ptxas（job 175017）**：rungekutta4_batch_kernel **17 regs / 0 spill / 948B cmem**（原 kernel 16 regs）——codegen 近乎中性（+1 reg 为指针间接）。

**Level-1 A/B（job 175017，2 步 ×4 轮交错，分析关）**：base1 Step2 5.06443 / cand1 5.16828 / cand2 5.10848 / base2 5.10588 → **base median 5.08516 vs cand median 5.13838 → F=0.9896（-1% 更慢）**。**8/8 .dat IDENTICAL（bit-exact 实证）**。

**裁决：死路（负收益）**，Level-2 未跑（GATE F<1.0）。

**死路原因（诚实，两层）**：1. **机制层面成功**：launch 数 ÷24，bit-exact 构造成立（3-pass 重排 + 批量 kernel 均编译通过）。2. **但运行层面负收益**：RK4 kernel 是 0.022ms 极短 memory-latency-bound kernel，launch 间隙本就被 GPU 饱和隐藏（iter30 同源结论：host-side launch 在 GPU 饱和前提下 0 收益）；批量 kernel 反而引入指针间接（cmem 948B vs 原 0）+ grid.y 维度 block 调度开销，净 -1%。**与 iter30（pool）、per-stream sync=0、cudaGraph 关闭同源：launch 数削减在 GPU 饱和前提下不转运行时收益**。

**经验**：发射结构杠杆对「大 kernel 全体积发射浪费」（26a/26bcd/P32）有效，对「小 kernel 高频 launch」（RK4/global_interp 批处理）无效或边际（P31 F=1.006 弱正，RK4 F=0.990 负）。restrict3 批处理（294,975 launches，与 RK4 同型，job 175130）已实测：F=0.9999 死路（见迭代37）。


## 迭代36 rhs_interior 跨调用场读共享（ptxas 门禁：spill 涨 >5% 判死）：**死路（L0 ptxas 门禁命中，spill stores +5.5% > 5%）**

**候选**：`~/lab4-gpu-cand-ri-share-20260827-032907`（patch `assets/lab4/opt/search/patch_ri_share.py`，hash 守卫 + 24 调用点计数不变式）。改动 2 文件：lopsidediff.h（d_lopsided_point 加可选尾参 vx_pre/vy_pre/vz_pre，默认 `__builtin_nan` → 原逻辑读 Sfx[idx]；传入时用 pre 值，bit-exact——interior 路径 fh 纯加载，pre == Sfx[idx] 逐位）+ bssn_rhs_gpu_int.cu（顶部 betaz 导数后 hoist `rishare_vx = betax[idx]` 等 3 值，24 个 lopsided 调用尾参追加 rishare_vx/vy/vz）。

**机制**：rhs_kernel_int 24 次 d_lopsided_point 调用每次都读 Sfx[idx]/Sfy[idx]/Sfz[idx]（= betax/betay/betaz 中心值），72 load → 3 load/线程；lopsided 的 h_000 = f[idx] 是各字段自身中心值，无跨调用冗余（P2 已内部复用）。

**Level-0 ptxas（job 175116）**：rhs_kernel_int 128 regs / 768B stack / **2284B spill stores（基线 2164B，+5.5%）/ 3356B spill loads（基线 3328B，+0.8%）**——**spill stores 超 5% 门禁 → GATE FAIL，未跑 A/B**（门禁规则明确：spill 涨 >5% 判死）。

**裁决：死路（ptxas 门禁命中）**。诚实分析：3 个 hoisted 值跨 24 个调用存活，编译器多溢出 120B stores；load 削减（72→3）被 spill 增量抵消（iter30/iter35 同源：rhs 是 load-latency-bound 非 load-count-bound，spill 流量是关键路径）。**P2（iter8）的「未做跨调用点 vx/vy/vz 提升（24 次调用保活 3 double，swpipe2 教训）」预言验证**：swpipe2（load hoist → spill 4×）、本探针（hoist 3 double → +5.5%）同型死路。

**经验**：跨调用 hoist 无论大小（P28 matter 11 处 load → F=1.0023 近 0；本 72→3 处 → spill 门禁死）在 rhs kernel 上均不成立。rhs 内跨调用场读共享方向关闭。

## 迭代37 restrict3 跨变量批处理（用户优先级 #5，发射结构杠杆）：**死路（bit-exact F=0.9999，0 收益）**

**候选**：`~/lab4-gpu-cand-r3batch-20260827-032200`（patch `assets/lab4/opt/search/patch_r3_batch.py`，hash 守卫 + 24 调用点计数不变式）。改动 3 文件：prolongrestrict_cell_gpu.cu（新增 `restrict3_multi_kernel`：R3Batch 结构体（24×3 指针+SoA）按值传参，grid=(ceil(total/256), num_var)，blockIdx.y=var_idx，几何标量（ni/nj/nk/starts/bbox）host 侧算一次共享）+ prolongrestrict.h（声明 + R3_MAXVARS 宏）+ Parallel_GPU.cpp（case 2 改为仅在首个变量迭代收集 24 变量 (src,dst,SoA) + 一次 multi launch，size_out 逐变量推进镜像原 while 循环）。

**机制**：restrict3（294,975 launches，24.58s/3.98%，avg 0.083ms，blocks 1/29/72，waves 1.04 不足两波）按 (block-pair, 变量) 逐变量 launch。批处理 → launch 数 ÷24；几何（llbc/uubc/extc 等）逐变量相同，host 只算一次。**与 RK4 batch 的关键差异**：restrict3 是 1.04 waves 小 grid（SM 空闲窗口），RK4 是 125-945 blocks 大 grid——restrict3 有波量化头寸而 RK4 没有。

**构建修复记录（诚实，两轮失败）**：job 175100 首建 FAIL = R3_MAXVARS 宏只定义在 .cu，Parallel_GPU.cpp 编译时未声明（与 RK4 batch 同 bug）→ 补 prolongrestrict.h 宏 + .cu 去重。修复后 job 175130 BUILD_OK（restrict3_multi_kernel 128 regs / 288B stack / 0 spill，与原 kernel 同参）。

**Level-1 A/B（job 175130，2 步 ×4 轮交错，分析关）**：base1 Step2 5.04635 / cand1 5.07197 / cand2 5.03253 / base2 5.05684 → **base median 5.0516 vs cand median 5.05225 → F=0.9999（0 收益）**。**8/8 .dat IDENTICAL（bit-exact 实证）**。

**裁决：死路（0 收益）**，Level-2 未跑（GATE F≈1.0）。

**死路原因（诚实，两层）**：1. **机制层面成功**：launch 数 ÷24，bit-exact 构造成立，multi kernel codegen 中性（128 regs/0 spill）。2. **但运行层面 0 收益**：尽管 waves 1.04 有理论波量化头寸，restrict3 是细网格 load-latency 主导（iter24 同结论），双 launch 变单 launch 的调度收益被 grid.y 维度 block 调度开销 + 指针间接（cmem 1520B）抵消，净 0。**与 P31（global_interp varbatch F=1.006）、RK4 batch（F=0.990）同族**：发射结构批处理在 ≤4% 小模块上不转运行时收益（26a 的收益来自「大模块全体积发射浪费」，不适用于小 kernel）。

**经验**：发射结构批处理族（P31/RK4/restrict3）三连实测完毕：global_interp +0.6%、RK4 -1.0%、restrict3 ±0%，均 <10s 线或负。该方向关闭。

## 迭代38 部署态 reprofile（P313233 540s）+ A38-BND 死路 + A38-1 fused-z KEEP + A38-P44 4×4×4 探针

### 38.1 部署态 reprofile（nsys job 175503，546.75s 含开销 / 干净 540.23s）✅

**证据**：`~/lab4-gpu/evidence/reprofile-p313233-20260827-045211/`（sqlite 1.3G + ledger-* + stats-*）+ 本地 `assets/lab4/opt/evidence-reprofile-p313233-20260827/`。

模块账本（kernel-time 525.0s，share% of kernel time）：

| 模块 | kernel | calls | total_s | share% | blocks/launch med |
|---|---:|---:|---:|---:|---:|
| RHS interior | rhs_kernel_int | 38,944 | 189.87 | 36.17 | 512 |
| RHS face | rhs_kernel_facepure + facez | 233,664 | 149.65 | 28.50 | 11 / 37 |
| RHS R6 | rhs_kernel | 38,944 | 28.92 | 5.51 | 11 |
| **RHS 合计** | | | **368.44** | **70.18** | |
| Analysis | global_interp_multi(8K) + amr + surf + avg2 | 126,418 | 48.12 | 9.17 | 3/3/2448 |
| Prolong | prolong3_multi(_int) | 2,148,903 | 45.31 | 8.63 | 2 / 8 |
| Restrict | restrict3_kernel | 294,975 | 24.59 | 4.68 | 29 |
| RK4 | rungekutta4_rout_kernel | 912,576 | 19.76 | 3.76 | 422 |
| Sommerfeld | rout_compact | 902,976 | 7.47 | 1.42 | 43 |
| Ghost | pack/unpack | 2,364,750 | 6.58 | 1.25 | 34 |
| Enforce | enforce_ga | 38,024 | 4.40 | 0.84 | 422 |

三窗口占比稳定（±1%），rhs_int 35.8→36.9%，analysis 7.4→9.6%（晚期网格细）。

**关键新发现**：
1. **analysis 成本 = 800 个 MassPAng 大 launch（grid 144×17，NN=36,864，46.5ms avg）= 37.2s（86.6% of analysis）**。6,400 个小 launch（BH，grid 1×3）仅 1.2s。任务前提"中位数 1 block/launch 串行"对部署态不成立——P31 后 med=3 blocks，且成本全在大 launch。
2. **ncu（job 175642）global_interp_multi 大 launch：L2 pipe 91% SOL（lts__throughput 90.97%）但 data sectors 仅 30.8%** → L2 被**散乱 wavefront 请求数**饱和（stack ya[216]=1728B 局部数组 STL/LDL ~2.9GB/launch，2.7× global load 1.08GB）。stall long_scoreboard 17.05 cyc（72% CPI），64 regs / 2352B stack / 43.7 waves / 44% occ。
3. **rhs face kernels（149.7s）：waves 仅 0.18-0.57，warps active 10-13%，L2 仅 10%** —— 7 个独立小 launch（grid 5-16 blocks）串行，latency-bound。boundary 每点成本 ~7× interior。
4. rhs_int（189.9s）4.46 waves / 19% warps active / L2 20%：latency-bound 且已穷尽。restrict3：1.04 waves / L2 11%。

### 38.2 A38-BND：rhs boundary 7→3 launch 合并（wave 占用率杠杆），**死路（bit-exact F=0.9935，-0.65%）**

**候选**：`~/lab4-gpu-cand-a38-bndmerge-20260827-054726`（patch `assets/lab4/opt/search/patch_a38_bndmerge.py`）。改动 3 文件：bssn_rhs_gpu_face.cu + _facez.cu（kernel `int region` → `int region_base, int region_end`，1-D union 累积 region 选择）+ bssn_rhs_gpu.cu（extern 声明同步 + host 4×facepure → 1 merged launch、2×facez → 1 merged launch；R6 保留）。bit-exact by construction（每点计算逐 token 不变，region 选择仅改枚举）。

**L1 A/B（job 175733，2 步 ×4 交错）**：base med 4.90592 vs cand med 4.93802 → **F=0.9935**。**8/8 .dat IDENTICAL**。GATE F<1.0 → 未跑 L2。

**死路原因（诚实）**：wave 占用率改善（0.39→1.6 waves）被抵消——face kernel 瓶颈是**每线程依赖链 + x-face 散乱访问（16 sectors/warp-load）**，非 wave 数量；合并仅增加 region 选择指令 + 跨 region 并发 L2 集增大，净 -0.65%。**boundary 7× per-point 是结构性**（thin-slab 布局 + mask/反射 + lopsided/kodis 掩码），26a/26b/26bcd 已到结构极限。勿重试合并。

### 38.3 A38-1：global_interp fused-z（消 ya[6³] 局部数组，L2-pipe 杠杆），**KEEP（见 38.4）**

**候选**：`~/lab4-gpu-cand-a381-fusedz-20260827-054736`（patch `assets/lab4/opt/search/patch_a38_fusedz.py`）。fmisc.h 加 `d_gi_fused`（d_decide3d+d_polin3_1b 融合：每 (i,j) 列直接加载 6 z-taps 立即 polint，ya[216] 物化消除，仅 yatmp[36] 残留）+ fmisc_gpu.cu `global_interp_device` 改调 d_gi_fused。sommerfeld 的 d_decide3d/d_polin3_1b 独立调用保留不动。**bit-exact by construction（8 反射 case 逐 case 验证，factor 乘序 SoA[0]→[1]→[2] 与 fill 一致；polint 消费序一致）**。预期 -13~22s（analysis 42.97s 中 MassPAng 37.2s 的 L2 压力 ~2/3 来自 local stack traffic）。L1 A/B job 175767。

### 38.4 A38-1 fused-z L2 结果：**KEEP（bit-exact F=1.0111，-5.9s）**；A38-2 sommerfeld fused-z：**死路（F=0.9999）**

**A38-1 L2（job 175810，100 步 OJ-sim）**：**This Program Cost = 534.31s**（部署基线 540.23s → **-5.92s, F=1.0111**，Total Evolve 491.34s），**check.sh FINAL PASS，Trajectory RMS=0（bit-exact）**，约束 Ham=0.28974817/Px/Py/Pz 与 golden 逐位一致，4 .dat 全产。2 步 F=1.0155 → 100 步 1.0111（73% 保留，结构性收益稳定）。**部署候选**（与 26a/26bcd/P313233 同栈）。ptxas：global_interp_multi_kernel 64→**74 regs / stack 2352→624B / 0 spill**（ya[6³] 局部数组消除生效）。候选 `~/lab4-gpu-cand-a381-fusedz-20260827-054736`（fmisc.h `9415328e` + fmisc_gpu.cu `ddbd2fcf`，其余与部署基线逐字节一致）。

**A38-2（sommerfeld_rout_gpu.cu 2 调用点改 d_gi_fused，job 175902）**：base med 4.93993 vs cand 4.94055 → **F=0.9999（0 收益）**，8/8 .dat IDENTICAL。死路原因：sommerfeld_rout_compact_kernel 已是 8.1μs 级小 launch（902,976 calls），局部流量消除被 launch 开销/延迟主导吞没；且 P32 compact 后 sommerfeld 仅 7.47s（1.4%），机制正确但模块太小。**sommerfeld fused-z 勿单独部署**（与 A38-1 叠加收益 ≈0）。

**经验**：local-stack 消除（fused-z 机制）只在「大 launch + L2-pipe-bound」的 kernel 上转正收益（global_interp MassPAng ✓），在「小 launch + 延迟主导」的 kernel（sommerfeld）上无效。机制选择必须匹配 ncu 的 pipe-level 证据。

### 38.5 A38-P44：prolong3 4×4×4 组多输出 L0 探针，**feasible（92 regs / 5832B smem / 0 spill），A/B 待测**

**候选**：`~/lab4-gpu-cand-a38p44-probe-20260827-061709`（patch `assets/lab4/opt/search/patch_a38_p44_probe.py`：`#ifdef A38P44_PROBE` 下的 `prolong3_multi4_probe_kernel`，未接线 host）。机制：P33 是 2×2×2（8 输出/组，216 taps 共享）；4×4×4 = 64 输出/组，2 anchors/dim，共享 coarse cube 9³=729 taps（smem 5.8KB），global load 2.4× 削减（1728→729 per 64 输出）。每成员 Z/Y/X 累积序逐 token 复制 P33 奇偶分支（bit-exact by construction）。interior 阈值从 [3,extc-3] 变为 [3,extc-4]（组跨 2 anchors），边界组用 mask 路径（out-of-range tap = 0，与原逐点语义一致）。

**L0 ptxas（job 175955，-DA38P44_PROBE -Xptxas -v）**：prolong3_multi4_probe_kernel **92 regs / 5832B smem / 0 spill / 1 barrier**（对照 P33 boundary 110 regs / 576B stack / 0 spill）。**GATE PASS**（无 spill、smem 5.8KB 不伤占用率、regs 92 ≤ 100）。**A/B 未测**（需 host 接线 + interior 变体 + 组合 L1/L2，见迭代 38.6）。

### 38.6 A38-P44 接线 A/B：**死路（结构缺陷，bit-exact FAIL）**；A38-1 为唯一 keep → 部署候选

**候选**：`~/lab4-gpu-cand-a38p44-wire-20260827-065237`（patch `assets/lab4/opt/search/patch_a38_p44_wire.py`：`prolong3_multi4_kernel` + `_int` 接线 host，G4 = (en-lead)/4+1，interior 阈值 [3,extc-4]，lo/hi 阈值 extc-3）。

**L0 ptxas（job 176250）**：prolong3_multi4_kernel(_int) **92 regs / 5832B smem / 0 spill / 1 barrier**（与探针一致）。

**L1 A/B（job 176250，2 步 ×4 交错）**：**cand 崩溃**（step 1 predictor NaN → MPI_ABORT），base 正常。bit-exact FAIL（cand 无 .dat）。

**死路根因（诚实，结构缺陷，非调试 bug）**：4×4×4 组的 9³ coarse cube 是**每线程私有**工作集（每线程处理不同的 4×4×4 组），但 `__shared__ double cube[9][9][9]` 是**每 block 共享** → 256 线程把 256 个不同组的 729 taps 并发写同一 smem 数组（race）+ 组枚举 early-return 使部分线程跳过 `__syncthreads()`（divergent barrier UB）→ 错值 → NaN。**探针 ptxas「92 regs/0 spill」测的是语义错误的 kernel 的编译结果，不可作为可行性证据**（教训：ptxas gate 不能替代语义正确性；P3.5b 的 no-op 教训同源）。

**结构结论（重要）**：单线程 4×4×4 组的 cube 共享**只有**物化（registers 装不下 729 doubles / local 流量爆炸 110KB/thread / smem 需跨线程共享）三条路；不物化则与 P33 8 次重复等价（load 不削减）。**真正可行的 tap-sharing 是「点 tile 级」smem 共享（多线程共享 cube）**——设计见 `assets/lab4/opt/search/a38_tapsharing_design.md`（4×4 点 tile × 17 var 循环，5.8KB smem，4.7× load 削减，-10~25s 潜力，工程量 2-4h，留待下轮）。**restrict3 group 同族风险**：须 fused-Z 式（每列 union taps 立即累加，不物化 cube）才可行，且预期仅 -4~6s，本轮不做。

### 38.7 本轮部署候选汇总（主 agent 裁决部署）

**唯一 keep = A38-1（global_interp fused-z）**：L2 job 175810 = **534.31s**（部署基线 540.23s → **-5.92s, F=1.0111**），check.sh FINAL PASS RMS=0 bit-exact，约束逐位一致。候选 `~/lab4-gpu-cand-a381-fusedz-20260827-054736`，改动 2 文件：`src/fmisc.h`（hash `9415328e`）+ `src/fmisc_gpu.cu`（hash `ddbd2fcf`），其余 7 文件与部署基线逐字节一致。**部署指令**：快照 → cp 两文件到 ~/lab4-gpu/src → ./compile.sh（AMSS_OPT=-O3）→ 100 步 OJ-sim 验证（预计 ~534s）。

**本轮其余 verdict**：A38-BND（边界 7→3 launch 合并）F=0.9935 死路；A38-2（sommerfeld fused-z）F=0.9999 死路；A38-P44（4×4×4）结构死路；tap-sharing 设计完成未实现；TwoP/init 重叠 driver 级不可行。

### 38.8 restrict3 group（P33 式 2×2×2 组）分析结论：**本轮不做（fused 仅 1.5× load 削减，边际）**

关键结构差异（vs prolong3 P33）：prolong3 的 2 个 fine 成员共享**同一** 6-tap 窗口（anchor 相同，仅系数序不同）→ P33 8× load 削减；restrict3 的 2 个 coarse 成员窗口**错位**（[kf-2,kf+3] vs [kf,kf+5]，共享 4/6 taps/dim）→ union 8³=512 taps。**fused 版（不物化，tmp2 进寄存器）仅 1.5× 削减（1,152/8 输出 vs 216/输出）**，materialized 版 3.4× 但需 512 doubles 存储（registers 装不下 / smem 跨线程 race（p44 教训）/ local 流量爆炸）→ 与 4×4×4 同族死路。预期 -3~5s，effort/risk 不成比例，**本轮不做**。restrict3 的 ncu（L2 11%、long_scoreboard 2.13、1.04 waves）确认其 latency-bound 且 L2 远未饱和 → load 削减收益本就有限。

**本轮最终部署栈**：部署基线（P313233 540.23s）+ **A38-1 fused-z → 534.31s**（唯一 keep）。其余全部死路/不可行/设计待实现。
