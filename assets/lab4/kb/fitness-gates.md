# Fitness 函数与验证门 (Lab4)

> KernelEvolve 元组 `(F, π_sel, O, τ)` 中的 F 与 τ。对应 KernelPro 的 log-reward calibration + correctness as hard constraint。

## 适应度 F（fitness）

```
F(v) = t_ref / t_candidate
```

- `t_ref`：baseline（已部署栈）的每步演化时间（演化-only，Analysis_Time=1000 跳过每步分析，或 OJ-equiv Analysis_Time=0.1）。
- `t_candidate`：候选同配置同节点同作业的每步时间。
- **正确性是硬约束**：check.sh 非 FINAL PASS → `F(v) = 0`（不论多快）。
- F > 1.0 才算正收益；F ≤ 1.0 标注死路（回写 search-memory）。

## 正确性门（check.sh FINAL PASS 是硬约束；bit-exact 非要求）

check.sh（`scripts/check_result.py`）真实判据（硬编码常量，非自设）：
- `RMS_LIMIT = 1.0e-3`：轨迹 RMS ≤ 0.1% → PASS（**不是要求 RMS=0**；已部署版本恰好位级一致才报 0）。
- `CONSTRAINT_LIMIT = 2.0`：约束 maxima（Ham/Px/Py/Pz）均 ≤ 2.0 → PASS。
- `TIME_TOLERANCE = 1.0e-8`：轨迹时间列匹配容差。
- `FINAL: PASS` iff RMS ≤ 1e-3 AND 所有约束 maxima ≤ 2.0。

**bit-exact 非要求**：浮点重排（不同归约顺序、跨 rank 分布式计算改求和序等）可接受，只要全量 check.sh PASS。AGENTS.md 仅要求“不魔改 runner/评测用例刷分、不硬编码输出、不按输入分支”，未要求位级一致。

辅助检查（非硬约束，用于诊断）：
- 输出 `.dat`（bssn_BH/psi4/ADMQs/constraint）去首行时间戳后 `tail -n +2 <file> | sha256sum` 比对 baseline：若一致则位级安全（RMS 必为 0）；若不一致需进一步用 check.sh 判（可能仍在 1e-3 内）。
- 物理正确性：TwoPuncture Mp=0.598837、Mm=0.401163、ADM=0.983557，Newton 收敛 |F|<5e-12。

**累积警惕（重要）**：浮点漂移随步数累积。§16.18 教训：level-0 RHS 复制 5 步 max|d|=1.485e-04（远 < 1e-3，短跑 PASS），但 40 步累积到约束 Ham=8.15 >> 2.0（FAIL）。故**短跑 PASS ≠ 全量 PASS**，每个候选仍须 Level 2 全量（40/100 步）验收。

## 三级验证（KernelPro 闭环 + README §3）

### Level 0：静态检查（秒级，不占队列）

- `sha256sum` 候选 vs formal（确认基线未漂移）。
- `gfortran -cpp -E` 候选(OFF) vs formal，去行指令后 diff（位级改动是否意外改变代码）。
- `gfortran -fopt-info-vec-all -c <file>.f90`（向量化确认）。
- `bash -n` / `python -c "ast.parse"`（语法/配置）。
- 配置防漂移清单：`Dissipation=0.15`、`Final_Evolution_Time=40.0(CPU)/100.0(GPU)`、`Analysis_Time=1000.0`、`MPI=30(CPU)/1(GPU)`、`OMP=1(CPU)/8(GPU)`、`GPU_Calculation=no(CPU)/yes(GPU)`。
- 脚本：`assets/lab4/opt/fitness/level0_static.sh`

### Level 1：短跑 A/B（1-3 min，占队列但快）

- **OFF 与 ON 放同一作业**跑（同节点抵消漂移），缓存 TwoPuncture（`--twop-cache`）。
- CPU：5 步（Final=5, Analysis=0.1 模拟 OJ），grep 每步 `Computer used` + 输出 sha256 对比（位级一致则 RMS 必为 0；不一致需 Level 2 判）。
- GPU：2 步短跑测每步时间。
- **注意**：短跑加速可能是假象（GPU 512 线程静默 launch 失败），只证"相对变化与位级一致"，不证"绝对正确"。
- 脚本：`assets/lab4/opt/fitness/level1_short_ab.sh`

### Level 2：全量验收（8-25 min，唯一权威门）

- CPU：40 步（Final=40, Analysis=1000 或 0.1）+ check.sh FINAL PASS。
- GPU：100 步（Final=100, Analysis=0.1）+ check.sh（30 min 墙内，一作业仅 1-2 个全量）。
- 仅用于：(a) 部署确认；(b) OJ 提交前；(c) 短跑异常定位。
- 脚本：`assets/lab4/opt/fitness/level2_full_check.sh`

## 终止规则 τ（termination）

- 墙钟耗尽（GPU 30 min/作业，CPU 30 min/作业）。
- 连续 N 步无 F 改善（stall，建议 N=3）。
- 达标（F 使 OJ 预估达目标分，或 check.sh PASS 后用户确认部署）。

## 选择策略 π_sel

- 默认 **greedy**（选 F 最高的已验证节点扩展），墙钟充裕可上 MCTS（UCT，progressive widening，log-reward）。
- 每节点 = 一个完整编译+剖析+验证过的候选；边 = 单一变量变换（一次只改一个变量，README §2）。

## 作业批处理（摊薄排队）

- 一个作业做多件事：多构建 + 多运行 + 对比 + check 全塞进一个 `hpc submit`。
- 提交参数：CPU `-c 60 -t 30m`；GPU 按分区 `hpc submit -p lab4g10`。
- 输出全部 `tee` 到 `evidence/<实验名>-<ts>/summary.txt`。

## 部署/OJ 红线（AGENTS.md）

- formal 源只读；候选 `cp -r` 隔离；改动用 patch 脚本记录源哈希。
- 提交前：87/88 项 `sha256sum -c`，无 build/日志/cache/备份文件。
- **不改 runner/timing/评测用例刷分；不硬编码输出；不按输入分支**（诚信边界）。
- 候选 agent 只在隔离副本工作；正式部署/OJ 提交由主 agent 执行。
