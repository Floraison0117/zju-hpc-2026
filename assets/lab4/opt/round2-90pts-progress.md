# Round 2 冲刺记录：目标 OJ >90 分（2026-08-26 启动）

## 目标与模型

- 基准：部署态 26bcd 栈，真实 OJ **601.364s / 81.074 分**（2026-08-26 提交）。
- 评分模型（5 精确 OJ 点 + 340s→100 锚点，log 拟合）：`score ≈ 297.95 − 33.90·ln(T)`。
  - 90 分 ≈ 461s；91 分 ≈ 448s；92 分 ≈ 435s。
- 工程目标：**OJ-sim Program Cost ≤450s**（节点噪声 ±5%，OJ≈sim−0.7%）。需再砍 ~156s。

## 已测杠杆（本轮 P31-P37，2026-08-27 03:40 更新，全部远程证据确认）

- **P313233 组合（P31+P32+P33）**：**L1 GATE PASS（bit-exact F=1.1148）→ L2 PASS（job 174954）**：**Program Cost 539.48s**（基线 605.78s，**-66.3s，F=1.123**），check.sh FINAL PASS，Trajectory RMS=0（bit-exact），约束逐位一致，Total Evolve 495.73s。候选 `~/lab4-gpu-cand-p313233-comb-20260826-171357`。**部署候选 #2（三合一）**。
- **P33 prolong3 多输出重写（prolonggrp）**：L2 PASS — **Program Cost 579.66s**（基线 605.78s，-26.1s，F=1.045），check.sh FINAL PASS RMS=0。L1 F=1.0677。候选 `~/lab4-gpu-cand-p33-prolonggrp-20260826-165912`。**部署候选 #1**（已并入 #2）。
- **P32 sommerfeld compact（sommcompact）**：L1 F=1.0279，8/8 IDENTICAL，L2 随组合通过。候选 `~/lab4-gpu-cand-p32-sommcompact-20260826-164623`。
- **P31 global_interp varbatch（givb）**：L1 F=1.0061（弱），L2 随组合通过。候选 `~/lab4-gpu-cand-p31-givb-20260826-163501`。
- **RK4 跨变量批处理（rk4batch）**：**死路** — L1 F=0.9896（-1%），8/8 IDENTICAL，Level-2 未跑（GATE F<1.0）。job 175017。
- **rhs_interior 跨调用场读共享（ri-share）**：**死路** — L0 ptxas 门禁命中（spill stores +5.5% > 5%），未跑 A/B。job 175116。
- **restrict3 跨变量批处理（r3batch）**：**死路** — L1 F=0.9999（0 收益），8/8 IDENTICAL，Level-2 未跑。job 175130。

## 子代理运行史

- Run1 `617a316f`（16:22-17:08）：诊断 + P31/P32/P33 实施，P33 L1 GATE PASSED；因无超时 SSH 挂起被 interrupt。
- Run2 `c2fc0b59`（17:10 续）：P33 L2 跑完（579.66s ✓）后卡死 ~8.5h（steer 重递送循环），interrupt。
- Run3 `4996d3bb`（02:05 续）：干净新会话，任务 = 组合 L1→L2 + 剩余杠杆（RK4 批处理、rhs 场读共享、restrict3）+ 回写 search-memory。
- **教训：subagent 长时间无输出（>30min）必须检查；steer 重递送循环会饿死 agent 动作。**


## 基线验证（本轮主 agent 已做）

- formal 哈希核对 ✓：bssn_rhs_gpu.cu `9baee005`、derivatives.h `83da2088`、prolongrestrict_cell_gpu.cu `44b8dc55`。
- SSH/`hpc submit -p lab4g10` 通道 ✓。
- compile.sh 默认：TWOP_GPU=OFF、OMP_TUNE=ON、PACKED_RELAX=ON、COS_TABLE=ON、AMSS_OPT=-Ofast ✓。
- Input：MPI=1、OMP=8、GPU=yes、Final=100、Analysis=0.1、Dissipation=0.15 ✓。

## OJ 提交机制（已核实）

- 提交包 = `~/lab4-gpu` 清理为 9 项（src/、CMakeLists.txt、compile.sh、run.sh、AMSS_NCKU_Input.py、AMSS_NCKU_Program.py、check.sh、golden/、scripts/）+ sha256 清单；无 build/日志/cache/备份。
- 平台上传为用户操作（仓库无 OJ 网络通道工具记录）。主 agent 负责包整理 + 验收，提交需用户执行或明确授权的通道。

## 部署模板（已核实）

- 远程 `~/reprofile-tools/p26bcd_deploy.sh`：hash 守卫 → 快照 → cp 候选文件 → compile.sh 重建（AMSS_BUILD_DIR=$BASE/build，cuda-13.3）→ 100 步 OJ-sim（TwoP live，清理 GW250118/Ansorg.psid/twopuncture_cache）→ check.sh → 清理提交包 9 项。
- 关键 env：OMP_NUM_THREADS=8、AMSS_ENABLE_TWOP_GPU=OFF、OMP_TUNE/PACKED_RELAX/COS_TABLE=ON、AMSS_CUDA_ARCHITECTURES=80、AMSS_OPT=-O3（注意：部署用 -O3，与候选一致；OJ 构建用 CMake 默认）。
- 部署前 hash 守卫用 9baee005（当前 bssn_rhs_gpu.cu）。

## 主 agent 部署执行（2026-08-27 03:40-04:50）✅ 完成

- **P313233 组合已部署并验证**：deploy 尝试 1（175173）因脚本清理步骤删除证据而不可见结果；deploy2（175289）build OK 但同样清理丢失日志；最终 verify job **175395** 证据完整：**OJ-sim 540.23s**（候选 L2 539.48s，误差 0.14%），check FINAL PASS RMS=0，约束逐位一致。证据 `~/p313233-verify-20260827-042840/`（job/build/run/check log）。
- **部署教训**：deploy 脚本最后一步 `rm -rf ... evidence ...` 会把自身日志删掉（stdout 重定向到 EV/job.log，NFS 上 rm 后剩空目录）。验证脚本必须把日志写到 formal 树外（如 ~/）。
- formal 已清理为 9 项提交包：86 src 文件，9 个改动文件哈希匹配候选 ✓，未改动文件仍 26bcd 基线 ✓。
- **当前状态：部署态 OJ-sim 540.23s → 预测 ~85 分。90 分需 ≤461s，差 ~80s。**

## 第 4 轮子代理（run 1155f44d，02:50 启动）

- 任务：1) 部署态 reprofile 校准 2) analysis 深挖（global_interp 单 block 结构）3) prolong 4×4×4 评估 4) TwoP/init 重叠可行性。
- 红线：formal 只读、不部署不提交 OJ、不碰 lab4.typ、三级门禁、回写 search-memory 迭代 38+。

## 主 agent 待办（子代理返回后）

1. 审核候选证据（L0/L1/L2 + F + check.sh FINAL PASS）。
2. 部署 P313233 组合（哈希见 search-memory 迭代34，8 文件 + 快照 + 100 步 OJ-sim + check.sh）。
3. 预计部署态 OJ-sim ~539s → 真实 OJ ~536s → **~85 分**（评分模型 297.95−33.90·ln(539.5)=85.0）。**90 分（461s）不可达**：本会话已穷尽剩余杠杆（RK4/rhs共享/restrict3 全死路），距 90 分缺 ~78s，无已知方向可补。决策点：接受 85 分部署 or 授权算法级重构。
4. 部署后验证：100 步 OJ-sim + check.sh FINAL PASS + 提交包 9 项 + sha256 清单。

## 规则红线（本轮不变）

- 不碰 labs/lab4.typ；不改 runner/timing/评测；不硬编码输出；formal 只读；主 agent 独占部署与 OJ 提交。
