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

## GitHub 记录 + 缓存清理（2026-08-27 06:45，用户指令）✅

- 远程 `~/lab4-gpu`（P313233 部署态，86 src）已镜像到仓库 `lab4-gpu/`（102 文件，哈希一致：fmisc_gpu d8684f83 / prolongrestrict 68a65584 / bssn_rhs 9baee005）并推送 GitHub（commit 10005e7，origin main）。
- `.gitignore` 已补齐：.env（含真实 API key）/ .pi / .agents / .codex-zjusct / labs/assets 符号链接 / tmp / .remote_work。labs/assets 是 Typst 构建符号链接，因 core.symlinks=false 会被 git 当作目录展开，已从缓存区移除并 gitignore。
- 缓存清理：`tmp/`（27M）+ `.remote_work/`（72M）已删。24 个文档引用的 tmp 快照文件已保留到 `assets/lab4/opt/snapshots/tmp/`（561K，含 bssn_rhs_gpu_deployed.cu 等，ABEGPU.md 引用仍有效）。
- 推送教训：SSH 端口 22 对 ~140MB pack 不稳定（Connection reset），改用 ssh://ssh.github.com:443（git config 自动 fallback）推送成功。

## 第 4 轮子代理（run 1155f44d → 5e654754，04:50-07:00）✅ 完成

- 任务结果：1) 部署态 reprofile（job 175503）：RHS 70.2%（int 190 + face 150 + R6 29）、analysis 48.1s（MassPAng 800×46.5ms=37.2s 占 86%）、prolong 45.3s。2) **A38-1 fused-z KEEP**（L2 534.31s，F=1.0111，bit-exact FINAL PASS，stack 2352→624B；2 文件：fmisc.h 9415328e + fmisc_gpu.cu ddbd2fcf）。3) A38-BND 死路（F=0.9935）、A38-2 sommerfeld 死路（F=0.9999）、A38-P44 4×4×4 死路（smem race + divergent barrier，结构缺陷）。4) tap-sharing 设计已存 `assets/lab4/opt/search/a38_tapsharing_design.md`（-10~25s 潜力，留待下轮）。5) TwoP/init 重叠不可行（driver 依赖）。
- 回写：search-memory 迭代 38.1-38.8。
- 本轮后再无已知正收益单变量杠杆（register/spill/live-set/发射/缓存路径全族已闭环）。距 461s/90 分差 ~73s；残余：tap-sharing -10~25s + 算法级授权。

## 第 4 轮部署（A381，✅ 完成 08:35）

- **A381 fused-z 已部署并验证**：job 176888 verify2，**OJ-sim 535.30s**（P313233 540.23 → -4.9s），check FINAL PASS RMS=0 bit-exact，约束逐位一致。证据 `~/a381-verify2-20260827-081220/`。
- **坑 1**：subagent A/B 后 formal 的 AMSS_NCKU_Input.py 残留 Final_Evolution_Time=2.0，首次 deploy 只跑 2 步 check FAIL（非代码问题）；已恢复 100.0。
- **坑 2**：deploy 脚本末尾清理 build 导致 verify 无二进制；verify2 改为完整重建+run+check。
- **坑 3**：lab4g10 单作业配额（a100 maxJobs=1），并行会话占位需等待（176641 Timeout 后重提）。
- 部署后正式态：fmisc.h `9415328e` / fmisc_gpu.cu `ddbd2fcf` / bssn_rhs `9baee005`。提交包 9 项已清理。
- 当前基线轨迹：605.78 → 540.23（P313233）→ **535.30s（A381）**。预测 ~85.1 分；90 分需 ≤461s，仍差 ~74s。

## 第 5 轮：算法级重构（2026-08-27 10:05，用户授权）

- 用户决策：**授权 RHS 算法级重构**（z 向滚动窗口 / 数据流重排 / 选择性重算）。
- 验收放宽：check.sh FINAL PASS（RMS≤1e-3、约束≤2）即可，**bit-exact 非必须**（浮点重排可接受）。
- 目标：OJ-sim ≤461s（90 分），工程预算 ≤450s。当前 536.8s，差 -76s（RHS 369s 需 ~1.2-1.3× 或组合）。
- 关键设计背景：plan-340s-sprint.md Phase 2（R1-R4 候选）、milestone-B 分析（66-double live floor，但包含 17 场值——若移 smem 可降至 ~49 doubles）；interior 特化后内核已瘦（30K vs 124K 静态指令），**split 族值得在瘦内核上重测**（旧测试是 pre-interior）。
- 红线：不改物理/网格/演化时间；不改 runner/timing/评测；不硬编码输出；check.sh FINAL 是硬门。

## OJ 实测回填（2026-08-27 10:00）✅ 真实分数确认

- **真实 OJ：GPU 85/120 · 536.760s**（scoreBeforeRounding=85.20075），trajectoryRMS=0（bit-exact），约束全 ≤2，100/100 time groups。sourceRevision `5b0edd5-r11`。
- **模型校准**：预测 84.87 vs 实际 85.20，残差 +0.33（模型精确，±1.7 残差内）。
- **校准后模型**：score = 297.95 − 33.90·ln(T)。90 分需 T≈461s（差 -76s）。
- **OJ 轨迹**：1604/0 → 1228/55 → 1044/64 → 706/76 → 601.4/81 → **536.8/85.2（当前）**。
- 部署栈全程 bit-exact：26bcd → P313233 → A381 fused-z。

## 90 分可达性（校准后终审）

- 真实曲线确认 log 模型精确（85.2 实测）。90 分需 461s，当前 536.8s，差 -76s（-14%）。
- 残余杠杆：tap-sharing -10~25s（→ ~510s/87 分，仍不足 90）；无其他已知正收益单变量杠杆。
- RHS 70%（369s）occupancy 被 66-double live floor 数学钉死 25%（milestone-B），算法级需授权。
- **结论：>90 分在当前约束（不改物理/位级安全/不改 runner）内不可达；tap-sharing 只能到 ~87。需用户决策：算法级重构授权 / 接受 85 / 继续 partial。**

## OJ 提交准备（2026-08-27 09:45，用户决定：先提交 OJ 实测）

- 提交包 `~/lab4-gpu` 已验证就绪：9 项 / 102 文件 / 86 src，无 build/cache/evidence/__pycache__。
- OJ 配置核对：MPI=1、OMP=8、GPU=yes、Final=100、Analysis=0.1、Dissipation=0.15 ✓。
- sha256 清单：`~/lab4-gpu-submission-a381.sha256`（102 行），本地副本 `assets/lab4/opt/evidence-reprofile-20260826/lab4-gpu-submission-a381.sha256`。
- 部署态实测：A381 OJ-sim 535.30s（check FINAL PASS RMS=0）。预测 ~85 分；真实 OJ 待用户上传后回填。
- **待用户上传 OJ → 回填真实 scoreBeforeRounding → 校准评分模型 → 决定是否需继续 tap-sharing/算法级。**

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

## 第 5 轮子代理执行结果（2026-08-27 11:30，run 5）✅ 完成

- **RHS 算法级重构 L0 全族死路**（证据 `~/lab4-gpu-cand-r5-zroll-20260827-110726/evidence/r5-l0/`）：
  - **#1 z-rolling（R3 主候选）**：17 场值 smem 暂存（34.8KB + barrier）实测 natural regs 仍 255（hw cap）、lb2 spill 2124→2036B（仅 -4.1%）。milestone-B "49-double floor" 假说证伪：场值在 Ricci 步骤必须存活，smem 只是换 load source 不缩短 live range。**GATE FAIL**。
  - **#3 Ricci 消融诊断**：整个 Step-4 怪物删除后 natural 仅 255→254、lb2 spill -17.5% → **峰值在 fdderivs 61-fh 段**（iter9 P3 结论在 thin kernel 复现），Lever B recompute 上限即此且 spill 不转 runtime → 死路。
  - **#2 split / #4 DAG**：分析判死（任何含 Ricci 的 kernel 持有 36 几何层 + 组装瞬时 ≈105 doubles 不可压；消融证据 254 regs）。
  - **结论：90 分（≤461s）在物理/诚信边界内不可达；85.2 分为实际极限**。部署态维持 A381（536.760s/85.20 分）。
  - **fallback tap-sharing（-10~25s → ~87 分）**：设计已存 `a38_tapsharing_design.md`，本轮未实现（A38-1 fused-z 后 analysis 新瓶颈未重测 ncu，前提待验证；建议主 agent 决定是否续做）。

## 第 5 轮最终结论（2026-08-27 11:45）✅ 全部 L0 证据闭环

- **RHS 算法级重构 L0 判定**（`~/lab4-gpu-cand-r5-zroll-20260827-110726/evidence/r5-l0/`，11 个 ptxas log + job1-5.log）：
  | 候选 | L0 结果 | 判定 |
  |---|---|---|
  | #1 z-rolling（R3） | natural 255 不变、lb2 spill 2124→2036B（-4.1%）、+34.8KB smem | **死路**（milestone-B 49-double floor 假说证伪） |
  | #2 2-way/3-way split | 分析判死（任何 Ricci kernel ≥105 doubles；旧 split v2/Lever A 同构） | 死路 |
  | #3 Ricci recompute | 消融诊断：whole monster 删除后 natural 仅 255→254、lb2 spill -17.5% 上限 → 峰值在 fdderivs 61-fh | 死路 |
  | #4 DAG 重排 | 与 #1/#3 同构（live set 固有 254 regs） | 死路 |
  | #5 tap-sharing fallback | **前提已验证成立**（ncu：L2 82.43% SOL，job 178262） | 待实现（-10~25s → ~87 分） |
- **90 分可达性定论**：90 分（≤461s）需 -76s，RHS 369s 的算法级重构全族 L0 死路（register floor 254-255 regs 是 BSSN 公式固有）。tap-sharing -10~25s 只能到 ~87 分。**85.2 分（A381，536.760s）为当前实际极限**。
- **部署状态**：formal 未动（8 文件 hash 与 A381 一致），Input 已还原（Final=100/Analysis=0.1），无部署候选。
