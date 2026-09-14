# M-10 EAWM-inspired auxiliary GPPO-History：本机 CUDA 有界矩阵

## 结论

本轮完成 EAWM-inspired auxiliary GPPO-History 的 pilot、三 seed 正式训练和固定 final-test。E 变体在训练期间确实将事件辅助损失回传到共享 History 编码器；它不是旧候选后果模型，辅助头也没有进入在线推理动作分布。

在新到达协议、弱通信、固定 64 个 final-test 父 tape 上，E 为 551/1152、H 为 544/1152 个按时物理到达任务，差值 +7（+0.608 个百分点）。按 seed 差值为 −2、+4、+5，只有 2/3 seed 改善；主机按时确认为 E 88/1152、H 95/1152，下降 7 个任务。64 个共享父 tape 的配对 bootstrap 95% 区间为 −1.215 到 +2.431 个百分点，且不覆盖训练 seed 不确定性。因此未建立稳定任务收益，不进入下一轮研究决策。

E 完整决策链均值为 4.077 ms，H 为 3.951 ms（final-test 加权样本）；E 的 P95/P99（seed 1101/2203/3307）为 6.572/7.934、6.075/6.987、6.085/7.006 ms，H 为 5.436/6.871、6.024/7.003、6.279/7.034 ms。计时包含公开张量构造、策略前向、动作 mask 和选择，排除环境推进与落盘；CUDA 计时已同步，0 步预热。这是本机 RTX 3060 Laptop GPU 仿真测量，不外推 GPU 实时性。

## 冻结配置与证据范围

- 到达协议：`arrival_to_region`，deadline basis=`physical_arrival`；host confirmation 单独记录。
- 弱通信：composite；奖励、动作 mask、NOOP、ACK、lease、fencing、能量和安全门禁不变。
- H：原 GPPO-History；E：相同共享 History + `L_total=L_PPO+0.1*L_event`。没有旧候选 prior、旧世界模型、通知机制或事件触发决策。
- tapes：train 32、validation 16、final-test 64；每场景 6 个任务；冻结清单 SHA-256=`52a6fda6baf91ec9928272c82490e381f4c11740e14c8eb9cfaeafb77fe8405f`。final-test 在训练完成和模型冻结后才读取。
- final checkpoint：固定预算末尾 `last.pt`，不是按 validation 最高点挑选；validation 仅记录曲线。

## Pilot 与正式训练

| 运行 | 环境步 | optimizer.step | 停止原因 | 耗时(s) |
|---|---:|---:|---|---:|
| H / 1101 pilot | 512 | 8 | environment_step_budget | 20.031 |
| E / 1101 pilot | 512 | 8 | environment_step_budget | 29.262 |
| H / 1101 | 8192 | 128 | environment_step_budget | 364.357 |
| E / 1101 | 8192 | 128 | environment_step_budget | 338.289 |
| H / 2203 | 8192 | 128 | environment_step_budget | 303.066 |
| E / 2203 | 8192 | 128 | environment_step_budget | 669.299 |
| H / 3307 | 8192 | 128 | environment_step_budget | 503.164 |
| E / 3307 | 8192 | 128 | environment_step_budget | 602.179 |

正式运行共 49152 环境步、768 次真实更新；含 pilot 的训练运行记录约 2829.648 s，另有 final-test 评估过程，没有追加 seed、步数或时间预算。Pilot checkpoint roundtrip 对 H/E 均通过（512 步、8 次更新，检查期间 0 次 optimizer.step）。为打通恢复和最终评估保留了运行器缺陷修复及失败输出；修复涉及训练模式切换、CPU/CUDA RNG 恢复、显式 resume、最终评估参数和规则目录复用，没有修改研究变量。

## Final-test 原始比较

| seed | H 物理/任务 | E 物理/任务 | E−H 物理 | H 主机/任务 | E 主机/任务 | H 回报 | E 回报 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1101 | 190/384 | 188/384 | −2 | 27/384 | 26/384 | 1030.511 | 1002.297 |
| 2203 | 184/384 | 188/384 | +4 | 34/384 | 33/384 | 944.395 | 1001.516 |
| 3307 | 170/384 | 175/384 | +5 | 34/384 | 29/384 | 749.938 | 825.496 |
| 合计 | 544/1152 | 551/1152 | +7 | 95/1152 | 88/1152 | 2724.844 | 2829.310 |

同一 final-test 的共享规则基线为 189/384 物理按时到达、34/384 主机按时确认，不能复制成三个训练 seed。规则点估计 49.219%，H 47.222%，E 47.830%，所以 E 未超过规则。H/E 各 seed 安全违规均为 0、未决任务均为 0；这只代表本轮已记录检查范围，不是生产安全保证。

父 tape 统计以 64 个共享 final-test tape 为配对单位，将三个训练 seed 在每个 tape 的完成率先取平均，再做 20000 次 percentile bootstrap：估计 E−H=+0.608 个百分点，95% CI=[−1.215,+2.431] 个百分点。该区间只描述 tape 不确定性；三个 seed 差异另列，不能当作训练随机性结论。

## 辅助任务诊断

这是对三份 E 正式训练 rollout ledger 的只读回放：每份 8192 条记录，共 24576 条；参数更新 0 次；当前/下一公开观测回放 mismatch 均为 0。它不是 validation/test 结果，也不是独立泛化证据。

| 字段族 | 有效支持 | accuracy | 多数类 baseline | 差值 |
|---|---:|---:|---:|---:|
| position_x | 53297 | 0.9163 | 0.9163 | 0.0000 |
| position_y | 54487 | 0.9217 | 0.9217 | 0.0000 |
| energy | 54434 | 0.5836 | 0.5648 | +0.0188 |
| task_state | 63767 | 0.9402 | 0.9402 | 0.0000 |

只有 energy 在训练账本回放上高于多数类 accuracy；其余字段族未超过多数类。energy 的第三类支持为 0，不能将它当作三类任务全部有效。该诊断不证明未见数据或调度有效，也不把总体 accuracy 当作类别均衡性能。

## 成本、调用与账本

E 不调用旧候选后果模型，在线辅助头在推理评估中关闭；本轮 `world_model_calls=0`。训练账本为 gzip 压缩逐步记录，包含 run/seed/episode/step、动作、old log-prob、value、reward、终止/截断与 bootstrap、公开字段及标签 mask。推理账本保存 actor/决策、通信代理和安全字段；通信代理字节是序列化仿真代理量，不是真实网络流量。

训练和评估分别保存 `best-inference.pt` 与含 optimizer/RNG/数据顺序/累计预算的 `last.pt`，以及 `updates.jsonl`、validation curve、checkpoint roundtrip 和 run status。辅助损失实际参与 E 的训练反向传播，没有同时更新独立世界模型。

## 回归验证

在专用工作树 basetemp 下，全套 pytest 为 **194 passed**。第一次使用系统临时目录时有 8 项因 Windows `WinError 5` 无法创建 fixture、186 项通过；该权限问题通过专用 basetemp 复核后不再出现。矩阵运行中发生的运行器恢复/评估失败也均保留原始输出并由最终完整运行闭合。

## 口径与偏差

预期起点为 `2761e6f8dcd93eb6c1e14730298ac96f971c4f01`，实际运行经过必要的运行器恢复/评估修复；逐运行提交、输入哈希、98 个原始矩阵文件的大小和 SHA-256 见归档 JSON。pilot 使用 `71ffcd90...`；正式运行使用 `48806523...` 或 `a28ad3d...`；矩阵冻结/最终评估记录为 `a266df2...`。这些是为使已登记预算可执行和可恢复的工程修复，不应合并伪称单一源码提交。

本轮没有连接服务器，没有训练 raw rollout ledger；历史 raw ledger 缺失不由测试 ledger 补齐。没有启动新的融合矩阵或研究路线。

本地 ZIP 为 `E:\Z博士\9.2日\m10-eawm-auxiliary-history-matrix-20260914-v1.zip`，大小 192312533 bytes，SHA-256=`aa1576c7ae2740494bcfe93c118fae335102bfe6361506bf5d2db6f63bdfb550`；归档中心目录可读出 104 个 ZIP 条目。GitHub 分支已推送，但本轮 Release 资产上传两次均在零资产状态停滞；由本轮创建的空草稿已删除，未将远端资产写成已验证。

**状态判断：** 训练链路、checkpoint 保存/加载和有界矩阵均完成；E 相对 H 的任务收益未达到稳定改善证据，主机确认下降且完整决策链均值略高；世界模型规划目标仍未完成。本结果仅适用于本轮 EAWM-inspired 辅助表征实验，不宣称世界模型普遍无效。
