# 修复后 P/T 矩阵账本核对与冻结 validation

日期：2026-09-19

范围：只读恢复链审计及已冻结开发 validation。没有训练、optimizer 更新、补步、新 tape 或独立 test。

## 结论摘要

恢复链的六个最终 checkpoint 均成功加载，策略与世界模型参数均为 finite，commit 中的计数与 checkpoint 一致。冻结 validation 共完成 **432/432 episodes、6080 个环境步**，每个配置、seed、通信条件各 16 个 episode，每 episode 6 个任务。没有读取独立 test。

主比较为 **T_train＋T_dispatch − P_train＋T_dispatch**，偏好固定为 `(0.8, 0.2)`，W1/W2 等权：按时物理到达率差为 **+2.08 个百分点**。16 个基础父 tape 的配对 bootstrap 95% CI 为 **[+0.17,+3.99] 个百分点**（10000 次重采样）。逐 seed 差值为 **1101：+1.56 pp；2203：−0.52 pp；3307：+5.21 pp**。因此 2/3 seed 为正，但收益主要受 seed 3307 拉动。

次比较 T_train＋T_dispatch 相对 P_train＋周期执行为 **+1.22 pp**，95% CI **[−1.22,+3.47] pp**，区间跨零。

这些 checkpoint 的逻辑训练量并不完全相等；置信区间只描述父 tape 不确定性，不覆盖训练 seed 不确定性。因此主比较仅构成开发信号，不能声称严格因果收益或稳定泛化。

## 训练恢复链

| Run | 环境步 reservation：reserved / verified / unknown | 策略调用 reservation：reserved / verified | 最终逻辑环境步 / 策略更新 | Pending 窗口 | Checkpoint SHA-256 |
|---|---:|---:|---:|---:|---|
| P_train/1101 | 8192 / 8192 / 0 | 63 / 63 | 7707 / 58 | 1 | `0CDE0624199964746B4D7739F1D5238D1339DF34407215679A2B3367C10D1D5E` |
| T_train/1101 | 8192 / 8192 / 0 | 63 / 63 | 8191 / 62 | 1 | `655C5AEA336340A0980FFF983EB711E2E528200D6C1997143ADFC15C7272C9BB` |
| P_train/2203 | 8192 / 8192 / 0 | 63 / 63 | 8192 / 63 | 1 | `579DC5C315F8828F52CF5A04F9ADC2C29613D0C1471CF9594FF90A780A90D9D7` |
| T_train/2203 | 8192 / 8192 / 0 | 64 / 64 | 8192 / 64 | 0 | `493DBA624E1ACA1B115D9CB85C8F95145AD12ED0D113CFD483F407106D` |
| P_train/3307 | 8192 / 8192 / 0 | 63 / 63 | 8192 / 63 | 1 | `28661EFFC5624EAA87B82AB91612930466948E9C89B22D1B09362C6A9FD28FBB` |
| T_train/3307 | 8192 / 8191 / 1 | 63 / 63 | 8122 / 62 | 1 | `A5C0440003B4B19C99726C8F6F38AB4B6012946FFFF13E0ADC94442861C590E2` |

全局 SQLite 预算：环境步 **49152 reserved / 49151 verified / 1 unknown / 0 pending**；策略优化器调用 **379 / 379 / 0 / 0**；world 优化器更新为 0。

唯一 unknown reservation 属于 `T_train/seed-3307`：reservation ID `673cf037a59247d987eddc55eabd0f68`，attempt `formal-v4-T_train-3307`，environment step 1，reservation index 97139。原因字段为“进程在未提交尾部退出，保留尾部且不退款”。该步没有确认为已执行，也未退款。

### T_train/1101 恢复说明

有效的 128 步 checkpoint（SHA-256 `8CC5DC446DD9588400EB9495E715C00A13B5B9DA1BE2B519B61595C14477F0B8`）作为恢复输入。首次更新因历史重放把续行动作错误地按“新命令动作 mask”检查而失败；该次失败更新没有形成完整提交事务。修复后从有效 checkpoint 继续，最终逻辑计数为 8191 步、62 次策略更新。已提交 ledger 没有证据表明原 128 步再次被晋升为训练样本；reservation 数和逻辑 checkpoint 计数分别保留，不能把差额推断为补齐或退款。

训练量不等：P/1101 为 7707 步/58 更新；T/1101 为 8191/62；P/2203 为 8192/63；T/2203 为 8192/64；P/3307 为 8192/63；T/3307 为 8122/62。它不是完全等训练量的严格单变量实验。

## 冻结 validation 结果

每格为按时物理到达任务数 / 96；I、W1、W2 分别为 ideal、轻度弱通信和中度弱通信。

| 配置 | Seed | I | W1 | W2 | W1/W2 等权完成率 |
|---|---:|---:|---:|---:|---:|
| P_train＋周期 | 1101 | 96/96 | 82/96 | 40/96 | 63.54% |
| P_train＋周期 | 2203 | 96/96 | 81/96 | 41/96 | 63.54% |
| P_train＋周期 | 3307 | 95/96 | 80/96 | 42/96 | 63.54% |
| P_train＋T_dispatch | 1101 | 96/96 | 82/96 | 37/96 | 61.98% |
| P_train＋T_dispatch | 2203 | 96/96 | 81/96 | 40/96 | 63.02% |
| P_train＋T_dispatch | 3307 | 95/96 | 79/96 | 42/96 | 63.02% |
| T_train＋T_dispatch | 1101 | 95/96 | 83/96 | 39/96 | 63.54% |
| T_train＋T_dispatch | 2203 | 96/96 | 81/96 | 39/96 | 62.50% |
| T_train＋T_dispatch | 3307 | 96/96 | 86/96 | 45/96 | 68.23% |

汇总完整 3-seed/条件整数分子：

| 配置 | I | W1 | W2 | W1/W2 等权 |
|---|---:|---:|---:|---:|
| P_train＋周期 | 287/288 | 243/288 | 123/288 | 63.54% |
| P_train＋T_dispatch | 287/288 | 242/288 | 119/288 | 62.67% |
| T_train＋T_dispatch | 287/288 | 250/288 | 123/288 | 64.76% |

等权结果先分别计算 W1、W2 完成率再取平均，不能把合并分母当成等权估计。

## 任务结果、安全与计算成本

| 配置 | 总按时物理到达 | 主机按时确认 | Deadline 失败 | 未决 | 能耗合计 | Actor 调用 | 续行步 | World 调用 | 通信消息代理量 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P_train＋周期 | 653/864 | 507 | 211 | 0 | 927.570 | 2035 | 0 | 2035 | 253938 |
| P_train＋T_dispatch | 648/864 | 504 | 216 | 0 | 927.461 | 1858 | 180 | 2038 | 254127 |
| T_train＋T_dispatch | 660/864 | 505 | 204 | 0 | 908.071 | 1821 | 186 | 2007 | 249490 |

三配置的记录内非法新动作、非法续行和安全违规均为 0；有合法非 NOOP 候选时的精确 NOOP 计数均为 0。以上只覆盖本次仿真记录，不是生产安全保证。

完整决策链耗时为每 episode 决策均值再汇总：P_train＋周期 **3.882 ms**，P_train＋T_dispatch **4.124 ms**，T_train＋T_dispatch **4.288 ms**。Actor 调用减少不等于总计算成本下降；world 调用并未同比减少。

通信代理字节字段在 432 条结果中均未观测，因此不报告字节数，也不把它记为零。决策时延只保存每 episode 的 mean/P95/P99 汇总，当前报告的是 episode 汇总统计，并非原始逐决策样本的精确 pooled quantile。验证在本机 CPU（Python 3.14.4、PyTorch 2.13.0+cpu、NumPy 2.5.0，4 intra-op / 1 inter-op 线程）完成。

## 数据与可复算制品

- 冻结 tape 总身份/内容清单 SHA-256：`5d1cfaed3683e3e1c512f4ff67d1f93566dae2b418942b67df3f1c81cb5eaea7`；本轮只使用每条件冻结 validation 中前 16 个父 tape。
- 本地运行归档：`E:\Z博士\migration-artifacts\wd-event-trigger-aware-gppo-20260918\window-budget-fix-v1\frozen-validation-20260919-v1`。
- episode ledger SHA-256：`09AC0CE5930291BE5D46A9899CDBEA5C427D0D41EDB2D03FDF0118214BE8AF28`。
- validation summary SHA-256：`5A310F734783E3BE32EA92E4A41A02ED83BE7DF62193337890BD752D275B52E0`。
- protocol SHA-256：`2D938F7E703B5B09F6DB4968A3DDA8295862D7BAD34F284942C1554FEE06C083`。
- recovery-chain audit SHA-256：`A51708D23CFF33AD5F589516A18D84E059267BCD065D9CB4BB8F323695F8DD0E`。
- source manifest SHA-256：`24EF4809BED0A28228FCE2BC471E9B0D0924E7E1A74D6DEFB9C699E71DFCE1EA`，清单覆盖 236 个实际源/配置文件。
- 本地制品 manifest SHA-256：`4A2D50AF71073BCC011F621D91D6A5BF679636ACCFA52AEF9C0597F5D77DD2EA`。

完整逐 episode ledger、六个 checkpoint、SQLite 原件及恢复审计仍在上述本地归档中；本次新增文档不包含大型模型或完整 SQLite 数据库。没有创建 Release，也没有将开发 validation 描述为独立 test 或正式收益验收。
