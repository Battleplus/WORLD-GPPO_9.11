# 最小调度基准：候选动作后果学习结果

## 结论摘要

本轮在隔离的 2 UAV、4 子任务、两条长度为 2 的依赖链基准上，训练了三个 seed 的候选动作 regret 评分模型，并在冻结模型后只评估一次同分布 test。模型按合法公开初始图状态为每个合法首步动作预测归一化 regret，选择预测值最低的动作。

预登记条件全部满足：三 seed 平均相对冻结贪心的归一化 regret 减少约 **57.4%**；3/3 seed 的平均 regret 低于贪心；按同构/异构实例分层、每实例先平均三 seed 差异的 10,000 次 bootstrap 95% CI 为 **[0.00802, 0.03934]**，下界大于 0；数据内容去重、旧 32 实例隔离、精确标签和合法动作审计均通过。该区间只覆盖 test 实例（父实例）不确定性，不覆盖训练 seed 总体不确定性。

这证明了该小基准中存在首步动作选择价值，并支持提出后续有界研究；不证明 GPPO 收益、端到端多步调度收益、弱通信收益或生产可用性。`C*` 与 `Q*(s,a)` 使用精确求解器的最优续行，只是离线标签/参考，不是部署策略。

## 论文结构与本轮实现边界

依据用户提供的论文《Multi-UAV Dynamic Task Assignment Based on Event-Triggered Graph Reinforcement Learning Under Weak Communication》，论文页码/公式对应登记见 [m10-minimal-scheduling-benchmark-20260915.md](m10-minimal-scheduling-benchmark-20260915.md) 的论文结构表。本轮实现并验证了：UAV 与子任务二类型图、任务依赖、UAV 执行资格、正整数飞行/处理时间、单 UAV 非抢占占用、等待至完成事件释放资源、并行分配和 makespan 目标。

明确未实现：论文的完整 AHGNN/事件触发通信、分布式 leader 机制、弱通信/损毁/能量机制、论文完整任务规模与 GPPO 训练。因此本制品是最小结构基准，不是论文完整复现，也不替换旧到达协议。

## 冻结协议与数据

配置文件为 `configs/minimal-scheduling-benchmark-v1.json`。训练/验证/test 使用同一生成逻辑但不同固定 seed：

| split | 实例 | seed | 合法候选数 | 精确标签 |
|---|---:|---:|---:|---|
| train | 256 | 721001 | 851 | 256/256 完成 |
| validation | 64 | 721002 | 210 | 64/64 完成 |
| test | 128 | 721003 | 438 | 128/128 完成 |

每个 split 同构/异构各半。旧 32 实例仅作回归参照，不进入本轮训练或 test；内容审计发现跨 split 和与旧实例的内容重复均为 0。每个实例的 `C*` 及全部合法动作的 `Q*` 均已求解并检查 `Q* >= C*`（容差 `1e-8`）。归一化分母 `S(s)` 只由公开处理时间与初始位置—任务位置曼哈顿距离构成，且同一实例候选共享同一 `S`。

数据生成/标签命令与输入哈希记录在 dataset `run-identity.json`；数据主文件 SHA-256 为 `4c188caffffaeeeb4fcff0c3c91ccf15efb9350ce8517b6755653282a1d90475`。

## 模型、训练与正确性

`CandidateRegretNet` 使用 UAV/子任务节点、依赖边、资格边和候选相关公开飞行特征；hidden=64，消息传递 2 轮，评分 MLP 宽度=64。模型输入不含 `C*`、`Q*`、最优动作、求解器状态、split/seed/路径或实例编号。损失为实例平均的 `Huber(delta=1) + 0.1 * pairwise-rank-softplus`；只有真实 regret 不相等的候选对参与排序项。

形式训练固定为 Adam `lr=0.001`、`weight_decay=0`、batch=16、最多 50 epochs、patience=8、每 seed 最多 800 updates、每 seed 20 分钟。三次正式训练均因 validation patience 提前停止，没有追加预算：

| seed | epochs | updates | best epoch | best validation regret | 停止原因 |
|---:|---:|---:|---:|---:|---|
| 1101 | 15 | 240 | 6 | 0.00827385 | validation_patience |
| 2203 | 22 | 352 | 13 | 0.00777355 | validation_patience |
| 3307 | 29 | 464 | 20 | 0.00849102 | validation_patience |

独立 smoke 使用 32 updates，仅验证预算/数值/加载边界，不参与模型选择。三次正式训练的 best/last、逐 update 日志、validation 日志和 recovery 状态均保留。恢复文件含模型、optimizer、epoch、更新计数、早停状态、全局 RNG 快照和数据顺序记录；本轮未将 smoke 或任何中间 checkpoint 混入正式 test。需要说明：本轮制品保存了恢复所需状态，但没有把中断后的续跑另行作为正式实验结果。

定向回归为 **14 passed**（最小环境、候选模型、旧到达协议）；脚本 `py_compile` 通过。覆盖标签重算、平局排序损失、标签不进入模型输入、UAV/任务置换一致性和有限性检查。

## Test 决策结果

规则只有一份共享结果；精确参考为每实例 `C*` 的离线最优首步参考，不复制为训练 seed。主指标是归一化 regret，原始 regret 也同时给出。

| 方法/seed | 归一化 regret 均值 | 原始 regret 均值 | 相对贪心减少（归一化） | 首步命中任一最优 |
|---|---:|---:|---:|---:|
| 冻结贪心（共享） | 0.039626 | 0.976563 | — | 74.22% |
| 模型 1101 | 0.012163 | 0.289063 | 0.027464 | 86.72% |
| 模型 2203 | 0.021091 | 0.507813 | 0.018535 | 79.69% |
| 模型 3307 | 0.017339 | 0.414063 | 0.022287 | 83.59% |
| 精确参考（离线） | 0 | 0 | — | 100% |

按组分层：

| 组别 | 实例 | 贪心 regret | seed1101 | seed2203 | seed3307 |
|---|---:|---:|---:|---:|---:|
| 同构 | 64 | 0.023956 | 0.014496 | 0.022000 | 0.014124 |
| 异构 | 64 | 0.055297 | 0.009829 | 0.020182 | 0.020554 |

三 seed 非平局候选对排序准确率为 80.49%、78.02%、82.42%。全候选预测 MAE 为 0.08437、0.09010、0.08236；这些是辅助指标，不能替代动作选择 regret。test 推理为 CUDA forward-only、10 个实例预热、无文件写入：

| seed | mean ms | P95 ms | P99 ms | 样本 |
|---:|---:|---:|---:|---:|
| 1101 | 2.6180 | 3.2991 | 5.2457 | 128 |
| 2203 | 2.7488 | 4.3977 | 5.2763 | 128 |
| 3307 | 2.3847 | 2.8776 | 5.1322 | 128 |

延迟只适用于本机 RTX 3060 Laptop GPU、CUDA 12.8、当前 Python/PyTorch 环境和上述计时边界；不外推 GPU 竞赛实时性。完整结果及逐实例预测见 test artifact。

## 预登记判断与边界

- 平均相对贪心 regret 减少约 57.4%，超过 10% 条件；
- 3/3 seed 平均改善，超过至少 2/3 条件；
- 分层父实例 bootstrap 95% CI 下界为 0.00802，满足大于 0；
- 输入/跨 split 审计、精确标签完整性和合法动作检查通过；
- 因此满足“值得后续有界研究”的预登记信号。

这不是端到端调度收益：模型只选择初始首步，`Q*` 使用精确最优续行；没有训练 GPPO、没有旧到达环境、没有弱通信、没有 OOD，也没有把所有候选/任务当作独立样本进行强泛化声明。异构与同构结果分层报告，规则基线保持单份共享 test 结果。

## 制品、哈希与归档状态

代码分支为 `minimal-scheduling-benchmark-20260915`，本轮生成和训练使用的源码提交为 `86a7d1036f23f414261ba52d9054cb2c3b8f7736`；最终归档提交另行记录在 Git 日志和 `artifact-index.json`。运行环境为 Windows build 26100、Python 3.11.15、PyTorch 2.7.0+cu128、NumPy 1.26.0、NVIDIA RTX 3060 Laptop GPU（驱动 571.96）。

关键文件：

| 制品 | SHA-256 |
|---|---|
| dataset.json | `4c188caffffaeeeb4fcff0c3c91ccf15efb9350ce8517b6755653282a1d90475` |
| content-audit.json | `efc0023ef76a728ea4a01900d96906380a161abd23dc187b598d38d75e3f2f4e` |
| test results.json | `4d07d1deae61395f6c2a391f2e2aecf1a0d27847515b19fab60508c763650e1b` |
| test results-per-instance.json | `1aa5290e63cca25db5d2b3cc000dbfec4934fb3c2cfacf3adbbb16cb77c54579` |
| best seed1101 | `a3306884f989d7307a25e2007c7fa2c46113ca3d9f08f2d34e61bc15234e664b` |
| best seed2203 | `c572e2e907ccc2a84f343877ee4c4e0200e9972a344e8eb3e52ca89bbc71f903` |
| best seed3307 | `7fbe399ed480c8192a7ccac844c34d0de2ec27de0e136bce62f74602b4c0562a` |

完整训练目录（含 last recovery、updates 和 validation 日志）、数据、逐实例 test 结果及本报告由独立 Release ZIP 统一封存；上传/独立下载核验状态以 `artifact-index.json` 为准。旧到达协议、旧 checkpoint、旧 Release 和历史负结果未覆盖。
