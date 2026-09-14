# M-10 到达协议 GPPO 融合单 seed pilot

**状态：** pilot 已完成；这是训练链路和初步行为证据，不是稳定增益、正式多 seed 对照或弱通信可用性结论。  
**run-id：** `m10-arrival-gppo-fusion-pilot-20260914-seed1101`  
**实际 run 输出：** `E:\Z博士\9.2日\WORLD-GPPO_9.11-local-runs\m10-arrival-gppo-fusion-pilot-20260914-v2`

**Release：** [m10-arrival-gppo-fusion-pilot-20260914-v1](https://github.com/Battleplus/WORLD-GPPO_9.11/releases/tag/m10-arrival-gppo-fusion-pilot-20260914-v1)  
**Release asset：** `m10-arrival-gppo-fusion-pilot-20260914-v1.zip`，3,645,759 bytes，SHA-256 `728884d396f88d12168a3a337699be11ab06af54dc33f02065872672c312bd62`。GitHub API 回读摘要一致；独立下载因传输超时未完成，因此不报告独立下载校验通过。

## 运行条件

- Git 起点：`bf71abeeb0e8cae703030f4def8ef58e9243722f`；本次训练时使用的 runner/fusion/training/protocol SHA 记录在 `run-identity.json`。
- 设备：NVIDIA GeForce RTX 3060 Laptop GPU，Torch `2.7.0+cu128`，CUDA `12.8`，Python `3.11.15`，NumPy `1.26.0`，4 threads，单进程、0 data workers。
- 协议：`world-gppo-9.11-arrival/0.1.0`，`arrival_to_region`，研发主指标为 `physical_arrival`；`host_confirmation` 独立保留。
- tape：composite weak communication；训练 32 条，验证 16 条；四组共享验证 tape、任务和环境合同。
- 策略预算：每学习组 512 environment steps；每组实际 8 次 `optimizer.step()`；每组上限128次更新/30分钟，均未触顶。
- 后果模型：`E:\Z博士\9.2日\WORLD-GPPO_9.11-m10-arrival-model-verify-20260913-v1\training\best-inference.pt`，SHA-256 `a5acd173b3c57301fd744cb1e6105789278de8f58438f0e42e7535a23e500b86`，`gppo-arrival-consequence-inference-v1`，horizon=6；本轮冻结，不更新。

## 结果（验证 16 episodes / 96 tasks）

| 组别 | 按时物理到达 | 按时主机确认 | 平均原环境回报 | 评估 actor calls | 评估 world calls | 评估完整链路均值 |
|---|---:|---:|---:|---:|---:|---:|
| 合法传统规则 | 53/96 | 12/96 | 20.8920 | 260 | 0 | 1.9718 ms |
| GPPO | 41/96 | 5/96 | 10.3401 | 274 | 0 | 5.7459 ms |
| GPPO-History | 42/96 | 7/96 | 11.2702 | 273 | 0 | 5.9613 ms |
| GPPO-History + 候选到达后果 | 44/96 | 7/96 | 13.1475 | 272 | 272 | 10.8856 ms |

各组 `pending_at_cutoff=0`，安全违规（重复接受、越权/失效 lease/fencing）均为 0。评估通信代理字节分别为 6,529,067、6,980,204、6,958,952、6,925,642；这是模拟审计序列化代理，不是真实网络流量。

融合训练期间实际消费 512 次 world-model prediction，候选行按当前 mask 逐动作输入 PPO；训练、采样和更新使用同一融合策略分布，旧 log-prob 与更新时重算 log-prob 一致。融合候选 prior 以零权重初始化并在 8 次更新中参与优化。

## 保存/加载核验

`checkpoint-load-verification.json` 对 GPPO、GPPO-History 和融合 checkpoint 均通过：state dict strict reload、optimizer state、recovery state 和首个公开决策一致。三份 policy SHA-256：

- GPPO：`59f1bba00f9d7b3ebbfb7c86a7f40a15e6ff878cd8986c829761300943d05db6`
- GPPO-History：`7eb5cb636ef43105becf5bec36160286394908195e597aaef842bc2b4f37cd5c`
- 融合：`a9fe80d6d5551e5ceedc103e24c74b7d8e577cff8730b3490a9584062cfe2381`

加载核验文件 SHA-256：`f1a40a9fdcc2baf1150ff52a5a68905e8cce00540475374a67dc904dfe36a70f`。

## 结论与边界

本 pilot 中融合组较 GPPO/History 的点估计更高（44/96 vs 41/96、42/96），但规则仍最高（53/96）；样本只有16条验证 tape、单 seed、8次更新，不能称稳定策略增益。融合延迟约为 History 的 1.83 倍，world 调用和额外计算成本必须单列。

因此：代码训练链路已真实执行，融合确实进入 PPO 分布；预测模型有效性和公平多 seed 调度收益仍未建立。按已登记边界，不自动启动多 seed 矩阵、扩样或新训练。

首次 v1 启动在规则基线入口处因 `None.policy.eval()` 失败，未发生 optimizer 更新；该失败输出已保留，v2 使用新唯一输出目录完成，未覆盖 v1。
