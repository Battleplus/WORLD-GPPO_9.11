# M-10 冻结 History 候选动作后果价值上界开发实验

## 结论

本实验在冻结的 H（seed=1101、累计 32768 个环境步）下完成了 24 个新开发父 episode、48 个预定前缀和 272 个候选分支。确认重复中只有 1/48 个前缀（1/24 个父 episode）比 H 多完成 1 个按时物理到达任务；其余 47 个前缀持平，没有前缀变差。该孤立信号未达到预登记的“至少 6 个不同父 episode 改善”条件，因此没有建立可重复的候选选择改善空间，不启动训练或进一步扫描。

这是一项固定 H、有限候选集合、固定续行控制器的开发诊断。发现重复只用于选动作，确认重复只用于评价；仿真真值未进入 H 的观测、动作 mask、隐藏状态或在线决策。所谓 oracle 仅指发现重复内的有限候选事后选择，不是可部署策略，也不是全局最优证明。

## 冻结输入与执行

- 基础源码提交：`8c06e528101f1af54131d90b499cfbd7fe04ab28`；执行分支：`history-action-value-upper-bound-20260915`。
- H checkpoint：`E:\Z博士\9.2日\m10-history-budget-extension-20260914-v1\runs\m10-eawm-aux-h-1101-budget-extension\checkpoint-32768.pt`。
- checkpoint SHA-256：`e77a8ccba9f65d806eeb27e8404c8b3ea0600295dfa0afe43bfb154892919027`。
- checkpoint 格式：`gppo-history-arrival-training-v1`；累计环境步 32768；seed 1101。
- 协议保持 arrival / physical-arrival deadline、Graph-5/25-action、原奖励、NOOP、合法动作 mask、通信/ACK/lease/fencing、能量和独占门禁。
- 生成器：`weak_communication_tape("train", count=24, base_seed=581001, level="composite")`；数据标记为 development，不是新的盲测。
- 每个父 episode 预定 decision-2、decision-5 两个前缀；每前缀候选顺序为 H、原规则、合法 NOOP、剩余合法动作中 H 概率最高者，去重后实际保留 1 或 3 个候选。
- 重复 0/1 为 discovery，重复 2/3 为 confirmation；同一重复内候选使用相同地址化外生 key 语义，后续由同一冻结 H 确定性续行。

实际命令：

```powershell
E:\Z博士\9.2日\world-gppo-911-local-pilot-venv\Scripts\python.exe tools\run_m10_history_candidate_value_upper_bound.py `
  --checkpoint E:\Z博士\9.2日\m10-history-budget-extension-20260914-v1\runs\m10-eawm-aux-h-1101-budget-extension\checkpoint-32768.pt `
  --out E:\Z博士\9.2日\m10-history-action-value-upper-bound-dev-20260915-v4 `
  --search-root E:\Z博士\9.2日 `
  --device cuda --threads 4
E:\Z博士\9.2日\world-gppo-911-local-pilot-venv\Scripts\python.exe tools\analyze_m10_history_candidate_value_upper_bound.py `
  --run E:\Z博士\9.2日\m10-history-action-value-upper-bound-dev-20260915-v4 `
  --out artifacts\m10-history-action-value-upper-bound-dev-20260915\analysis.json
```

## 分母、完整性和结果

| 项目 | 结果 |
|---|---:|
| 预定 / 实际父 episode | 24 / 24 |
| 预定 / 实际前缀 | 48 / 48 |
| 完整 confirmation 前缀 | 48 |
| confirmation 删失或缺失 | 0 |
| 候选分支 | 272 |
| 分支环境步 | 3560 / 50000 |
| 候选数分布 | 1 个候选：38 前缀；3 个候选：10 前缀 |
| 至少两个可比较候选 | 10 / 48 前缀 |
| 安全违规 | 0 |

confirmation 以固定任务集合中的按时物理到达数量为主指标：选中动作相对 H 的前缀平均差为 `+0.0208333` 个任务，1 个前缀改善、47 个持平、0 个变差。按父 episode 汇总的探索性 bootstrap 单位为“父 tape 内前缀 confirmation 差值的均值”，24 个独立父 tape、10000 次重采样、seed 20260915，95% 区间为 `[0.0000, 0.0625]` 个任务/前缀。该区间只描述父 tape 不确定性，不覆盖训练随机性；本实验只有一个固定 H checkpoint。

主机按时确认的选中动作相对 H 平均差为 `+0.0104167` 个任务/前缀，1 个前缀改善、47 个持平、0 个变差。不得把这一通信结果当作物理到达收益，二者分开报告。

发现重复上的事后最优相对 H 平均差为 `+0.0416667` 个任务/前缀；该数字仅是发现样本内的选择参考，不能与确认结果混用。

选中动作与 H 相同的比例为 `46/48 = 95.83%`，与原规则相同的比例也是 `46/48 = 95.83%`。原规则相对 H 的物理到达和主机确认差在本实验全部 confirmation 配对中均为 0；规则与 H 在两个发生候选分歧的前缀上都选择了 H 动作。

唯一确认物理改善前缀为 `train-mixed-seed-581007|decision-5`：H 与规则均选动作 7，发现阶段选中的剩余高概率合法动作 19 在两个 confirmation 重复中物理到达分别为 1/1，而 H 为 0/0。主机按时确认在两次重复中分别为选中动作 1/0、H 0/0。该前缀的有限样本通信代理字节分别比 H 高 1656 字节（序列化 JSON communication log），不是实网流量。另一个非 H 选择前缀 `train-mixed-seed-581019|decision-5` 在 confirmation 中物理结果为 1/1 对 1/1，主机确认均为 0/0，未产生收益。

候选方向在四次重复中完全一致的前缀为 33/48；由于多数前缀候选去重后只有一个，这只是随机波动诊断，不是稳定期望收益证明。

## 对四个问题的回答

1. H 存在一个很小的、孤立的候选改善信号，但不满足跨父 episode 的可重复标准；当前不能称为可重复改善空间。
2. 公开规则在全部配对中与 H 持平，并未取得唯一正向前缀的动作改善；但候选集合有效分歧仅有 10/48 个前缀，规则/H 的总体持平不能证明规则一般性最优。
3. 发现重复挑出的“最优”只在 1 个前缀的确认中保持物理改善，另 1 个非 H 选择确认持平；因此发现最优没有形成总体可重复证据。
4. 唯一正向前缀同时出现物理到达和少量主机确认改善，现有数据不支持把主要差距归因于消息确认；但样本极少，不能做一般因果归因。总体上，本轮更明确地检验了首步任务选择，通信确认仍是独立指标。

## 验证状态与失败记录

有效 v4 运行的 verification 为 passed：固定任务集合一致、分支前公开输入一致、输入无未来/隐藏真值泄漏、外生 key 配对一致、发现/确认隔离、删失不计为失败、ledger 可核对，安全违规为 0。

v1 因规则动作去重后的选择记录实现缺少 `rule_action` 而停止；v2 因审计目录包含 v1 自身部分生成 ID 而停止；v3 因重新调用观测函数消费 transient `task_arrival` 标志而验证失败。上述目录均保留但不纳入 v4 统计。v4 修复只涉及记录/验证路径，没有修改模型、阈值、协议或分支执行语义。

因此本轮预登记信号为：完整父 episode ≥12（满足）、至少 6 个改善父 episode（不满足，1 个）、confirmation 总体均值 >0（满足）、无泄漏/安全/账本缺口（满足）；总条件不满足。按要求停止，不训练、不扩样、不启动 GPPO 融合。

## 制品

完整分支账本和配置保留在 `E:\Z博士\9.2日\m10-history-action-value-upper-bound-dev-20260915-v4`；仓库内保存分析摘要、运行工具和本报告。原始 checkpoint 不复制、不修改；上一阶段中间 checkpoint 覆盖问题及原 ZIP 哈希保持不变。

