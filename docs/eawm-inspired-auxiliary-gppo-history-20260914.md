# EAWM-inspired auxiliary GPPO-History：开发验证记录

本记录对应独立分支 `eawm-auxiliary-gppo-history-20260914`，源版本为
`c17fc405b79e7f9dfc0f0f12ad0edb7384dca89e`。本轮没有正式训练、参数更新或
多 seed 矩阵；开发采集的参数更新数为 0。

## 方案边界

| 论文/会议概念 | 本轮实际实现 | 明确未实现 |
| --- | --- | --- |
| 事件感知历史表征 | 在 Graph-5/25-action GPPO-History 共享 GRU 表征上接训练专用辅助头 | 不是完整 EAWM 世界模型复现 |
| 事件标签 | 相邻、合法、可由公开身份对应的 UAV x/y、energy 三分类，以及 task `pending` 状态变化二分类 | 不预测隐藏故障发生；不读取未来物理真值 |
| 辅助目标 | 每个有效字段族先取平均，再对有标签字段族取平均；`L_total=L_PPO+0.1L_event` | 不改奖励、动作 logits、动作 mask、通信或安全门禁 |
| 在线使用 | `forward()` 只走策略和值函数；辅助头只在训练评价路径使用 | 不调用旧候选后果模型，不使用旧 prior，不恢复事件触发决策 |

辅助输入是当前公开 observation、策略自身 history 表征和实际采样 action 的 one-hot。
下一公开 observation 只用于构造监督标签；标签在进入损失前转为 detached tensor。
不同 episode 不连边，身份按公开 `entity_ids` 对齐，不按槽位推断实体。

## 开发数据结果

登记配置：独立开发 seed `471001`，8 个父 episode，最多 1024 环境步；Graph-5/25-action，
arrival-to-region，deadline 按 physical arrival，固定 composite weak-communication，
传统公开信息控制器。实际在 episode 终止后提前结束：8/8 父 episode、126 个环境步、126 条
相邻记录、约 0.3343 秒，未进行参数更新。

统计来自 `m10-eawm-auxiliary-dev-20260914-seed471001`：

| 字段族 | 有效标签数 | 类别计数 |
| --- | ---: | --- |
| UAV position x | 273 | decrease 1 / unchanged 251 / increase 21 |
| UAV position y | 289 | decrease 3 / unchanged 269 / increase 17 |
| UAV energy | 270 | decrease 159 / unchanged 111 / increase 0 |
| task public state | 313 | unchanged 292 / changed 21 |

身份匹配计数为 UAV 504、task 536。mask 原因累计为 identity mismatch 220、missing/invalid
903、nonfinite 0、episode boundary 0；公开值非有限计数为 0。位置和能量变化类别不均衡，
特别是 energy 没有 increase 类；这是本次小开发样本的事实，不是通过改阈值或删失败样本修正。
任务在尚未公开时身份为空，因此相应标签被 mask；这是合法的公开观察约束。

## 梯度、因果与推理验证

新增 `tests/test_eawm_auxiliary.py` 定向测试 10/10 通过；全套回归为 194 passed。
全套命令使用项目隔离 Python 并禁用无关全局 pytest 插件：

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& 'E:\Z博士\9.2日\world-gppo-911-local-pilot-venv\Scripts\python.exe' -m pytest -q --basetemp 'E:\Z博士\9.2日\WORLD-GPPO_9.11-eawm-aux-dev-wt\.pytest-all-eawm'
```

测试实际证明：相同相邻公开历史得到相同标签；未来变化不进入当前输入；身份换槽、缺失字段和
episode reset 正确 mask；辅助头不接收下一观测；辅助损失可反向传播到共享 GRU；系数 0、
同初始化同输入时 PPO 分支损失和梯度与原 History 一致；采样的 old log-prob 与未更新模型的
重算分布一致；推理 `forward()` 不调用辅助头；环境动作 mask/继续句柄接口未被辅助组件改变；
训练与推理 checkpoint 格式分离。

这些是接口、梯度链路和数据合法性证据，不是辅助任务预测质量或调度收益证据。

## 复现与制品

开发标签采集（已执行，不能覆盖非空输出）：

```powershell
& 'E:\Z博士\9.2日\world-gppo-911-local-pilot-venv\Scripts\python.exe' tools/generate_m10_eawm_auxiliary_dev.py `
  --out 'E:\Z博士\9.2日\m10-eawm-auxiliary-dev-20260914-seed471001'
```

未来有界训练入口已实现但本轮未调用：

```powershell
& 'E:\Z博士\9.2日\world-gppo-911-local-pilot-venv\Scripts\python.exe' tools/train_m10_eawm_auxiliary_history.py `
  --out 'E:\Z博士\9.2日\m10-eawm-auxiliary-train-seed1101' `
  --run-id eawm-auxiliary-train-seed1101 `
  --seed 1101 --device cuda --threads 4 `
  --environment-steps 8192 --max-updates 4096 --max-wall-seconds 3600
```

入口具有非空目录保护、显式 `--resume`、CUDA 不可用硬失败、arrival 协议固定、训练/推理
checkpoint 分离、RNG/数据顺序/早停状态保存，以及非有限 loss/梯度/参数/optimizer state
检查。上述命令只是下一轮入口，不代表本轮已训练或 CUDA 已验证。

开发制品位于：

`E:\Z博士\9.2日\m10-eawm-auxiliary-dev-20260914-seed471001`

其 manifest 中记录的 SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| `scenarios.json` | `88e403dd9fd24ddcbc5810d71f359b03de599b034bc672193c1201ea71c208f1` |
| `public-event-labels.jsonl` | `d058eb88ff98c7c86676214427cd31aacf7058a5ce21e687fd2117373485ca66` |
| `stats.json` | `8b88ce893032a959e238bda497d91f18d913789f0595e1a21c1d301b23a8e77e` |
| `manifest.json` | `a62102c73ca0d492dc5bc184367820c1d4769d399c6321c9cae59f176475da04` |

采集使用项目隔离环境：Python 3.11.15、PyTorch 2.7.0+cu128、NumPy 1.26.0；本轮采集
固定使用 CPU，未执行训练。历史正式 test/OOD、旧候选后果模型和正式策略结果未作为开发输入。

## 当前技术状态

已具备启动后续有界训练的技术条件：标签 schema、公开身份审计、训练专用辅助头、损失与
共享历史梯度链路、推理隔离、checkpoint 区分、采集入口和训练入口均已实现并通过本轮验证。
但本轮数据类别不均衡且尚未进行参数更新，所以“预测有效”“调度收益”“世界模型融合目标
完成”均未通过，需由上游决定是否启动后续预算。
