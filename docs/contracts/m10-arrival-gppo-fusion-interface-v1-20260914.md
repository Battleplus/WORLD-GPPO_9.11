# M-10 到达协议 GPPO 候选后果融合接口 v1

**状态：** 接口与固定 fixture 已实现；本版本没有启动训练、生成数据或 GPPO 融合实验。

## 适用合同

- 协议：`world-gppo-9.11-arrival/0.1.0`。
- 任务语义：到达目标区域即产生物理完成事实；`physical_arrival` 是研发主指标，`host_confirmation` 只作独立通信指标。
- 观测/动作：`m10-graph5-5type-25action-global27`，24 个 UAV–Task 候选加 NOOP。
- 运行约束：原默认通信、任务 deadline、奖励、动作吞吐、ACK、lease、fencing、能量和故障门禁；不启用事件触发，不改变通知配置。
- 信息边界：融合接口只接收公开 observation/Graph-5 snapshot；仿真分支真值、未来结局、隐藏状态和标签不会进入线上输入。

## 兼容性边界

| 制品 | 结论 | 原因 |
|---|---|---|
| `Graph5ArrivalConsequenceModel` + `gppo-arrival-consequence-inference-v1` | 可进入本接口 | 目标为到达时间、物理 deadline 概率、执行/能量失败；协议和动作合同可核验 |
| `Graph5JointConsequenceModel` + `gppo-m10-graph5-joint-consequence-v1` | 拒绝 | 旧模型输出是固定任务集合联合计数，不是本到达协议的三个候选目标；不能改名冒充兼容 |
| `M10WorldModel.context_for` | 不用于本路线 | 默认在动作确定前对合法候选求平均，会抹掉候选差异 |
| continuous-service 旧 GPPO checkpoint | 拒绝作为本实验起点 | 任务语义不同；即使张量形状相同，也不代表协议兼容 |

## 接口语义

`gppo_world/arrival_gppo_fusion.py` 提供：

1. `FrozenArrivalConsequenceScorer`：严格加载到达协议模型；检查格式、协议、SHA-256、Graph-5 输入和有限值；加载后 `eval()` 且冻结参数。
2. `CandidateFeatureContract`：固定特征顺序为 `arrival_mean_norm`、`arrival_logvar`、`deadline_probability`、`failure_probability`。其中到达均值除以冻结 `horizon_steps=6`，概率头使用 sigmoid；非法动作显式填零。
3. `CandidateAwarePolicy`：保留原 Graph-5 GPPO actor/critic，并将每个动作自己的四维预测通过一个小型可训练线性 prior 加到对应 actor logit；不把 25 个候选平均成一个 context。prior 零初始化，因此融合接口初始不凭随机权重改变基线行为。
4. `choose_candidate_action`：从公开 observation 构图、得到逐候选预测、计算 logits、应用原动作 mask 和固定最低 action-index 平局规则，再返回动作；调用方随后仍必须通过原环境 `step(..., submit_command=True)`，不能绕过执行门禁。

策略的历史向量仍由原 Graph-5 `history=True` policy 产生；融合组与 History 组必须使用同一历史字段、维度和初始化规则。候选预测只是辅助先验，不是真实遥测，也不是 Q 值或最优动作标签。

## 测试证据

```powershell
python -m pytest -q tests/test_arrival_gppo_fusion.py tests/test_m10_arrival_protocol.py
```

覆盖：逐候选行保留、模型冻结与评估模式、Graph-5/25-action 形状、非法动作 mask、低索引平局、旧协议拒绝，以及公开 observation→预测→policy→environment step 链路。

## 尚未声称的能力

本接口通过不等于 arrival 协议下的策略训练通过、预测质量通过、在线调度收益通过或弱通信可用性通过。当前仓库没有把旧持续服务 checkpoint 自动绑定为新到达协议 GPPO 起点；在有合约匹配的 arrival policy 和冻结模型制品前，不启动融合训练。
