# M-10 到达协议端到端 GPPO 融合计划（可执行版）

**登记日期：** 2026-09-14（汇报日期仍以会议确认记录为准）  
**代码起点核对：** `0396c7b3976b4ac71f7d4953e377ecf5a2f71b31`  
**本轮状态：** 计划与候选级融合接口完成；未训练、未生成新数据、未重跑历史评估。

## 目标和主张边界

要检验的唯一主张是：在冻结的 `arrival_to_region` 合同下，候选动作的后果预测能否在合法传统规则、GPPO 和 GPPO-History 之外提高按时物理到达。

物理按时到达是研发主指标；主机确认、通知和恢复单独报告。旧 `continuous_service_until_deadline` 结果、bounded-retry 工程收益、旧联合任务集合验证和历史负结果均保留在各自版本，不跨协议排名，也不归因给世界模型。

## 四组公平对照

| 组别 | 线上输入 | 训练/推理变化 |
|---|---|---|
| 合法传统规则 | 公开任务、距离/速度、deadline、能量和动作 mask | 不训练；同一到达协议和通信条件 |
| GPPO | 公开 Graph-5 + 原动作 mask | 新协议下独立策略预算 |
| GPPO-History | GPPO + 合法历史 | 与融合组共享历史字段、维度和初始化约定 |
| GPPO-History + candidate consequence | 同一 history + 每个合法候选自己的四维后果特征 | 冻结后果模型；只加候选 prior，不改奖励、通知或安全门禁 |

传统规则不能访问隐藏状态、仿真真值或候选分支标签。融合组不能把预测平均成全局 context，不能用标签提前选动作。

## 冻结的终点和统计

- 每个预定 episode 使用同一任务集合、tape、外生通信条件和观察截止；继续到任务完成、deadline 明确或记录 `pending_at_cutoff`，不延长任务 deadline。
- 逐任务保留物理到达、完成消息生成/发送/丢弃/送达、主机确认、deadline 和最终/截止状态。
- 主指标：按时物理到达数/预定任务数；次指标：按时主机确认、deadline 违约、正确拒绝、安全违规、通信代理字节、actor/world 调用和完整决策链延迟。
- 未决结局保留为删失/未决，不从分母删除；所有方法用同一适用性规则。统计按 episode/tape 和训练 seed 分组，不能把候选分支当独立样本。

## 当前实现和兼容性门

实现见 `gppo_world/arrival_gppo_fusion.py`，合同见 `docs/contracts/m10-arrival-gppo-fusion-interface-v1-20260914.md`。

运行前必须同时满足：

1. policy metadata 明确 `arrival_to_region`、`physical_arrival`、Graph-5/25-action 和 History 设置；旧持续服务 policy 直接拒绝。
2. 后果模型是 `gppo-arrival-consequence-inference-v1`，protocol 与 observation contract 一致，且登记完整 SHA-256；旧 `Graph5JointConsequenceModel` checkpoint 直接拒绝。
3. 所有候选预测来自当前公开 observation 的 Graph-5 snapshot；无未来、隐藏状态和标签字段；模型 `eval()`、参数冻结且全部输出有限。
4. 逐动作预测矩阵与动作 mask 同步，非法动作不可被 prior 重新放开；执行仍走原 ACK/lease/fencing/energy gates。

接口固定测试命令（本轮已执行，12 passed）：

```powershell
python -m pytest -q tests/test_arrival_gppo_fusion.py tests/test_m10_arrival_protocol.py
```

## 单 seed 有界探索 pilot（登记但本轮不启动）

该 pilot 是新预算，不重置或挪用已经结束的旧训练预算。后果模型必须是已经冻结并通过兼容性门的制品；本轮没有自动生成数据或补训模型。

| 项目 | 冻结值 |
|---|---:|
| 策略 seed | 1101 |
| 学习组 | GPPO、GPPO-History、GPPO-History + candidate consequence；传统规则只做同 tape 对照 |
| 每学习组环境交互上限 | 512 steps |
| PPO 实际 optimizer update 上限 | 128/group |
| 评估 | 预先固定的 16 条 arrival validation tape；仅开发/pilot 证据，不是独立 test |
| 后果模型额外数据 | 0；只读冻结模型 |
| 线程/进程 | 单进程；CPU 最多 4 线程；数据加载 0 worker |
| 墙钟 | 每学习组最多 30 分钟；任一组先到上限即停止该组 |
| 必须停止 | NaN/Inf、泄漏、非法动作、安全违规、输出冲突、终点未记录、资源不足或预算耗尽 |

pilot 启动前另写入实际 policy/model checkpoint、tape、配置和 SHA-256。输出目录非空必须显式 `--resume` 且 run identity 完全一致；不得覆盖历史运行。

### Pilot 命令边界

当前可直接执行的验证命令只有上面的接口/合同测试。仓库尚未把旧 `train_policy` 入口伪装成 arrival 融合 runner；实际 pilot 还必须将到达 `M10Config`、共享 tape、候选特征缓存、PPO rollout transition 和恢复状态接入同一个新 runner。伪造一个不存在的训练命令会制造不可复现证据，因此本轮不声称 pilot 已可启动。

后果模型若需要从冻结数据重新产生制品，使用既有入口并另设输出目录；以下仅是下一轮在输入哈希登记后可执行的命令模板，不是本轮执行命令：

```powershell
python tools/train_m10_arrival_consequence_model.py `
  --protocol configs/world-gppo-9.11-arrival-v0.1.0.json `
  --data <frozen-arrival-data> `
  --manifest <frozen-arrival-data>/manifest.json `
  --out <unique-arrival-model-run> `
  --run-id <unique-arrival-model-run> `
  --device cpu --seed 1101 --threads 4 `
  --epochs 8 --patience 3 --batch-size 16 `
  --max-updates 4096 --max-wall-seconds 3600
```

这条命令不应在没有冻结数据、模型 SHA 和新训练登记时运行；它也不替代尚未完成的策略融合 runner。

## Pilot go/no-go（查看结果前固定）

进入新的多 seed 对照仅在以下条件全部满足时提出：

- 三个学习组输出完整，所有 episode 有终态或明确截止状态，安全违规为 0，输入和 checkpoint 哈希稳定；
- 融合组没有因 candidate prior 引入非法动作、越权/错误 ACK、失效 lease 执行或隐藏信息泄漏；
- 融合组相对同 tape 的 History 组，按时物理到达差值不为负，且没有未决结局/通信成本的未解释增加；
- 该信号仍只称探索性 pilot。任一核心条件失败，停止融合矩阵并交付负结果/证据不足，不增加 seed、数据、目标或预算。

即使满足上述条件，也只能进入预登记的多 seed 评估，不能直接宣称全 episode 收益、GPPO 稳定增益或弱通信可用性通过。多 seed 阶段的环境步数、评估 tape、总墙钟和成本门槛须在 pilot 结束前另行登记，不能自动继承旧预算。

## 研究归因

- 预测误差/校准改善：世界模型证据。
- actor/world 调用、通信、延迟或通知变化：工程成本/工程改进证据。
- 按时物理到达、deadline 和恢复改善：只有在四组同合同配对结果中才是调度收益证据。

任何一项单独成立都不能替代另外两项。若规则已达到相同任务效果，或 History 与融合无稳定差异，当前路线应停止并报告世界模型没有额外调度价值；不能靠切换预测目标、删掉失败任务或延长 deadline 获得通过。
