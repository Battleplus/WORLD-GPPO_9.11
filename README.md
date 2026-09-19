# WORLD-GPPO_9.11

> **最新进度（2026-09-19）**：已完成事件触发 P/T 矩阵的恢复账本核对，并在冻结 validation 上完成 432/432 个 episode（6080 环境步）；本轮没有训练、参数更新、补步或生成新数据。主比较 T_train＋T_dispatch 对 P_train＋T_dispatch，在 W1/W2 等权按时物理完成率上为 **+2.08 个百分点**，父 tape bootstrap 95% CI **[+0.17,+3.99] 个百分点**；逐 seed 差值为 **+1.56、−0.52、+5.21 个百分点**。由于六个 checkpoint 的逻辑训练量不完全相等，这只能作为开发信号，不是严格因果收益结论。安全违规记录为 0（仅限本次日志覆盖范围）。详见[本阶段报告](docs/experiments/pt-frozen-validation-20260919.md)。checkpoint、逐 episode ledger 和完整审计制品保存在本地独立归档；本次新增文档仅记录进度，不包含大制品。

README 下方较早章节是历史阶段记录；如与上述日期更新冲突，以有日期的最新报告及对应协议/账本为准。

本仓库是后续研究的唯一归档入口（目标远端：`Battleplus/WORLD-GPPO_9.11`）。内容从旧项目 `GPPO-WORLD-9.2` 的提交 `05702ae861e460e667acf8f0184e9a3709ef6fd4` 迁移而来；旧仓库、旧 Release、失败记录和负结果不被覆盖，且不跨协议混合统计。

当前主题仍是“无人机弱通信任务调度中的世界模型优化”，但项目已不再停留在接口或训练入口阶段：到达协议、通知工程候选、联合后果模型、到达协议下 GPPO 融合、多 seed 对照、偏好/事件/世界模型联合链路，以及事件触发 P/T 开发验证均已产生实际运行证据。当前没有建立“世界模型稳定改善调度”的结论；规则基线、学习基线、工程收益和研究负结果继续分别保留。请先阅读 [迁移来源](docs/transition/legacy-provenance.md)、[任务合同](docs/contracts/world-gppo-9.11-task-contract.md)、[后果模型设计](docs/world-model/action-consequence-design.md) 和 [最新 P/T 报告](docs/experiments/pt-frozen-validation-20260919.md)。

## 项目学习笔记：从起点到当前

这部分不是把所有运行统一包装成“成功”，而是记录研究过程中真正学到的内容：哪些合同已经实现，哪些局部收益成立，哪些方案在独立验证中没有站住，以及下一次读代码或看结果时最容易混淆什么。

### 1. 起点：先把因果性、安全边界和证据规则写清楚

项目最早解决的不是“换一个更大的网络”，而是定义世界模型在线时究竟能读取什么。T-00/T-01 固定了合法公开观测、实际执行动作、future/truth denylist、完整 episode/tape/seed 切分、graph/action version，以及 ACK、lease、fencing 和真实 action mask 的权威关系。

核心学习：预测模型只能读取决策时已经合法公开的信息；仿真真值、未来消息和反事实结果只能用于离线监督或评价。安全链始终高于模型输出，模型不能写 belief、mask、版本或直接提交动作。

### 2. 第一代世界模型：预测能学会，不等于策略会受益

T-02～T-04 完成了动作条件 Graph-WM、事件头、GES、校准和只读 Shadow runtime。动作 shuffle、no-action、合法动作反事实等测试证明模型确实使用动作信息；事件预测和部分 state/reward/cost 指标也超过了简单基线。Shadow 阶段验证了零写入、版本一致性、风险回退和本地延迟。

T-05 随后把冻结 latent 接入 GPPO，并完成四组、多 seed、固定预算训练。工程链路和安全 Gate 成立，但真实 held-out 策略收益没有稳定建立。这形成项目最重要的第一条经验：

> 预测误差下降、事件分类变好、接口被真实调用，都不能自动推出任务完成率提升。

相关证据见 [T-05 最终报告](nodes/T-05/evidence/final-report.md) 和 [阶段重新分析](docs/09-project-reassessment-20260905.md)。

### 3. 到达任务协议：必须把“物理完成”和“主机知道完成”分开

M-10 将任务语义冻结为 `arrival_to_region`，研发主指标是 deadline 前物理到达；完成消息生成、发送、合法收到和主机确认使用独立时间链。16 条配对 tape、96 个任务的系统闭环显示：

| 条件 | 按时物理到达 | 按时主机确认 |
|---|---:|---:|
| ideal | 96/96 | 95/96 |
| 冻结弱通信 composite | 48/96 | 9/96 |

弱通信下 48 个未到达任务的互斥主因是：调度未选择 20、执行门禁/资源 16、时间或执行预算 9、命令传输 3。已物理到达但未按时确认的 39 个任务则落在通知未生成/丢弃、截止前未合法收到、晚确认和身份/重复拒绝链路。这里学到的是：分类名称不等于代码缺陷，且“任务完成”和“控制端获知完成”不能互相替代。详见 [到达协议系统闭环](docs/results/m10-arrival-system-availability-closure-20260914.md)。

### 4. 通知可靠性：工程收益真实，但不能冒充世界模型贡献

项目实现了独立完成通知、稳定通知 ID、主机幂等处理、独立 ACK 和有界重试。独立 16 个父 tape 的验证中，弱通信按时主机确认由 10/96 提升到 24/96，按时物理到达均为 47/96；通信代理量增加 60,504 bytes。ideal 条件还出现 1 个物理到达和 1 个确认损失。

因此原系统继续作为默认，`原调度器 + bounded_retry` 保留为工程候选。这个结果说明通知协议能修复“已完成但主机不知道”的一部分问题，但它没有提高弱通信物理任务净完成数，而且存在通信与时序成本。详见 [通知方案独立验证](docs/results/m10-notification-independent-validation-20260914.md)。

### 5. 调度规则扩展：局部直觉可能破坏全局任务竞争

`arrival-slack` 和 combined 方案试图优先救回看似紧迫的任务，但弱通信物理到达明显下降，因此没有设为默认。负结果说明：一个任务看起来更紧迫，不代表占用 UAV 后对整个任务集合更有利；资源占用、后续可执行资格和其他任务机会成本必须一起考虑。

这一步把研究问题从“预测单个任务何时到达”改成“预测选择某动作后，当前任务集合的联合后果”。

### 6. 联合后果模型：开发集出现排序信号，独立价值仍未建立

联合后果数据固定同一决策前缀、同一公开任务集合、同一续行控制器和配对外生随机条件，只改变首步合法动作。开发 pilot 生成 843 条分支；validation 有 9 个可比较前缀，其中 4 个存在非零候选差异。轻量模型在开发 validation 上排序命中 9/9，而公开联合基线和零差值基线均为 5/9；但配对差 MSE 反而劣于零差值基线，而且 validation 同时用于开发和早停。

后续独立验证受到动作相关删失和有效分母差异影响，正向点估计不足以识别总体效果。最终结论是“独立决策价值尚未建立”，不是“世界模型普遍无效”。详见 [开发 pilot](docs/results/m10-joint-consequence-development-pilot-20260914.md)、[独立验证](docs/results/m10-joint-consequence-independent-validation-20260914.md) 和 [阶段收束](docs/results/m10-joint-consequence-stage-closure-20260914.md)。

### 7. 到达协议下的端到端 GPPO 融合：链路真实，增益不稳定

随后完成了真正的到达协议融合 runner。候选后果预测按候选动作进入 PPO logits；采样保存的 old log-prob 与更新时重算使用同一策略分布；冻结世界模型没有被策略优化器更新。单 seed、512 步 pilot 中，规则/GPPO/History/融合分别为 53/96、41/96、42/96、44/96，融合完整链路均值约 10.89 ms，History 约 5.96 ms。

正式三 seed、每组 8192 步、64 个新测试父 tape 的结果是：

| 方法 | 按时物理到达 | 完整链路均值 |
|---|---:|---:|
| 共享合法规则 | 196/384（51.04%） | 1.40 ms |
| GPPO | 565/1152（49.05%） | 4.66 ms |
| GPPO-History | 540/1152（46.88%） | 6.23 ms |
| History＋候选后果 | 542/1152（47.05%） | 10.67 ms |

融合只比 History 多 2/1152 个按时到达任务，三个 seed 差值为 −11/+9/+4；父 tape bootstrap 区间跨 0。它没有超过 GPPO，也没有超过规则，同时完整链路均值比 History 高约 71%。因此它保留为研究制品，不作为工程默认。详见 [单 seed pilot](docs/results/m10-arrival-gppo-fusion-pilot-20260914.md)、[多 seed 矩阵](docs/results/m10-arrival-gppo-fusion-matrix-20260914.md) 和 [最终选型](docs/results/m10-arrival-gppo-fusion-selection-20260914.md)。

### 8. 四模块探索：模块“进入训练链”与模块“产生收益”是两件事

后续研究把 GPPO、显式偏好、动作条件世界模型和事件预测统一到同一训练链，并逐步验证了逐候选预测消费、偏好条件 actor/critic、世界模型与事件头梯度、checkpoint 恢复和事务账本。过程中暴露了三类重要问题：

- 状态级向量价值不能替代动作级策略学习信号；旧 PreCo 工程适配曾出现偏好梯度不足，后来改为逐样本偏好加权 PPO 进行隔离研究；
- 世界模型共享编码器更新、候选特征分支和事件辅助损失可能改变策略分布，必须分别记录更新前后 KL、NOOP 概率与参数归属；
- 某些运行出现整 seed 全 NOOP，不能只凭“梯度非零”或低预测误差解释，需要区分基础策略、偏好分支、候选分支和训练访问分布。

这些探索形成了可运行的联合架构和大量负结果，但截至当前，偏好可控性、事件模块独立增量和世界模型稳定任务收益都没有形成可推广结论。部分大型 ledger/checkpoint 仍仅在本地独立归档，不把未完成的 GitHub Release 写成已发布。

### 9. 事件触发 P/T：减少 actor 调用不等于降低完整成本

最新阶段比较周期训练 `P_train` 与事件触发训练 `T_train`，并区分周期执行和 `T_dispatch`。实现过程中修复了两个关键工程合同：跨策略更新的自然决策窗口，以及 `submit_command=false` 的合法续行在历史重放中不能重新套用“新命令 mask”。同时引入跨 attempt 共享、在 `env.step` 前预留的 SQLite 预算账本，避免失败重试绕过总预算。

冻结 validation 的 432 个 episode 显示：`T_train＋T_dispatch` 相对 `P_train＋T_dispatch` 在 W1/W2 等权按时物理到达率上为 +2.08 个百分点，95% CI 为 [+0.17,+3.99]；但逐 seed 是 +1.56/−0.52/+5.21 个百分点，且六个 checkpoint 的逻辑训练量不同。Actor 调用下降，但 world 调用没有同步下降，完整决策链均值反而从约 4.12 ms 增至 4.29 ms。因此它只是值得后续公平验证的开发信号，不是已建立的训练节奏收益。完整恢复链、预算差异和结果见 [最新 P/T 报告](docs/experiments/pt-frozen-validation-20260919.md)。

### 10. 截至当前可以确认什么

| 类型 | 可以确认 | 不能据此声称 |
|---|---|---|
| 工程实现 | 到达协议、Graph-5/25-action、History、世界模型/事件头、偏好路径、通知重试、PPO 融合、恢复和事务账本均有实际实现或运行证据 | 所有组合都已达到生产可用 |
| 预测能力 | 部分世界模型和事件指标优于简单基线；候选后果模型在开发集出现排序信号 | 预测更准必然改善调度 |
| 通知工程 | bounded retry 在独立弱通信集提高主机按时确认 | 它提高了物理任务完成，或属于世界模型贡献 |
| 调度/策略 | 多轮 GPPO、History、融合和事件触发实验已经真实训练或评价 | 当前世界模型融合具有稳定跨 seed 增益 |
| 安全 | 已记录仿真范围内多轮运行未观察到安全违规 | 生产安全保证或未记录链路也为零风险 |
| 成本 | 已记录代理通信量、actor/world 调用和仿真决策延迟 | 代理字节等于实网流量，或本机时延等于部署时延 |

### 11. 阅读结果时必须遵守的口径

1. 旧持续服务协议、到达协议和最小调度基准不能跨版本排名。
2. `physical_arrival`、`host_confirmation` 和“观察截止时最终获知”必须分别报告。
3. validation 参与开发或早停时，只能称开发证据；独立 test 不能在查看后继续调参。
4. 父 tape bootstrap 通常只覆盖场景不确定性，不覆盖训练 seed 总体不确定性。
5. 通信代理字节不是真实网络流量；分支仿真耗时也不是部署推理延迟。
6. 安全违规为 0 只限账本实际覆盖范围，不能替代真实部署保证。
7. 训练完成、接口接通、预测有效、策略收益和工程采用是五个不同结论。
8. 失败、删失、unknown reservation、未提交尾部和未上传制品必须保留，不能靠补写或换分母消失。

### 12. 当前工程与研究选型

- 合法规则仍是当前仿真工程参考；GPPO 是学习基线。
- `bounded_retry` 是有独立确认收益、但成本门槛未定的工程候选，不是默认。
- 冻结候选后果融合、联合后果模型、偏好/事件四模块和事件触发训练均保留为研究制品或开发证据。
- 当前没有证据支持把任何世界模型融合方案升级为工程默认，也没有证据证明世界模型方法普遍无效。
- 后续实验必须预注册任务效果、通信/计算成本、训练预算、独立评价和停止条件；负结果不触发自动扩训或更换指标。

### 13. 下一阶段唯一方向：严格等预算验证事件触发训练

当前最值得回答的问题不是继续增加网络、预测头或训练量，而是：上一轮 `T_train＋T_dispatch` 的正向开发信号，在排除训练量不等之后是否仍然存在。下一阶段只比较：

- **P：**周期训练 `P_train`＋事件触发执行 `T_dispatch`；
- **T：**事件触发训练 `T_train`＋事件触发执行 `T_dispatch`。

这两个组必须冻结同一个世界模型和事件模型，并保持奖励、偏好、网络、触发阈值、训练父 tape、初始化、外生通信随机协议和安全门禁不变。world optimizer 继续为 0；不得同时增加网络结构、候选 prior、预训练、奖励塑形或新的事件标签。

公平性合同必须补齐上一轮的混杂因素：

1. 每个 seed 完成相同的有效环境步和策略 optimizer 更新数；
2. 使用同一事务化预算执行器，`unknown reservation`、未提交尾部和失败重放分别记账；
3. checkpoint、ledger、RNG、训练顺序和恢复点必须属于同一完整事务；
4. actor 决策次数允许因触发机制不同而变化，但必须记录，不能通过补动作或强制重规划把轨迹做成相同；
5. 训练失败不能用额外步数补齐，若无法形成等预算模型，则停止正式因果比较。

训练完成后只允许在一批从未读取的新父 tape 上做一次独立测试。当前 validation 继续作为开发证据，不能再次充当盲测。主指标固定为 W1/W2 等权的 deadline 前物理到达率差：

\[
\Delta = \text{physical-arrival}(T) - \text{physical-arrival}(P)
\]

同时报告按时主机确认、deadline 失败、未决任务、实际能耗、actor/world 调用、通信代理量、完整决策链 mean/P95/P99，以及账本覆盖范围内的安全违规。Actor 调用减少不能替代完整计算成本；通信代理字节仍不等于真实网络流量。

预注册的继续信号应至少包括：平均物理到达率提升 1 个百分点、至少 2/3 seed 改善、父 tape 配对 95% 区间下界大于 0、没有新增记录内安全违规，且完整计算成本符合预先声明的门槛。父 tape 区间只描述场景不确定性，不能冒充训练 seed 总体不确定性。

若该严格复验通过，后续才单独比较 T 框架中“打开/关闭世界模型候选特征”，从而区分事件触发训练收益和世界模型增量。若复验不通过，则封存当前正向点估计并停止事件触发训练扩展；不得通过追加 seed、改阈值、换指标或延长训练追求正结果。

### 14. 如何提升，以及论文应建立在什么证据上

当前没有超过强规则基线，主要问题不是“世界模型没有被调用”，而是预测目标、策略消费方式和真实调度目标尚未充分对齐。已有实验支持以下判断：

1. 下一状态、latent、事件分类或 reward 预测变准，不保证合法候选动作的真实排序变好；
2. 当前规则直接使用距离、资格、资源和 deadline 等公开结构，在短任务和弱冲突场景中是强基线；
3. 固定续行控制器下的候选价值不一定能外推到持续学习的 GPPO 访问分布；
4. 如果“完成更多任务”和“消耗更多能量”的公开可辨冲突很少，偏好向量就缺少足够的行为监督；
5. 候选特征即使改变参数，也可能不足以改变最终动作排序，或改变动作却不改善任务结局。

因此，若第 13 节的严格 P/T 复验通过，下一代世界模型只应检验一个可证伪改动：将通用状态预测收敛为**事件决策窗口对齐、候选相关、偏好条件化的任务集合相对后果预测**。目标可写为：

\[
\Delta(s,a,p)=
\begin{bmatrix}
\Delta\,\text{按时物理完成} \\
-\Delta\,\text{deadline失败} \\
-\Delta\,\text{实际能耗}
\end{bmatrix}
\]

其中 `s` 只包含当前合法公开状态与历史，`a` 是合法候选动作，`p` 是任务—能耗偏好，`Δ` 是相对预先冻结合法参考动作的配对差。标签窗口从当前触发决策点延伸到下一触发点或相关任务 deadline；续行控制器、外生随机键和任务集合必须在候选间配对。仿真真值只用于离线标签和评价，不能进入在线策略。

策略消费应采用从零开始的残差与不确定度门控，而不是让候选预测直接覆盖强基础策略：

\[
\text{logit}(a)=\text{base-logit}(a)+g(u_a)\,\alpha\,\widehat{\Delta}(s,a,p)
\]

其中 `g(u_a)` 在预测不确定、输入越界或版本不匹配时退回基础策略。这个设计的目的不是保证正结果，而是使“模型是否提供额外决策信息”成为可测量命题，并减少整 seed NOOP 塌缩风险。

后续实验顺序必须保持单变量：

1. 先完成等预算 P/T 独立复验；
2. 固定 `T_train＋T_dispatch`、History 和偏好加权 PPO 为强学习基线；
3. 比较无世界模型、现有通用世界模型和候选相对后果模型，保持容量、预算和安全合同尽量一致；
4. 只使用最终冻结 checkpoint 读取一次独立 test；
5. 同时报告任务效果、预测排序、完整计算成本和安全范围，不能用 MSE 或 logits 变化替代任务收益。

若要形成“世界模型改善调度”的算法论文，至少需要：多数训练 seed 同方向、父 tape 配对区间不跨零、在预注册 W1/W2 范围超过 T 基线，并且收益不是由全 NOOP、放宽 deadline、牺牲确认率或不可接受的计算成本换来；相对合法规则是否超过必须单独报告。删除候选头后收益应消失或明显减弱，才能支持世界模型增量归因。

如果最终仍不能超过规则，项目仍可收敛为系统/实证论文，但论文主张必须改为：在弱通信 UAV 调度中，世界模型何时有用、为何预测改善经常不能转化为决策收益。可复核贡献包括到达与确认分离协议、任务级失效归因、幂等通知与有界重试、动作条件世界模型的真实 PPO 融合、事务化预算与恢复账本，以及跨多种融合方式保留的负结果。此路线不能包装成 SOTA 调度算法。

对应的研究叙事应保持为：**先公平验证事件触发训练，再固定强 T 基线，只改变候选后果监督目标；正结果支持算法论文，负结果支持可信的系统实证论文。**无论哪条路线，都不再以增加网络、追加 seed 或延长训练作为默认修复。

本仓库用于把“事件感知世界模型”迁移到 GPPO 动态任务分配系统，并保存从设计、数据、模型、联调到实验验收的完整证据。

一句话概括最终目标：

> 让世界模型学习“在当前 belief 图中实际执行某个动作后，系统可能怎样变化”，再把经过验证的预测 latent 提供给 GPPO；GPPO 仍然是唯一动作选择器，真实 action mask 和执行安全链始终拥有最终权威。

本项目以 [`Battleplus/GPPO-8.29@2a9bb9f`](https://github.com/Battleplus/GPPO-8.29/commit/2a9bb9f87b9d543df144f4d108ba970c924151f9) 为固定设计基线，参考 EAWM 的自动事件、Event Predictor 和 GES 思想，但针对 UAV–Region–Target 异构图重新实现，不直接照搬 Atari 图像模型。

> 2026-09-05 重新分析：T-00～T-05 已按各自已执行协议验收，四组 × 三 seeds 消融已封存；原始目标要求的 GPPO-History 对照仍未完成，不能等同于原始完整验收全部满足。稳定策略增益尚未建立。阅读 [重新分析](docs/09-project-reassessment-20260905.md)、[修订规划](docs/10-revised-experiment-plan-20260905.md) 与 [原始要求覆盖表](nodes/requirements-status.json)。历史结果见 [T-05 报告](nodes/T-05/evidence/final-report.md)。

当前执行入口：[9 月中旬分阶段交付计划](nodes/M-09/README.md)。按会议要求优先基础功能、延迟验证和汇报交付，再推进世界模型研究。

## 2026-09-06 最新交付

M-09 S5 已形成候选交付包并完成服务器独立复现、PPTX/PDF 渲染检查与归档校验：[S5 交付目录](nodes/M-09/S5/README.md)。默认 A 为纯 GPPO + 可关闭只读 Shadow；历史 B 为 EAWM-GPPO adapter 消费复现。S3 明确跳过且未训练；实际彩排/评审尚未举行，9/13 冻结与 9/15 汇报仍待确认。GitHub Release：`m09-s5-delivery-freeze-v1-20260906`（候选）。

2026-09-05 已完成 R-02、J-02A 和服务器 J-02B 四配置三 seed 实验。J-02B 开发门槛失败，完整训练文件已发布至 [GitHub Release](https://github.com/Battleplus/GPPO-WORLD-9.2/releases/tag/j02b-server-archive-20260905)。原始 GPPO-History 对照和稳定策略增益仍未完成；原始 R-02 大文件尚待独立 Release。

## 9 月中旬汇报准备度

截至 2026-09-06，面向会议交付的累计准备度评估为 **40%～50%（当前加权检查点 46/100）**；M-09 的正式阶段验收为 **1/6**。两个数字回答不同问题：46/100 计入 T-05、R-02、J-02A/J-02B 的既有框架、服务器训练与归档；1/6 只统计按本轮会议标准重新验收通过的阶段。

| 已形成的基础 | 为什么计入进度 | 为什么还不能算会议交付完成 |
|---|---|---|
| GPPO 基础任务分配、合法动作与执行安全链 | 已有可运行代码、checkpoint 和测试 | 尚未按本轮六场景 60 条轨迹重新验收 |
| T-05 世界模型接入和四组服务器训练 | 证明训练链、Shadow、adapter 和回退可运行 | 没有证明稳定策略增益，也不是会议所述五类型版本 |
| J-02A/J-02B 数据合同和服务器训练 | 证明防未来泄漏、合法分支和完整训练归档 | J-02B 开发门槛失败，不能直接进入后续策略扩展 |
| S0 版本与术语核查 | 已锁定可复用三类型基线并明确缺口 | 五类型实现、四类扰动完整闭环仍未找到/未验收 |

当前真正阻塞中旬演示的是：会议所述五类型源码与模型缺失；紧急任务、能量/换电和复合扰动没有完整可控实现；“慢 3～4 倍”缺少同硬件、同负载的公平复测；汇报稿和演示包尚未冻结。完整的计算方法、证据链、风险与后续顺序见 [会议准备度说明](nodes/M-09/readiness.md)，机器可读评估见 [readiness.json](nodes/M-09/readiness.json)。

## 为什么需要世界模型

新增独立路线：[J-01 Graph-JEPA 实验](nodes/J-01/README.md)已完成三组 × 三 seed 离线训练，接入门槛未通过，未启动 JEPA-GPPO 训练。它与原 EAWM 并列，不替换 GPPO。[完整结果](nodes/J-01/evidence/final-report.md) / [下一版草案](nodes/J-02/README.md)。

当前 GPPO 能读取实时 belief 图，在 16 条 UAV–Region 候选边和一个 NOOP 中自主选择合法动作，并通过 graph/action version、ACK、lease 和 fencing 保证执行安全。

它的主要局限是：决策以当前图为主，没有一个经过训练的内部模型显式表示“这个动作执行以后可能发生什么”。世界模型要补充的正是这一层能力：

- 从多步可见历史中保留系统变化信息；
- 区分不同 executed action 导致的不同后果；
- 预测下一图/状态差分、reward、cost、continuation 和不确定度；
- 通过自动事件监督，让 latent 更关注关键状态变化；
- 在确认安全、校准和兼容性后，把冻结 latent 提供给 GPPO。

世界模型不是新的动作控制器，也不直接向环境下达命令。

## 最终系统是什么

```text
历史可见 belief 异构图 G_t
+ decision_time 前已到达的 evidence/message
+ 实际确认执行的动作 a_t
+ 时间、版本和有效性 mask
                 │
                 ▼
       动作条件异构图世界模型
  Graph encoder + action encoder + dynamics
                 │
        temporal latent [h_t, z_t]
      ┌──────────┼──────────┬──────────┬──────────┐
      ▼          ▼          ▼          ▼          ▼
 下一图/差分  自动事件   reward/cost continuation uncertainty
                 │
                 ▼
         可关闭的 frozen latent adapter
                 │
                 ▼
               GPPO
                 │
                 ▼
  真实 action mask + version + ACK/lease/fencing
```

### 世界模型输入

- 决策时刻可见的 UAV、Region、Target 异构图；
- `decision_time` 前已经收到的证据和消息；
- 实际执行并确认的 UAV–Region 动作或 NOOP；
- graph/action version、当前 `decision_time` 和有效性/padding mask；动作后才知道的下一决策时间差禁止作为输入。

### 世界模型输出

- 时序 latent `[h_t, z_t]`；
- 下一图或节点/边状态差分；
- ordinal、nominal、structural、evidence 自动事件概率；
- reward、cost vector 和 continuation 预测；
- epistemic/aleatoric uncertainty；
- model/input version 和 `valid` 状态。

### GPPO 如何使用输出

T-05 主实验只把冻结 latent 经过可选 adapter 加入 actor/critic。自动事件 logits 默认不直接进入 actor。模型异常、超时、版本不一致或不确定度过高时，系统使用 zero context，恢复原始 no-WM GPPO 路径。

## GPPO 与世界模型的职责边界

| 能力 | GPPO/现有安全链 | 世界模型 |
|---|---|---|
| 在合法候选动作中选择具体动作 | 唯一负责 | 禁止直接选择或提交 |
| 定义 16 条候选边和 NOOP | 权威合同 | 只读取 |
| 维护真实 action mask | 权威状态 | 禁止写入 |
| 维护 belief 和 graph/action version | 权威状态 | 禁止写入 |
| ACK、lease、fencing 和 stale 拦截 | 权威执行链 | 禁止绕过 |
| 预测动作后的状态、事件和成本 | 不负责 | 负责 |
| 输出预测 latent 和不确定度 | 不负责 | 负责 |
| 模型故障时继续运行 | 原 GPPO 路径 | 必须允许无损关闭 |

## T-00～T-06 每一步的意义

| 节点 | 要解决的问题 | 主要实现/产物 | 通过以后意味着什么 | 当前状态 |
|---|---|---|---|---|
| [T-00](nodes/T-00/README.md) | 模型到底能读取什么，怎样保证不偷看未来 | 因果 Transition schema、字段注册表、future/truth denylist、统一 recorder、基线测试 | 输入输出和安全边界已经冻结，可以可信采集数据 | **passed** |
| [T-01](nodes/T-01/README.md) | 世界模型用什么真实轨迹训练，怎样防止 train/test 泄漏 | random legal、greedy、GPPO 三类轨迹；完整 episode/tape/seed split；数据/策略 checkpoint 和 SHA-256 | 已有可复现、可审计的数据，可以开始训练世界模型 | **passed** |
| [T-02](nodes/T-02/README.md) | 不考虑事件监督时，模型能否学到“动作导致的后果” | Graph encoder、action encoder、temporal dynamics、next-state/reward/cost/continuation/uncertainty heads | 得到第一个真实 Graph-WM checkpoint，并用合法动作反事实证明模型使用动作 | **passed** |
| [T-03](nodes/T-03/README.md) | 自动事件和 GES 是否让 latent 更关注关键变化 | 自动事件生成器、按模态 Event Heads、hard/smooth GES、WM/EA-noGES/EAWM 消融 | 得到事件感知世界模型，并能分离 Event Head 与 GES 的贡献 | **passed** |
| [T-04](nodes/T-04/README.md) | 模型在线运行是否可信、校准、及时且不污染系统 | 只读 Shadow runtime、ID/OOD 校准、risk-coverage、延迟和安全回退报告 | 世界模型可以在线观察和预测，但仍不影响正式动作 | **passed** |
| [T-05](nodes/T-05/README.md) | 世界模型 latent 对 GPPO 是否有真实增量价值 | frozen latent adapter、zero-context fallback、旧 checkpoint 兼容、四组公平实验 | 完成世界模型基础迁移，并以负结果严谨判断当前版本未带来稳定增益 | **passed** |
| [T-06](nodes/T-06/README.md) | 短期 imagined rollout 是否有额外价值 | GPPO 合法候选动作的 1～3 步 rollout、不确定度截断、真实环境验证 | 可选的预测规划扩展；失败时保留 T-05，不影响基础迁移 | planned / optional |

### T-00：冻结合同，而不是先写网络

意义是防止后面训练出一个“指标很好但偷看未来”的模型。本节点把在线可见字段、未来 target、proposal 与 executed action、版本语义和统一记录方式明确分开。

已验证：原 GPPO 核心/训练/并发/事件桥接 50 项测试通过；本仓库因果合同、动作合法性、版本和 recorder 测试通过。证据见 [T-00 节点](nodes/T-00/README.md)。

### T-01：建立可训练、不可泄漏的数据

意义是让世界模型同时看到不同策略和不同压力场景，而不是只学习单一 GPPO 的窄分布。数据按完整 `scenario/tape/seed` 分组切分，同一个事件带绝不能跨 train/validation/test。

当前已封存：

- 126 个 episode、502 条 transition；
- random legal、greedy、GPPO 三类行为策略；
- normal、single、sequential、overlap、burst、long gap、weak communication；
- 17 个动作全部覆盖；
- split overlap 为 0；
- 在线 truth-only 字段为 0。

数据和采集用 checkpoint 位于 [T-01 Release v0.1.0](https://github.com/Battleplus/GPPO-WORLD-9.2/releases/tag/t01-data-v0.1.0)。其中 512-step GPPO 只用于扩大数据覆盖，不是历史 50k 正式模型，也不是世界模型。

### T-02：先证明基础世界模型真的理解动作

第一版只训练动作条件 Graph-WM，不加入 Event Predictor/GES，也不修改 GPPO。核心验证不是单纯看 loss，而是进行：

- 正确 action；
- action shuffle；
- no-action；
- last-value/frequency；
- summary-vector GRU；
- Graph World Model。

当前 T-02 已通过并封存于 [Release v0.1.0](https://github.com/Battleplus/GPPO-WORLD-9.2/releases/tag/t02-base-wm-v0.1.0)。
合法动作反事实的 state/reward/cost 均显著退化，checkpoint 可独立恢复；同时透明保留了 Flat-GRU
状态误差更低、no-action state CI 跨 0 的负结果。T-02 只证明基础模型链路成立，不声称下游 GPPO 增益。

### T-03：迁移 EAWM 的事件感知思想

自动事件来自相邻可见图变化，而不是人工指定“发生 UAV_DAMAGE 就选某动作”。事件分为：

- ordinal：连续量 DOWN/SAME/UP；
- nominal：类别 SAME/CHANGED；
- structural：节点、关系、候选边和合法 support 变化；
- evidence：新证据、重复、冲突、确认和过期。

Event Predictor 是世界模型辅助头，GES 用于调节高事件密度边界对训练的影响。它们服务于 latent 表示学习，不替代 GPPO，也不等同于人类偏好学习。

当前 T-03 已完成三 seed 固定预算消融并封存于 [T-03 Release v0.1.0](https://github.com/Battleplus/GPPO-WORLD-9.2/releases/tag/t03-eawm-v0.1.0)。
EAWM-hard 的 macro-F1 为 `0.4668±0.0050`、macro-AUPRC 为 `0.4320±0.0136`；基础 state/reward/cost
预测的最大逐 seed 相对退化均低于冻结的 5% 上限。TTL 缺失、失败配置和测试集已查看的协议限制均已显式保留，
详见 [T-03 节点证据](nodes/T-03/README.md)。

### T-04：先 Shadow，再允许策略读取

世界模型在线维护 latent、记录预测和实际结果，但不修改动作。只有以下门禁全部通过才进入 T-05：

- belief/action mask/version 写入次数为 0；
- stale hidden state 不提交；
- 异常、超时、OOD 高风险能够回退；
- ECE、Brier、risk-coverage 达到冻结标准；
- P50/P95/P99 延迟满足预算。

T-04 已通过并封存于 [T-04 Release v0.1.0](https://github.com/Battleplus/GPPO-WORLD-9.2/releases/tag/t04-shadow-v0.1.0)。
真实基线环境的 belief、action mask、graph/action versions 及动作提交接口均保持零写入；完整 observe 的 P95/P99
为 `6.99/8.05 ms`。合成 OOD 的范围和 8.59% ID 假阳性率已透明记录，不作生产级 OOD 泛化声明。

### T-05：冻结 latent 接入 GPPO

保持动作空间、mask、奖励、PPO 预算、场景和 seed 一致，至少比较：

1. GPPO；
2. WM-GPPO；
3. EA-noGES-GPPO；
4. EAWM-GPPO。

必要时增加 GPPO-History，以排除“只是多看历史”的解释。只有真实 held-out 环境、多个 seed 和安全指标共同支持，才能声称世界模型对 GPPO 有增益。

T-05 已完成：冻结 `[h,z]` residual adapter、post-action Shadow hook、逐 transition versioned latent sidecar、旧 checkpoint 无损回退，以及四组 × 3 seeds × 50k 的正式 GPU 训练。固定 50k checkpoints 全部在同一有序 100-tape Test bank 上评估；12/12 runs、24 checkpoints、12/12 evaluations 和 1,200 traces 均完成哈希复核。真实环境/belief/mask/version/动作提交写入为 0，世界模型冻结且延迟 Gate 全部通过。结果没有证明稳定性能增益，详见 [最终报告](nodes/T-05/evidence/final-report.md)。

### T-06：可选想象规划

T-06 不属于当前基础迁移完成条件。它只允许对 GPPO 提出的合法候选动作做 1～3 步短期 rollout，并根据不确定度截断。若只改善 predicted return、没有改善真实 held-out 结果，本节点应标记失败并回退 T-05。

## 最终会交付什么

完成 T-00～T-05 后，仓库应当包含：

1. **可运行的世界模型代码**：图编码器、动作编码器、时序 dynamics、预测 heads、loss、训练和评估入口；
2. **真实 checkpoint**：基础 Graph-WM、EA-noGES、EAWM+GES，附带不可变下载链接和 SHA-256；
3. **自动事件系统**：事件生成、模态注册、Event Predictor、GES 和相应测试；
4. **只读 Shadow runtime**：校准、OOD、延迟、版本一致性和故障回退；
5. **GPPO latent adapter**：可配置关闭、zero-context parity、旧 GPPO checkpoint 兼容；
6. **公平实验报告**：逐 seed 结果、置信区间、失败 run、业务指标、安全指标和资源成本；
7. **完整证据链**：代码 commit、配置、数据 manifest、split hash、seed、checkpoint、日志、指标和节点结论。

## 什么才算“迁移完成”

以下条件必须全部满足：

- 动作条件异构图世界模型与严格数据切分真实实现；
- state/event/reward-cost/continuation/calibration 均有独立评估；
- GPPO 可以读取冻结 latent，并能无损关闭世界模型；
- belief、action mask、version、ACK/lease/fencing 不受污染；
- 至少完成 GPPO、WM-GPPO、EA-noGES-GPPO、EAWM-GPPO 四组公平消融；
- 每项结论都能追溯到真实 checkpoint、配置、seed、日志和源码提交。

计划文件、未保存的本地模型、单个最佳 seed 或模型内部 predicted return 都不能作为“完成”的依据。

## 明确不做什么

- 不让世界模型取代 GPPO；
- 不让预测图覆盖真实 belief；
- 不绕过 action mask、版本、ACK、lease 或 fencing；
- 不把 event logits 默认直接输入 actor；
- 不声称能准确预测不可观测的外生随机事件；
- 不在单步预测和校准未通过前开展长时域规划；
- 不把自动事件监督包装成人类偏好学习；
- 不在没有实际测量时声称降低延迟或计算量。

## 仓库结构

```text
GPPO-WORLD-9.2/
├── gppo_world/            # 合同、数据、世界模型与后续运行模块
├── tools/                 # 采集、审计、训练和评估入口
├── tests/                 # 因果性、模型、安全和兼容性测试
├── docs/                  # 总体设计、架构、执行与验收规范
├── nodes/T-00...T-06/    # 每个节点的状态、Gate 和真实证据
├── nodes/status.json      # 机器可读的权威节点状态
└── README.md              # 项目总入口
```

## 文档阅读顺序

1. [范围、分工与安全边界](docs/00-scope-and-boundaries.md)
2. [架构与数据合同](docs/01-architecture-and-contracts.md)
3. [T-00～T-06 执行规划](docs/02-execution-plan.md)
4. [节点、checkpoint 与证据保存规范](docs/03-checkpoint-and-evidence-policy.md)
5. [实验矩阵与验收定义](docs/04-experiment-and-acceptance.md)
6. [当前任务进度](docs/05-current-progress.md)
7. [T-05 服务器训练 AI 接力说明](docs/06-server-ai-handoff.md)
8. [T-05 正式服务器 Campaign 实时接力存档](docs/07-t05-live-server-campaign-handoff.md)
9. [节点总索引](nodes/README.md)
10. [T-05 封存后的诊断与下一步](docs/08-post-t05-diagnostics-and-next-plan.md)

节点的权威当前状态以 [`nodes/status.json`](nodes/status.json) 为准。README 负责解释路线，节点证据负责证明结果；文档中的计划不能替代真实实验。

## 快速验证

在安装 PyTorch、NumPy 和 pytest 的 Python 环境中运行：

```powershell
python -m pytest -q
```

T-01 数据、采集用 GPPO checkpoint 和原始 manifest：

- [Release 页面](https://github.com/Battleplus/GPPO-WORLD-9.2/releases/tag/t01-data-v0.1.0)
- [T-01 节点证据](nodes/T-01/README.md)

T-02 基础世界模型 checkpoint、训练日志与指标：

- [T-02 Release v0.1.0](https://github.com/Battleplus/GPPO-WORLD-9.2/releases/tag/t02-base-wm-v0.1.0)
- [T-02 节点证据](nodes/T-02/README.md)

T-03 事件感知模型、逐 seed 消融、失败 run 与指标：

- [T-03 Release v0.1.0](https://github.com/Battleplus/GPPO-WORLD-9.2/releases/tag/t03-eawm-v0.1.0)
- [T-03 节点证据](nodes/T-03/README.md)

T-04 Shadow、校准、真实基线零写入审计与回退记录：

- [T-04 Release v0.1.0](https://github.com/Battleplus/GPPO-WORLD-9.2/releases/tag/t04-shadow-v0.1.0)
- [T-04 节点证据](nodes/T-04/README.md)

## 方法来源

- [GPPO-8.29](https://github.com/Battleplus/GPPO-8.29)
- [世界模型任务目标与改进目标](https://github.com/Battleplus/GPPO-8.29/blob/2a9bb9f87b9d543df144f4d108ba970c924151f9/docs/world-model/current/%E4%B8%96%E7%95%8C%E6%A8%A1%E5%9E%8B%E4%BB%BB%E5%8A%A1%E7%9B%AE%E6%A0%87%E4%B8%8E%E6%94%B9%E8%BF%9B%E7%9B%AE%E6%A0%87.md)
- [EAWM 官方实现](https://github.com/MarquisDarwin/EAWM)

## 当前能力声明

T-00～T-05 已有封存证据，世界模型基础迁移完成。T-05 正式消融、checkpoint/日志/指标、兼容与安全 Gate 均已通过；仓库保留失败尝试和负结果。当前不支持“世界模型稳定提升 GPPO”的声明，T-06 仍是未开始的可选研究项。
