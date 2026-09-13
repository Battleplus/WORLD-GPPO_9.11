# M-10 弱通信调度与完成通知针对性改进

状态：已完成一次有界配对验证；本轮不训练、不重跑后果模型评估、不连接服务器。原规则结果复用已核验的 `m10-system-closure-paired-20260914-v1`，只运行三个有实现变更的新变体。

## 1. 范围与冻结参数

协议为 `world-gppo-9.11-arrival/0.1.0`，主口径为物理到达；主机确认独立统计。所有变体使用相同16条 test tape、96个任务和同一 ideal/composite 通信 profile。

- 原规则：单次完成通知、原 urgency-distance 规则；不重复运行，复用旧 ledger。
- notification：可靠通知版本；稳定 `completion_notice_id`，独立 completion-notice 与 completion-ACK 链路，重试间隔1.0秒，最多2次重试，保留3.0秒。
- scheduling：只用合法公开信息，按 `deadline - now - public_distance_estimate` 最小余量选择，仍每周期最多一个新分配。
- combined：同时启用 notification 和 scheduling。

重试、过期、重复、身份和 fencing 校验均保留；没有使用可靠旁路、隐藏真值、放宽 deadline 或安全门禁。完整新 ledger：

`E:\Z博士\9.2日\WORLD-GPPO_9.11-local-runs\m10-targeted-improvements-full-20260914-v2\targeted-improvements.json`

大小 159,383,244 bytes，SHA-256：`6b9ac4d5365224fc08584ae5e9e2607337440502aefc815bba4d788cc525ba9a`。

## 2. 原始结果比较

| 变体 | 条件 | 按时物理到达 | 按时主机确认 | 物理到达事件 | 主机确认事件 | 观察到安全违规 | 平均返回 |
|---|---|---:|---:|---:|---:|---:|---:|
| original（复用） | ideal | 96/96 | 95/96 | 96 | 96 | 0 | 59.5168 |
| notification | ideal | 96/96 | 95/96 | 96 | 96 | 0 | 59.5168 |
| scheduling | ideal | 96/96 | 96/96 | 96 | 96 | 0 | 59.4313 |
| combined | ideal | 96/96 | 96/96 | 96 | 96 | 0 | 59.4313 |
| original（复用） | composite | 48/96 | 9/96 | 48 | 10 | 0 | 16.4983 |
| notification | composite | 46/96 | 21/96 | 46 | 29 | 0 | 14.7153 |
| scheduling | composite | 32/96 | 6/96 | 32 | 8 | 0 | 2.4016 |
| combined | composite | 31/96 | 14/96 | 31 | 23 | 0 | 1.5637 |

ideal 条件没有出现物理到达退化；弱通信条件下，通知机制提高按时确认但伴随2个物理到达损失，调度启发式则显著损害任务效果。因此本轮不采用 scheduling 或 combined 作为系统默认方案。

## 3. 通知路径定位与开销

旧 original ledger 中26个物理到达任务的 `completion_message_id` 为 null，但旧丢弃记录没有 `message_kind`，无法仅凭旧日志严格区分“生成后丢弃”和“生成记录缺失”。源码表明物理到达时会先创建 completion record，再调用发送函数；但该旧运行没有足够字段完成逐条拆分，故保留为 `generated_or_dropped_unresolved`，不追溯编造精确数。

notification 新 ledger 已把路径显式记录。composite 中46个物理到达任务产生92次通知传输尝试：初次尝试46次（23 dropped、23 sent），重试46次（其中28次 sent、18次 dropped）；最终 completion-notice 状态为 sent 51、dropped 41、received 29、expired 17、duplicate_ignored 4、identity rejection 1。completion-ACK 为 received 30、dropped 3。按时主机确认相对 original 为 `+12`（21 对9），逐任务差异为15个救回、3个新增损失；物理到达为 `-2`（46 对48），没有物理任务被“通知重试”直接救回。

新增独立 completion-notice/completion-ACK 的 composite 序列化代理字节量为 38,082 bytes；notification 全通信代理字节量为 6,354,451，相比 original ledger 计算的5,899,716增加454,735 bytes。后一个差值还包含通知到达时序导致的可见状态和控制流变化，不能全部归因于通知载荷；代理字节不是实网流量。

5个旧运行中的 stale/duplicate 路径未发现已成功处理的同一有效通知；新机制将完成通知从周期性 `task/pending` 序列通道分离，并用稳定通知 ID 做幂等处理。新增身份拒绝路径不算安全违规，且不接受撤销/替换执行身份。

## 4. 调度选择定位

原弱通信 ledger 的20个 `scheduling_no_selection` 任务全部有非空 `first_legal_candidate_time`，说明当时至少存在合法公开候选；其中首次合法时刻的行为为：19次选择了其他任务，1次 NOOP；14次其他任务命令实际 accepted，3次 task_unavailable，1次 resource_unavailable，1次 resource_busy，1次 NOOP。该证据支持“单候选/排序与待办队列竞争”是可检查的调度问题，但不证明这些任务在所有控制器下必然可行。

arrival-slack 启发式在 composite 下相对 original 的逐任务差异为：9个物理到达救回、25个物理到达损失，净变化 `-16`；按时主机确认1个救回、4个损失，净变化 `-3`。combined 为9个物理到达救回、26个损失，净变化 `-17`；按时主机确认10个救回、5个损失，净变化 `+5`。因此该单一修正方向被否定，不继续调参或扩大验证。

调度变体没有新增安全违规；资源拒绝、过期和 stale 等正常门禁路径仍与安全违规分开统计。剩余失败仍同时包含命令丢失、资源/执行门禁和时间预算，不能统称为排序缺陷。

## 5. 结论与边界

1. 可确认的实现限制：旧完成通知日志粒度不足；完成通知与 `task/pending` 共享序列通道造成5条可观测 stale/duplicate 拒绝；这两点已分别修复为新版本的显式记录、独立通道、稳定 ID 和幂等接收。
2. 通信/资源限制：弱通信仍造成命令和遥测丢失、资源不可用、租约/执行门禁和时间不足；这些不自动等同于代码缺陷。
3. 通知改进有确认收益但有通信开销和2个物理到达副作用，不能作为无条件默认改进。
4. 调度改进没有额外物理到达价值，且产生明显副作用；本轮不采用。
5. 剩余问题不能仅凭本轮证据归因于需要学习预测。首先仍需解决公开信息可用性、资源/租约能力和完成通知合同；不因该结果恢复世界模型训练。

本轮支持的结论是有限仿真条件下的改进边界，不是生产或竞赛可用性通过。原始结果、旧 Release、负结果和旧 ledger 限制均保留。

