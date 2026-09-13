# M-10 通知方案独立验证预注册

登记时间：2026-09-14  
登记时源码 HEAD：`721eb77caa1a9549fc6e1529c21e5528fd1907c0`  
目的：在不追加机制、不训练、不运行已否决调度变体的前提下，验证原系统与原调度器 + `bounded_retry` 通知的效果—成本权衡。

## 冻结范围

- 变体：`original`（原调度器、单次通知）与 `notification`（原调度器、稳定通知 ID、ACK、固定间隔有界重试）。不运行 `arrival-slack` 或 `combined`。
- 协议：`world-gppo-9.11-arrival/0.1.0`；任务完成为 `arrival_to_region`；主口径 `physical_arrival`，`host_confirmation` 独立报告。
- 生成器：`weak_communication_tape(split="test", count=16, base_seed=193001, name="mixed")`；每个父 tape 6 个任务，共16个父 tape、96个任务；ideal/composite各运行两个变体，共64 episodes。
- 新父 tape 身份：`test-mixed-seed-2193001` 至 `test-mixed-seed-2193016`。开发回归集为 `test-mixed-seed-2093001` 至 `test-mixed-seed-2093016`，不得合并统计。
- 两变体共享相同场景对象、任务、动作调度器、执行门禁、ACK/lease/fencing/能量合同和每个消息身份确定的通信 profile。每个变体单独建立环境，但不使用全局顺序 RNG 来重排通信故障。
- 通知参数冻结：retry interval 1.0 s、最多2次 retry、retention 3.0 s；通知使用独立逻辑 completion_notice/completion_ack 链路，所有尝试、丢弃、ACK和序列化代理字节计账。该逻辑链路不解释为免费物理信道。

## 查看结果前的判断规则

本轮是权衡验证，不因缺少业务成本门槛宣布正式通过。主报告分别给出按时物理到达、按时主机确认、最终确认、终态覆盖、安全、通信代理字节和完整决策延迟，并以父 tape 配对；候选任务分支不当作独立 episode。

- 若确认改善伴随物理到达下降或通信/计算成本增加，保留权衡结论，不自动采用为默认；
- 若两项差异很小或父 tape 样本不足，写证据不足；不扩样、不按结果改参数；
- 安全违规为必要门槛，但不能替代任务效果；不放宽 deadline、安全门禁或消息身份校验；
- 旧开发回归结果只作历史对照，不与本轮合并为独立收益。

## 冻结输入哈希

| 输入 | SHA-256 |
|---|---|
| `gppo_world/m10_environment.py` | `45f1695e9d9e7054bdc1bc1215b9cc1f15cb1a3362941d8a7519919936afc5b4` |
| `gppo_world/task_policy_view.py` | `d3eb71c35261fc89f6812a041b3fbc5b2d8629f6dda1cd606340305475fe8575` |
| `tools/run_m10_targeted_improvements.py` | `79d5b245d50fb149a128e2b62098e07bfe36d9bdf7ec86faa7b8a1cee669968b` |
| `tools/run_m10_arrival_system_closure.py` | `604a1915aba68ab50d7b7feab9c01673091e9bb950a0d0e30fbf740d55fe0b50` |
| `tools/run_m10_arrival_rule_baseline.py` | `6605d55acf43aabf2876b44511040a308060540eecdb0532739efb7e63c3bec1` |
| `configs/world-gppo-9.11-arrival-v0.1.0.json` | `2ce326bda92670203792faf97eaca36ef2dbfdff9f287de963011826d7ad8d22` |

该登记文件和新运行入口属于验证记录；运行开始后不修改上述机制或判断规则。
