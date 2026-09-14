# M-10 结果口径表

| 字段 | 定义 | 本轮处理 |
|---|---|---|
| 预定单元 | 16 父 tape × 4 prefix × 2 条件 = 128 | 全部保留选择记录 |
| 适用 | prefix 时公开任务集合 `J` 非空 | ideal 52、weak 45；动作前确定，不受方法结果影响 |
| 不适用 | prefix 时没有公开 pending 任务集合 | ideal 12、weak 19；不算失败/删失 |
| 分支有效 | `J` 中所有任务 deadline 结局已观察 | 各方法分别统计 |
| 分支删失 | 至少一个 `J` 任务窗口内未观察到 deadline 结局 | 不当作失败，不静默删除 |
| 缺失 | 适用单元预期的已选方法分支没有 ledger | 本轮为 0 |
| 主效果 | 固定 `J` 内按时物理到达任务数 | 只在有效分支计算 |
| 配对差 | 同父、prefix、条件、repeat、`J` 的 model 减参考方法 | 共同有效时计算 |
| 父级均值 | 先在父 tape 内平均配对差，再对父 tape 等权平均 | 避免 prefix/repeat 伪独立 |
| CI | 父级均值的非参数 bootstrap，seed 1101、10000 次、percentile 95% | weak model−public：14 父 tape，[0,0.2832] |
| 保守界 | 删失 on-time count 取 `0..|J|` 的简单范围 | weak model−public：`[-0.9861,+1.2306]`，仅适用单元 |
| communication_count | 模拟通信 ledger 记录数 | 不是网络包数 |
| communication_proxy_bytes | JSON 序列化代理字节量 | 不是真实网络流量 |
| decision_wall_ms | 分支仿真与记录耗时 | 不是部署推理延迟 |
| 安全违规 | 已记录的重复/越权/fencing 等违规计数 | 本轮 3104 分支为 0；不替代生产保证 |
| 通信指标 | host confirmation 等独立敏感性结果 | 不改写 physical arrival 主指标 |
