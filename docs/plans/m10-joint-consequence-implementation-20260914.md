# 联合后果阶段实现与 pilot 登记

本阶段目标是把“候选动作对任务集合联合后果的预测”变成可审计、可停止的实验入口。当前只提交协议、schema、生成器、基线和 fixture 测试；不生成正式数据，不训练，不重跑通知/调度实验。

## 已实现

- `gppo_world/joint_consequence.py`：固定任务集合、逐任务结局、删失 mask、候选—参考配对差和父/前缀/重复/动作身份校验。
- `gppo_world/joint_consequence_baseline.py`：只读取 Graph-5 公共快照的联合机会成本评分，保留候选差异，不访问仿真真值。
- `tools/generate_m10_joint_consequence_dataset.py`：基于现有 M10 反事实环境的显式适配器。默认生成器参数是开发 pilot，输出目录非空即拒绝；运行本身不属于本提交动作。
- `schemas/m10-joint-consequence-v1.schema.json`：记录字段合同。
- `gppo_world/joint_consequence_data.py`、`gppo_world/joint_consequence_model.py`：新 schema loader、候选逐行动作轻量回归模型和有效标签 mask loss。
- `tools/train_m10_joint_consequence_model.py`：独立的 best/last、RNG/数据顺序/optimizer/早停状态保存与显式有界恢复入口。

## 登记的开发 pilot 命令

以下命令仅是冻结后的可执行命令，本轮没有运行：

```powershell
python tools/generate_m10_joint_consequence_dataset.py `
  --out runs/m10-joint-consequence-pilot-20260914-v1 `
  --train-count 8 --validation-count 4 --test-count 0 --ood-count 0 `
  --base-seed 193001 --prefix-steps 1 2 3 4 `
  --horizon-steps 6 --max-candidates-per-prefix 25 --exogenous-repeats 3
```

预期数据规模上限为 12 个父 episode、每父 4 个前缀、每前缀最多 25 个候选、每候选 3 个重复；实际规模由公开任务集合和合法动作 mask 决定，跳过原因写入 manifest。该命令不产生 test/OOD 盲测。

## 后续模型 pilot（条件性登记，已实现入口，未运行）

模型训练消费本协议的新 schema，不能调用旧单候选到达时间训练器。冻结后使用 seed `1101`、最多 512 updates、4 epochs、patience 2、20 分钟墙钟，train/validation 选模，test/OOD 只在冻结后一次性评估。策略融合不在本阶段自动启动。

## 证据顺序

先审计有效标签率、删失率、同前缀候选差异、重复间波动和联合公开基线；再决定是否值得训练。预测误差、离线排序、固定续行的选中结果、在线策略增益和通信/计算成本分别报告。
