# 联合任务集合后果开发 pilot：启动前登记

登记时间：2026-09-14（Asia/Shanghai）

## 运行边界

- 源码分支：`joint-consequence-design-20260914`；启动前以实际 Git HEAD 为准。
- 协议：`world-gppo-9.11-joint-consequence/0.1.0`，父协议为 arrival-to-region；研发主指标为 physical arrival。
- 通知：原 `single_shot` 配置；不启用 bounded retry。
- 续行：首步为候选动作，之后固定 `traditional_public_controller_v1`；不使用隐藏状态或事后最优动作。
- 数据：train 8、validation 4 个父 episode；每父 4 个前缀；每前缀最多25个合法候选；每候选3个外生重复；test/OOD 为0。
- 预算：单 seed `1101`；训练最多512次实际 optimizer update、4 epochs、validation patience=2、20分钟墙钟。pilot、正式数据、策略融合和通知实验不合并记账。

## 查看数据后的固定判断规则

以下规则在数据生成完成后不得按结果修改：

1. 每个 split 至少保留一个父 episode；父 episode 的所有前缀、候选和重复只能属于该 split。
2. 候选记录必须通过 Graph-5/25-action 合法性、固定任务集合一致性、重复/动作唯一性和 manifest SHA-256 审计。
3. 只有截止结局已观察的任务参与联合计数；删失任务通过 mask 排除，不转成失败。
4. “可比较前缀”定义为同一父/前缀/重复内至少两个合法候选，且联合聚合标签均未删失；不因结果好坏筛选。
5. “存在初步可重复差异”定义为至少一个可比较前缀中候选标签差异非零，并且该差异在三次重复中不是仅由单次分支的幸运结果解释；重复波动和差异必须分别报告。该小样本规则只决定是否进入可行性训练，不构成统计显著性证明。
6. 若全部联合标签被 mask、没有可比较前缀、父/前缀泄漏、隐藏/未来字段进入输入，或共享随机条件无法由账本支持，则停止参数更新。

## 资源记录

启动前环境：专用 venv `E:/Z博士/9.2日/world-gppo-911-local-pilot-venv`；Python 3.11.15；Torch 2.7.0+cu128；CUDA 12.8；GPU RTX 3060 Laptop 6 GiB；固定单进程、数据 workers=0。系统 Python 的 CPU-only Torch 不用于本 pilot。启动时记录 GPU/内存占用、线程数和运行时版本。

