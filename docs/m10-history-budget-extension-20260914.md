# M-10 基础 History 策略预算充分性验证

日期：2026-09-14  
分支：`history-budget-sufficiency-20260914`  
源码提交：`caf2b2bb4e8e8bcf7d1c75214e4f5a47bd6de1c9`

## 范围与环境

本轮只延长原 GPPO-History（H）训练：从已有累计 8192 步续跑到累计 32768 步。没有训练 E、辅助任务、世界模型、候选 prior、PreCo 或事件触发，也没有修改奖励、任务语义、通信参数或执行门禁。

设备为本机 NVIDIA GeForce RTX 3060 Laptop GPU（驱动 571.96，6 GiB）；Python 3.11.15，PyTorch 2.7.0+cu128（CUDA 12.8），NumPy 1.26.0，Windows 11 build 26100。单训练进程；runner 在 CUDA 模式下不调用 CPU 线程限制，线程字段按实际记录为 `null`。评估计时未预热（`warmup_steps=0`），不含环境推进和磁盘写入。

## 恢复与实际预算

三个 8192 步源 checkpoint 均通过恢复预检：格式为 `gppo-history-arrival-training-v1`，包含模型、optimizer、`environment_steps=8192`、`optimizer_updates=128`、Python/NumPy/Torch/CUDA RNG、PPO 配置和数据顺序。恢复边界是完整 rollout batch；runner 每个 batch 重新建立环境和 History，不需要保存 batch 内环境快照。原配置保持 learning rate `3e-4`、clip `0.2`、rollout `256`、update epochs `4`、minibatch `256` 等不变。

每个 seed 实际新增 24576 环境步、384 次 optimizer step（96 条更新记录，每条 4 次 `optimizer.step()`），累计 32768 步、512 次更新，均因环境步预算停止。三组新训练 rollout ledger 各 24576 条记录，未把旧 8192 记录改写成新训练。

需要如实保留一个制品问题：完成后的无操作 `--resume` 检查曾将标记为 16384/24576 的阶段文件覆盖为最终 32768 状态。因此当前 16384/24576 文件虽存在，但其内部元数据均为 32768，不能作为中间 checkpoint 证据；本轮没有为了修复该问题重跑训练。最终 32768 checkpoint、8192 源 checkpoint、更新记录、训练 ledger 和最终测试结果仍可独立校验。runner 已在本提交中增加“已有阶段不覆盖、阶段不一致则 fail-closed”的保护。

## 新 final-test

训练前生成并冻结 64 个全新同分布 weak-composite 父 tape，每个 6 个任务，共 384 个任务；不读入训练或 validation，和旧 tape ID 重叠数为 0。生成/身份信息见 `tape-manifest.json`，SHA-256 为 `a998177c2e8293a39389eb2a4b26f6da0e2aedeb46bdbb759157553ebd8a7435`；其中冻结 tape 文件 SHA-256 为 `3fc1a9e667739f9966931318486b32305a0a3da8863b43f22dc13eb4a97d30aa`。

主指标是按时 physical arrival；所有六个学习评估均为 384/384 任务已明确结局，`unresolved=0`，安全违规为 0。规则只在同一 64-tape 测试集运行一次，不复制成三个 seed。

| seed | H 8192 physical | H 32768 physical | 差值 | H 8192 host | H 32768 host | H 32768 延迟 mean / P95 / P99 ms |
|---|---:|---:|---:|---:|---:|---:|
| 1101 | 172/384 (44.79%) | 182/384 (47.40%) | +10 (+2.60pp) | 33 | 38 | 2.6671 / 3.8841 / 4.4795 |
| 2203 | 185/384 (48.18%) | 193/384 (50.26%) | +8 (+2.08pp) | 33 | 36 | 2.7766 / 4.2834 / 5.6510 |
| 3307 | 165/384 (42.97%) | 181/384 (47.14%) | +16 (+4.17pp) | 23 | 30 | 2.6868 / 4.0489 / 4.6139 |
| 合计（任务仅作描述） | 522/1152 (45.31%) | 556/1152 (48.26%) | +34 (+2.95pp) | 89 | 104 | — |

三 seed 均改善，但平均增幅为 2.95 个百分点，略低于预登记的 3 个百分点门槛。新测试集上的共享规则为 192/384（50.00%）按时物理到达、43/384 按时主机确认；规则不是三个独立训练 seed。32768 步 H 的任务完成率点估计仍低于规则，不能写成超过传统方法。规则汇总文件没有提供与 H 同口径的 deadline、安全和拒绝字段，未从缺失字段推算这些数值。

H 的 deadline failures（与 384 任务和 `unresolved=0` 一致的评估汇总）分别为：8192 步 212/199/219，32768 步 202/191/203（seed 顺序 1101/2203/3307）。正确拒绝分别为 H-8192：249/239/262，H-32768：240/242/243；六个评估均未观察到安全违规。

## 不确定性与学习曲线

主差值以相同 tape、相同任务集合和相同条件配对。对每个 tape 先在三个训练 seed 内平均 H32768−H8192 的任务完成率，再以 64 个父 tape 为重采样单位做 20000 次 paired bootstrap。结果为均值 `+0.0295139`，tape-only 95% 区间 `[-0.0026042, +0.0616319]`（即 `[-0.2604pp, +6.1632pp]`）。区间只描述测试 tape 不确定性，不覆盖训练 seed 不确定性；三个 seed 的整数差值 `+10/+8/+16` 必须单列，不能被该区间替代。全部测试 tape 均有有效结局，没有因删失而静默删分母。

可复核的 validation 观察如下。8192 步历史曲线来自原运行；延长段保存的曲线点为 26624、28672、30720、32768。由于上述阶段文件覆盖问题，不能把 16384/24576 的文件名当成有效中间 checkpoint 或独立曲线节点：

| seed | 原 8192 曲线（步:按时到达/96） | 延长段有效观察（步:按时到达/96） |
|---|---|---|
| 1101 | 4096:41，6144:39，8192:37 | 26624:46，28672:47，30720:47，32768:47 |
| 2203 | 2048:39，4096:46，6144:46，8192:46 | 26624:48，28672:48，30720:46，32768:47 |
| 3307 | 2048:39，4096:40，6144:38，8192:36 | 26624:38，28672:39，30720:40，32768:41 |

这些 validation 点仅为学习曲线观察，不用于挑选 final-test checkpoint；最终测试固定比较 8192 与 32768。

## 预登记判断

“预算限制得到支持”要求同时满足：平均提升至少 3pp、至少 2/3 seed 改善、父 tape bootstrap 下界大于 0、无安全违规和关键账本缺失。本轮只有 seed 改善（3/3）以及安全/终态覆盖条件满足；平均提升 2.95pp，bootstrap 下界为负。因此该信号未满足。

结论是：在本次从 8192 到 32768 的预算扩展和固定环境条件下，观察到一致方向的局部改善，但证据不足以支持“8192 预算不足”这一预登记判断，也不证明 H 已收敛。H 32768 仍低于共享规则点估计。该结果不回答 E、世界模型或 GPPO 融合的研究问题，不自动触发继续训练。

## 制品

机器可读审计汇总：[summary.json](../artifacts/m10-history-budget-extension-20260914/summary.json)。完整运行根目录为 `E:\Z博士\9.2日\m10-history-budget-extension-20260914-v1`，包含六个 evaluation ledger、三个训练 rollout ledger、更新记录、源/最终 checkpoint、测试 tape、manifest、恢复预检和运行身份。

关键文件 SHA-256：

- `results.json`: `3692dd15e9fad2f7dacff3c78142c3799323b0f4ff9d44d9a4090e5e9f2adf58`
- `frozen-tapes.json`: `3fc1a9e667739f9966931318486b32305a0a3da8863b43f22dc13eb4a97d30aa`
- seed 1101 `checkpoint-8192.pt`: `ba7337d1aeacd3fca652132a3ddd53cbe7c7fddad4bf7a02e62a6c4a973d5d7d`
- seed 2203 `checkpoint-8192.pt`: `e25bd7f39a596f3de5b65c6a7d19493fecaf3ad0984c836285d5ca14f1e636d2`
- seed 3307 `checkpoint-8192.pt`: `b46e2ab66422d4204e69fb48b16567b823b8cdcc730295a9d43b70475a0f1e04`
- seed 1101 `checkpoint-32768.pt`: `e77a8ccba9f65d806eeb27e8404c8b3ea0600295dfa0afe43bfb154892919027`
- seed 2203 `checkpoint-32768.pt`: `9b5cbbb1665f4d34b8b20df5129c35f830d2b4829f15f1412927405c5c69ab4a`
- seed 3307 `checkpoint-32768.pt`: `cd807e2a24aeda80177288b4d8da0cc1ee6d7ec4d80092f726d647948d116e12`

本轮只完成本地结果收束和审计提交；没有执行新的训练或测试。GitHub Release/大型资产远端封存不作为本地实验完成的依据，需另行记录上传及独立下载校验状态。

本地独立归档：`E:\Z博士\9.2日\m10-history-budget-extension-20260914-v1.zip`，大小 289062963 bytes，SHA-256 `e57254c0115a724e6a8286fb320ec86107eed48caba18585d57918157c30c94`。压缩包共 159 个条目，未发现凭据类文件名；尚未声称远端 Release 上传或独立下载校验通过。
