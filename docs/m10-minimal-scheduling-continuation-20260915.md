# 候选后果模型可执行续行验证

## 范围

本轮只验证冻结候选后果模型选择一个首步动作后、固定最早完成时间贪心续行的完整 makespan。没有训练、调参、重复模型决策或 GPPO；精确求解器只在各执行控制器结束后给离线最优参考。基准仍为 2 UAV、4 子任务、两条长度为 2 的依赖链；不触及旧到达协议。

## 上一轮静态主指标复核（先于新评估）

静态复算输入为已保存的 128 条逐实例、三模型 seed 结果。冻结模型 SHA-256 均与 protocol 一致：

| Seed | 模型原始平均 regret | 共享贪心原始平均 regret | 原始减少量 | 相对减少 | 原始归一化 regret |
|---:|---:|---:|---:|---:|---:|
| 1101 | 0.289063 | 0.976563 | 0.687500 | 70.40% | 0.012163 |
| 2203 | 0.507813 | 0.976563 | 0.468750 | 48.00% | 0.021091 |
| 3307 | 0.414063 | 0.976563 | 0.562500 | 57.60% | 0.017339 |
| 三 seed 平均 | 0.403646 | 0.976563 | 0.572917 | **58.67%** | 0.016864 |

原始单位为仿真时间。按实例先平均三个冻结模型的 `G-regret minus model-regret`，再对同构/异构各 64 个实例分层重采样 10,000 次，原始减少量 95% CI 为 **[0.210938, 0.997396]** 仿真时间单位。区间不覆盖训练 seed 总体不确定性。

上一轮所报 **57.44%** 和 95% CI **[0.0080218, 0.0393435]** 用的是归一化 regret；每实例 regret 除以该实例公开 scale 后，按同一实例平均 seed 差异并分层 bootstrap。它们不是原始仿真时间单位结果。按本轮预登记原始主指标，减少超过 10%、3/3 seed 均值改善、配对区间下界大于 0，故静态门槛通过，允许本轮唯一评估继续。

静态复算文件为 `artifacts/m10-minimal-consequence-continuation-20260915/static-raw-regret-recomputation.json`；其输入逐实例文件 SHA-256 为 `1aa5290e63cca25db5d2b3cc000dbfec4934fb3c2cfacf3adbbb16cb77c54579`。三份 checkpoint 完整 SHA-256 见下方制品索引。

## 冻结的新验证集与方法

使用 seed `731001`，同一生成分布，先生成同构 64、异构 64；相对于已有 train/validation/test 448 个实例及旧开发基准 32 个实例，按完整场景内容去重，重复时沿固定随机流顺序继续抽取并登记，不按调度结果筛选。新实例在首次执行任何组之前冻结并哈希。

对每个实例分别执行：

- `G`：每一步均使用原 earliest-finish/task-ID/UAV-ID 贪心；
- `M1101/M2203/M3307`：仅首步使用相应冻结模型对公开输入评分，之后所有动作均使用同一个贪心规则。

精确求解在所有控制器执行之后运行，最多 30 秒/实例、全程最多 30 分钟；若无法证明最优，仍保留 G/M 完整执行比较，但最优差距标为不可验证。模型推理只一次/实例/seed；无训练更新。任务依赖、资格、非抢占、并行 UAV、事件等待及同时完成处理沿用冻结协议。

配置 `configs/minimal-scheduling-continuation-v1.json` 在数据生成前提交并哈希。关键 checkpoint：

| Seed | best checkpoint SHA-256 |
|---:|---|
| 1101 | `a3306884f989d7307a25e2007c7fa2c46113ca3d9f08f2d34e61bc15234e664b` |
| 2203 | `c572e2e907ccc2a84f343877ee4c4e0200e9972a344e8eb3e52ca89bbc71f903` |
| 3307 | `7fbe399ed480c8192a7ccac844c34d0de2ec27de0e136bce62f74602b4c0562a` |

## 执行命令

在代码与协议冻结提交之后运行生成：

```powershell
$py = 'E:\Z博士\9.2日\world-gppo-911-local-pilot-venv\Scripts\python.exe'
& $py tools\generate_minimal_scheduling_continuation_instances.py `
  --out runs\m10-minimal-scheduling-continuation-20260915-v1\instances `
  --known-data E:\Z博士\9.2日\WORLD-GPPO_9.11-min-scheduling-benchmark-wt\artifacts\m10-minimal-consequence-dataset-20260915\dataset.json `
  --known-instances E:\Z博士\9.2日\WORLD-GPPO_9.11-min-scheduling-benchmark-wt\artifacts\m10-minimal-scheduling-benchmark-20260915\instances.json `
  --seed 731001
```

生成及内容审计通过、`instances.json` 哈希冻结后运行：

```powershell
& $py tools\run_minimal_scheduling_continuation.py `
  --instances runs\m10-minimal-scheduling-continuation-20260915-v1\instances\instances.json `
  --protocol configs\minimal-scheduling-continuation-v1.json `
  --models E:\Z博士\9.2日\m10-minimal-consequence-training-20260915-seed1101\best-inference.pt E:\Z博士\9.2日\m10-minimal-consequence-training-20260915-seed2203\best-inference.pt E:\Z博士\9.2日\m10-minimal-consequence-training-20260915-seed3307\best-inference.pt `
  --out runs\m10-minimal-scheduling-continuation-20260915-v1\evaluation `
  --device cuda --per-instance-seconds 30 --wall-seconds 1800
```

运行设备无 CUDA 时显式失败，不切换设备。结果写入唯一目录，逐实例 ledger 在每完成一个实例后持久化。

## 结果（待独立新集完成后填写）

此处只填新实例执行与配对结果，不用旧 test 代替。若实验未完整完成，必须保持“未完成/证据不足”，不得追加实例或预算。

