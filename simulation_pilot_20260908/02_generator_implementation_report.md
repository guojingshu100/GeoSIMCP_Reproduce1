# S1--S4 模拟生成器实现报告

## 1. 生成器目的

`gccp_sim_s1_s4_v1` 用于构造一个严格配对的横截面模拟数据集，研究“与目标有关的因果信息在什么情况下能帮助估计 prediction error 的局部尺度”。本阶段只实现数据、split、校验和文档，不拟合 GCCM 或任何 conformal 方法。

## 2. S1--S4 精确方程

版本为 `gccp_sim_s1_s4_v1`，`n=2000`、`seed=20260907`、`alpha=0.10`。使用一次 `numpy.random.default_rng(20260907)`，按固定顺序生成：

- `s_x,s_y ~ Uniform(0,1)`，相互独立；
- `x1,x2,eta,x_noise,epsilon ~ Normal(0,1)`，彼此独立且与坐标独立；
- `g_true = sin(2*pi*s_x) + cos(2*pi*s_y)`；
- `g_proxy = g_true + 0.35*eta`；
- `h(z) = 0.6 + 1.4/(1+exp(-1.2*z))`。

四个场景严格使用：

| 场景 | `mu_true` | `sigma_true` | `y` |
|---|---|---|---|
| S1 | `2*x1` | `h(x1)` | `mu_true + sigma_true*epsilon` |
| S2 | `2*x1` | `h(x2)` | `mu_true + sigma_true*epsilon` |
| S3 | `2*x1 + 1.5*g_true` | `h(g_true)` | `mu_true + sigma_true*epsilon` |
| S4 | `2*x1` | `1.3` | `mu_true + sigma_true*epsilon` |

自动校验从保存后的 observed/truth CSV 重新读取数值，再逐式计算右侧并与保存值比较，而非只检查相关系数。

## 3. Observed 与 oracle truth

每个场景分别保存 observed 和 truth CSV。Observed 列严格为 `sample_id,scenario,split,local_calibration_role,s_x,s_y,x1,x2,g_proxy,x_noise,y`。Truth 列严格为 `sample_id,scenario,split,local_calibration_role,mu_true,sigma_true,epsilon,g_true,eta`。

`mu_true`、`sigma_true`、`epsilon`、`g_true`、`eta` 未出现在任何 observed 文件中。潜在 `g_true` 不提供给后续拟合方法，只有带 measurement noise 的 `g_proxy` 可见。

## 4. 冻结 outer split 和 internal role

Outer split 使用独立的 `numpy.random.default_rng(620907)`，固定为 train/local_calibration/validation/final_calibration/test = `1000/300/200/200/300`。

Local calibration 内使用 `numpy.random.default_rng(620908)` 固定划分 `scale_fit=200` 和 `tuning_calibration=100`。其他 outer split 的 `local_calibration_role` 为空。Outer split 和 internal role 是两个独立列，后续参数选择完成后仍可用 `split == "local_calibration"` 恢复全部 300 行。

## 5. 配对随机数设计

四个场景共享相同的 sample ID、坐标、`x1`、`x2`、`eta`、`x_noise`、`epsilon`、`g_true`、`g_proxy`、outer split 和 internal role。场景之间只有 `mu_true`、`sigma_true` 和由它们得到的 `y` 不同。

六个场景对的 `y` 向量均经逐元素比较确认不完全相同；S1/S4 以及 S2/S4 的 `mu_true` 经数值检查完全一致。

## 6. 校验结果

`generator_check.json` 的总体状态为 `PASS`，`test_generator.py` 的 3 个自动测试也全部通过。

| 项目 | 结果 |
|---|---|
| 每个 observed/truth 文件行数 | 2000 / 2000 |
| Outer split | train 1000；local_calibration 300；validation 200；final_calibration 200；test 300 |
| Local internal role | scale_fit 200；tuning_calibration 100 |
| S1 `sigma_true` | min 0.6037547189438385；max 1.9866846004707228 |
| S2 `sigma_true` | min 0.6259159506691434；max 1.9857352698161960 |
| S3 `sigma_true` | min 0.7172586122262634；max 1.8833988967749580 |
| S4 `sigma_true` | min=max=1.3 |
| finite、positive、唯一 ID、observed/truth 对齐 | PASS |
| split 完整性与跨场景一致性 | PASS |
| paired common variables | PASS |
| observed/oracle leakage | PASS |
| 两个临时目录重复生成的八个 CSV SHA-256 | 全部逐文件相同，PASS |

完整的逐项状态、文件名、行数、范围和 SHA-256 位于 `generator_check.json`。正式输出的 SHA-256 与临时重复生成记录一致。

## 7. 各场景的科学解释

- S1 检验 mean driver 与 residual-scale driver 对齐的情况；`x1` 同时控制 mean 和 noise scale。
- S2 中 `x1` 驱动 conditional mean，`x2` 是 variance driver 或 residual-scale driver。`x2` 不是 conditional mean 的 cause，但它改变 Y 的条件噪声尺度和条件分布，因此不能描述成对 Y 完全非因果。
- S3 用 latent spatial `g_true` 同时驱动 mean 和 scale，方法只能观察带噪 proxy `g_proxy`。这检验 proxy 是否能为局部误差尺度提供有用信息。
- S4 是 homoscedastic control，用来检查局部尺度建模在没有真实异方差时是否产生虚假收益。

## 8. 局限性

- 这是横截面模拟，不是 temporal dynamical system。
- S3 使用确定性的 latent spatial function 加 measurement noise，不是 stochastic spatial error field。
- 本生成器只检验 residual-scale alignment，本身不能验证 GCCM causal discovery。
- 后续必须分别报告 learned GCCM、oracle weighting 和 ordinary Euclidean weighting，不能互相替代。
- 生成和校验成功不构成 GCCP 优于 baseline 的证据。
