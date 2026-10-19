# GCCP 四场景初步模拟比较协议草案

## 1. 研究问题与可证伪目标

当前问题是：什么时候与目标有关的因果信息能帮助估计 prediction error 的局部尺度？

比较不预设 GCCP 必须胜出。主要可证伪假设是：只有当 learned distance 捕捉到 residual magnitude 或 noise scale 的真实异质性时，它才应相对普通 Euclidean distance 改善 coverage-width trade-off 或 Interval Score。在 homoscedastic S4 中，局部尺度方法没有理论义务优于简单控制；若 learned GCCM 明显占优，反而要优先排查 selection noise、data leakage 或过拟合。

变量只要影响条件方差或 noise scale，就可能对本研究问题具有因果意义；不能因为它不进入 conditional mean 就称其“非因果”。同样，能解释 conditional mean 的变量不自动解释 residual magnitude。

## 2. 场景状态与生成器冻结要求

当前项目没有实际 generator，所以下表只记录任务说明规定的语义，不补写未知方程：

| 场景 | 已规定的语义 | 仍需在 generator 中明确冻结 |
|---|---|---|
| S1 | 同一个变量控制 conditional mean 与 noise scale | 精确方程、系数、分布、空间结构、observed/hidden 身份 |
| S2 | 不同变量分别控制 conditional mean 与 noise scale | 两变量方程、相关结构、是否都 observed |
| S3 | spatial common cause 同时影响 response 与 noise scale，并有 observed proxy | latent field、proxy measurement equation、proxy noise、哪些 truth 不可见 |
| S4 | homoscedastic control | mean equation、常数 scale、其他变量分布 |

实现前必须由研究者确认或找回这些方程。不能从 Seattle 或旧 GeoSIMCP synthetic notebook 复制一个相似生成器来代替。

generator 的最小冻结配置应包括：`n=2000`、scenario 名、replicate seed、各随机成分的派生 seed、全部结构参数、split 数量、输出 schema 和版本号。一个 master seed 应通过确定性规则派生 component seeds，避免调用顺序变化导致结果变化。

## 3. 数据角色与固定 split

每个 scenario/replicate 使用相同的 2000 个 `sample_id` 和同一 split assignment：

| 数据角色 | n | 允许用途 |
|---|---:|---|
| `training` | 1000 | 拟合 predictor；需要 training residual 时只能用 out-of-fold prediction 产生。 |
| `local_calibration` | 300 | 产生 absolute residual，学习局部 scale 和 learned GCCM distance。 |
| `validation` | 200 | 选择 `k`、scale floor 和其他预先列出的候选参数。 |
| `final_calibration` | 200 | 配置冻结后，仅估计 normalized conformal score 的最终 quantile。 |
| `test` | 300 | 配置和 final quantile 均冻结后，仅作一次最终评价。 |

现有 PM2.5 generator 不存在，因此不能说它已遵循这一 split。历史 PM2.5 实验是 `990/141/142/142` 四分法，没有独立 local/final calibration；不能直接复用其 split 逻辑作为新模拟协议。

跨 S1--S4 应固定：

1. `sample_id` 集合和顺序；
2. split assignment；
3. 坐标和可共享的基础随机量；
4. predictor 类型和超参数；
5. 两个比较方法的 validation 候选数量和选择规则。

如果场景设计允许，应使用 paired random numbers，使方法差异和场景差异不被不必要的 Monte Carlo 噪声放大。哪些随机量可以共享必须写进 generator config，不能事后决定。

## 4. Observed 数据与 oracle truth 分离

建议每个 scenario/replicate 保存两个逻辑上分离的表，字段名最终由真实方程决定：

- observed table：`sample_id`、`scenario`、`split`、坐标、所有可给 predictor/GCCM 使用的 observed features、`y`；
- oracle table：`sample_id`、`mu_true`、`sigma_true`、latent common cause、真实 driver/weight 等只用于诊断的字段。

训练和 learned GCCM 接口不得接收 oracle table。评估代码可以在产生所有 learned results 后按 `sample_id` 合并 oracle truth，计算 scale recovery diagnostics。Oracle 权重必须使用独立方法名，例如 `Oracle-GCCM-weight`，不得写成 `GCCP` 或 `learned GCCM`。

## 5. Predictor 与 residual target

对每个 scenario/replicate，只在 `training` 上拟合一个固定 predictor `f_hat`，两个 primary 方法共享同一个 fitted object 和同一组 prediction。predictor hyperparameters 应在正式比较前冻结；若必须选择，只能在 training 内交叉验证。

局部尺度的基本学习目标为：

`a_i = |y_i - f_hat(x_i)|`,  对 `i in local_calibration`。

这与历史 PM2.5 GCCM 有本质区别：历史 GCCM 的 target 是 detrended `PM2.5`，不是 `a_i`。本 pilot 的 primary learned GCCM 应学习 feature 与 `a_i`（或经预注册的正变换，例如 `log(a_i+delta)`）之间的空间 cross mapping。若同时想保留 outcome-target GCCM，应标为 secondary diagnostic，不能与 residual-target GCCM 混为一项。

## 6. Primary scale comparison

建议第一项最小控制比较只改变 distance：

1. `Normalized-EUC`：在同一标准化 observed feature 空间使用普通 Euclidean distance；
2. `Normalized-GCCM`：使用 learned GCCM 权重形成 weighted Euclidean distance。

共同标准化器只在当前 scale-fit 数据上拟合。若 learned weights 为 `w_j>=0` 且和为 1，沿用历史可解释缩放：

`gamma_j = p*w_j`,

`d_GCCM(x,x') = sqrt(sum_j gamma_j*(z_j-z'_j)^2)`。

这样等权时 `gamma_j=1`，严格退化为普通 standardized Euclidean distance，可作为单元检查。

对任一 distance `d_m`，建议先使用简单可解释的 KNN scale estimator：

`s_hat_m(x;k,epsilon) = max(epsilon, mean{a_i: i 属于 x 的 k 个最近 scale-fit 样本})`。

第一轮不要同时改变 kernel family、scale statistic 和 distance。两个方法使用完全相同的 `k` candidates、`epsilon` 规则、tie handling 和 search budget。可以预注册一个有限 `k` grid，例如 `[10,20,30,50,75,100]`；最终 grid 必须在运行前写入 frozen config。`epsilon` 建议由 scale-fit residual 的固定低分位数或固定数值规则得到，不可看 validation/final/test 结果后修改。

是否加入 geographic kernel 当前项目无法确认。为了直接回答“learned causal feature distance 对局部误差尺度是否有增益”，首个 primary comparison 宜先比较纯 feature distance。如果决定加入 geography，则必须在 EUC 与 GCCM 两边使用完全相同的 geographic distance、融合形式、bandwidth candidates 和搜索预算。

## 7. Validation 所需的内部 split

要用 validation Interval Score 选择 normalized method 参数，必须先有一个不依赖 `final_calibration` labels 的 provisional normalized-score quantile。因此 `local_calibration=300` 需要一个内部机制。

明天前可执行的简单方案是固定拆成：

- `scale_fit=200`：学习 GCCM weights、标准化器和 KNN scale estimator；
- `tuning_calibration=100`：只计算 provisional normalized scores 和 provisional quantile。

该内部 assignment 对两个方法完全相同，并在所有场景中按相同 `sample_id` 配对。对每个候选配置：

1. 在 `scale_fit` 计算 residual target，并拟合 EUC/GCCM scale；
2. 对 `tuning_calibration` 计算 `r_i=a_i/s_hat(x_i)`；
3. 用 split-conformal finite-sample order statistic 得到 provisional `q_tune`；
4. 用 `validation` labels 计算 interval、coverage、mean width 和 Interval Score；
5. 按同一规则选 EUC 和 GCCM 配置。

更充分的正式版本可在 local calibration 内 cross-fit normalized score，减少 200/100 拆分造成的信息损失，但计算和实现更复杂。pilot 先使用固定 200/100，报告中明确样本量限制。

如果所有 GCCM algorithm settings（例如 `E/tau/b` 和 convergence rule）都在运行前冻结，则不需要额外数据来选择这些设置。若要用数据选择 GCCM embedding 或 convergence threshold，就必须在 `scale_fit` 内再做 inner split/cross-fitting；不能使用 `tuning_calibration`、validation 或 final-calibration labels 来反复调整 GCCM。

## 8. Validation 选择规则

建议对两个方法使用完全相同的预注册规则：

1. 在 validation coverage `>=0.90` 的候选中选择 mean Interval Score 最低者；
2. 如果没有可行候选，选择 validation coverage 最高者，再以 Interval Score 打破平局；
3. 保存 `coverage_constraint_feasible=false`，不得把 fallback 写成满足 90% coverage；
4. 若多个候选在数值容差内并列，固定选择较大的 `k`，以减少不稳定局部估计。

历史 PM2.5 使用“不可行时直接取全局最小 Interval Score”，导致 validation coverage 明显低于 0.90。新 pilot 可以采用上面的保守 fallback，但必须在首次运行前冻结，不能看到结果后改规则。

选择完成后，允许在不接触 final/test labels 的前提下用完整 `local_calibration=300` 重新拟合该方法的 scaler、learned GCCM weights 和 scale estimator。该 refit 规则也必须提前写入 config。

## 9. Final calibration 与 test

配置冻结后，对 `j in final_calibration` 计算：

`r_j = |y_j-f_hat(x_j)| / s_hat(x_j)`。

当 `n_final=200`、`alpha=0.1` 时：

`k_final = ceil((200+1)*(1-0.1)) = 181`，

`q_final = 第 181 小的 r_j`。

测试区间为：

`[f_hat(x)-q_final*s_hat(x), f_hat(x)+q_final*s_hat(x)]`。

严格执行以下访问顺序：

1. predictor、preprocessing、GCCM weights、scale hyperparameters 全部冻结；
2. 读取 final-calibration labels，只计算 `q_final`；
3. 冻结 `q_final` 和 manifest/hash；
4. 构造所有 test intervals；
5. 最后一次性读取 test labels 评价。

Final-calibration labels 不得用于 predictor、GCCM weights、preprocessing、`k/epsilon` 或任何超参数选择。Test labels 只用于最终指标。

## 10. 指标与报告边界

每个 method/scenario/replicate 至少保存：

- empirical coverage 和 covered count；
- mean/median interval width；
- Interval Score 及逐样本 score；
- scale estimator 的最小值、分位数、最大值和 finite/positive 检查；
- KNN 距离和 neighbor overlap；
- learned GCCM weights、convergence diagnostics 和 `n_eff`；
- 与 `Normalized-EUC` 的 paired per-sample score difference。

Oracle truth 只用于附加诊断：`corr(s_hat,sigma_true)`、scale ranking、oracle-weight 上限和 latent/proxy sensitivity。主表必须把 learned methods 与 oracle rows 分开。

单个 seed 的 pilot 只用于验证方向和代码，不用于稳定性能结论。正式 Monte Carlo 的 replicate 数、置信区间和随机效应汇总应在 pilot 通过后另行冻结。

## 11. Generator 必须通过的检查

每个 S1--S4 pilot 输出必须同时满足：

1. 同一 config/seed 运行两次得到相同文件内容或同一规范化 hash；
2. observed 数值、`y`、`mu_true`、`sigma_true` 全部 finite；
3. `sigma_true>0` 对全部 2000 个样本成立；
4. 五个 split 两两不交且并集恰为全部 `sample_id`；
5. 每个 scenario 有 2000 个唯一 `sample_id`；
6. S1--S4 的 `sample_id` 顺序和 split 完全一致；
7. observed 输出不含 latent common cause、`sigma_true`、true driver weights 等 oracle-only 字段；
8. config、seed、方程版本和输出 hash 一并写入 `generator_check.json`。

当前没有运行这些检查，因为生成器和数据都不存在。

## 12. 立即执行就绪与下一任务

当前 Python 的 NumPy/Pandas/SciPy/sklearn 足以支持一个基础 generator 和轻量验证。R/GCCM 暂时被缺失的 `Rscript` 阻断，`geoconformal` 包导入还被缺失的 `geopandas` 阻断。不要在确认生成器前进行全局环境迁移。

精确的下一实现任务是：先找回或由研究者确认 S1--S4 的数学方程，然后创建一个版本化 generator 与 frozen config，生成单个 paired pilot replicate 到 `simulation_pilot_20260908`，并产生通过上述八项检查的 `generator_check.json`。只有该检查通过后，才实现 `Normalized-EUC`、`Normalized-GCCM` 和严格分离的 oracle diagnostic。
