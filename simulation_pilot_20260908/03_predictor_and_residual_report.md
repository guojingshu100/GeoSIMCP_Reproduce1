# Shared predictor 与 residual preparation 实现报告

## 1. 本阶段目标

本阶段基于已通过校验的 `gccp_sim_s1_s4_v1` 数据，冻结一个可供后续所有 uncertainty-localization 方法共享的点预测基线。每个场景单独拟合一个 predictor，因为四个场景的 response 不同；算法、features、预处理和超参数在 S1--S4 之间完全一致。本阶段只生成点预测和获准 split 的 residual，不实现 GCCM、Normalized-EUC、Normalized-GCCM、conformal quantile、prediction interval 或 test evaluation。

## 2. 冻结 predictor 配置

Predictor stage version 为 `gccp_rf_predictor_v1`，使用 sklearn `RandomForestRegressor`。完整配置如下：

| 参数 | 冻结值 |
|---|---:|
| `n_estimators` | 300 |
| `max_depth` | `None` |
| `min_samples_leaf` | 5 |
| `max_features` | 1.0 |
| `bootstrap` | `True` |
| `random_state` | 420907 |
| `n_jobs` | -1 |
| `criterion` | `squared_error` |

其余 sklearn 参数也已按 `get_params(deep=False)` 完整记录在 `predictor/predictor_config_v1.json`。当前环境为 Python 3.13.5、NumPy 2.1.3、Pandas 2.2.3、SciPy 1.15.3、scikit-learn 1.6.1 和 joblib 1.4.2。未使用 `StandardScaler`，未进行 feature selection、GridSearchCV 或 validation tuning。

为消除 `n_jobs=-1` 在并行汇总 300 棵树预测时约 `1e-15` 量级的浮点加法次序差异，冻结 CSV 使用 `fixed_estimators_order_mean`：按 `estimators_` 的固定顺序取得各树预测后求均值。模型拟合参数仍为 `n_jobs=-1`；重载模型的原生 `predict` 与冻结值在 `rtol=0, atol=1e-12` 下逐场景一致。

## 3. 授权与禁止的数据使用

模型拟合只读取 `train` 的 1000 个 labels。`local_calibration` 和 `validation` labels 只用于生成 residual 与本阶段获准的 non-oracle diagnostics。`final_calibration` 和 `test` 仅用 observed features 计算 `y_pred`；二者的 `y` 没有被提取、调试、存储或用于指标。

每个 `S*_predictions.csv` 有 2000 行，只含 `sample_id, scenario, split, local_calibration_role, y_pred`。每个 `S*_authorized_residuals.csv` 有 1500 行，只覆盖 `train, local_calibration, validation`，字段为 `sample_id, scenario, split, local_calibration_role, y_pred, residual, abs_residual`。两类文件均不含原始 `y` 或任何 oracle-only 字段。

Truth 文件在四个模型、prediction/residual 文件及其 SHA-256 全部保存后，才由隔离 oracle 函数读取；只读取 `local_calibration` 和 `validation` 的 truth rows。Oracle 结果没有参与拟合、feature selection、hyperparameter/method/bandwidth selection。

## 4. Feature 列表与科学理由

冻结 features 恰为：`s_x, s_y, x1, x2, g_proxy, x_noise`。`sample_id, scenario, split, local_calibration_role` 只作为元数据；`mu_true, sigma_true, epsilon, g_true, eta` 以及其他 truth 变量禁止进入 predictor。

六个 observed features 在所有场景中相同。坐标与 `g_proxy` 使 S3 的 predictor 能利用可观测空间信息；S1、S2、S4 仍保留相同输入，避免因场景切换 feature availability。`x_noise` 等潜在无关变量不在本 pilot 中筛选，以免引入额外选择自由度。

## 5. 拟合与预测流程

脚本先读取 `generator/config_v1.json` 与 `generator_check.json`，确认 generator 为 `PASS`，再重算并核对 8 个生成 CSV 的 SHA-256，写入 `frozen_input_manifest_v1.json`。随后对每个场景独立执行：

1. 只加载元数据和六个 features，核验 2000 行及五个 outer split；
2. 单独读取 1000 个 train labels 并拟合一个 frozen Random Forest；
3. 对所有 2000 行生成 `y_pred`，但不读取 final-calibration/test labels；
4. 仅对 train、local_calibration、validation 共 1500 行计算 `residual=y-y_pred` 和 `abs_residual`；
5. 保存 prediction、authorized residual 与场景专属 joblib model；
6. 重载 model 并核验预测；全部 non-oracle artifacts 保存和哈希冻结后，才运行隔离 oracle diagnostic。

四个场景的 observed predictor matrix、`sample_id` 顺序和 split assignment 完全一致，response 在所有场景对之间均不同，且保存了四个独立 model。后续方法可以直接按 `sample_id` join 同一批 frozen prediction/residual，无需重新拟合 predictor。

## 6. Non-oracle diagnostics

以下指标只使用授权 labels。`MAE` 与 `mean_abs_residual` 按定义相同，同时保留两列以满足阶段契约。

| 场景 | subset | n | RMSE | MAE | R2 | mean abs residual | median abs residual | P90 abs residual |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| S1 | train | 1000 | 0.9560 | 0.7246 | 0.8514 | 0.7246 | 0.5540 | 1.5644 |
| S1 | scale_fit | 200 | 1.5120 | 1.1371 | 0.5964 | 1.1371 | 0.9049 | 2.4901 |
| S1 | tuning_calibration | 100 | 1.3106 | 1.0327 | 0.6167 | 1.0327 | 0.8988 | 1.9294 |
| S1 | validation | 200 | 1.4488 | 1.1364 | 0.6934 | 1.1364 | 0.9694 | 2.3952 |
| S2 | train | 1000 | 0.9676 | 0.7321 | 0.8496 | 0.7321 | 0.5666 | 1.5681 |
| S2 | scale_fit | 200 | 1.4563 | 1.1346 | 0.6392 | 1.1346 | 0.9218 | 2.3185 |
| S2 | tuning_calibration | 100 | 1.2916 | 1.0093 | 0.5963 | 1.0093 | 0.8324 | 1.9261 |
| S2 | validation | 200 | 1.4550 | 1.1188 | 0.6993 | 1.1188 | 0.9830 | 2.1112 |
| S3 | train | 1000 | 1.0288 | 0.7887 | 0.8702 | 0.7887 | 0.6271 | 1.6935 |
| S3 | scale_fit | 200 | 1.4763 | 1.1353 | 0.7185 | 1.1353 | 0.8974 | 2.2825 |
| S3 | tuning_calibration | 100 | 1.4846 | 1.1309 | 0.6748 | 1.1309 | 0.8527 | 2.5456 |
| S3 | validation | 200 | 1.5745 | 1.1969 | 0.7452 | 1.1969 | 0.9041 | 2.6228 |
| S4 | train | 1000 | 0.9192 | 0.7202 | 0.8592 | 0.7202 | 0.5814 | 1.5215 |
| S4 | scale_fit | 200 | 1.4454 | 1.1349 | 0.6263 | 1.1349 | 0.9805 | 2.3746 |
| S4 | tuning_calibration | 100 | 1.3016 | 1.0712 | 0.6103 | 1.0712 | 0.9447 | 2.0903 |
| S4 | validation | 200 | 1.4577 | 1.1485 | 0.6911 | 1.1485 | 0.9516 | 2.2426 |

Train 指标较好属于 Random Forest 对拟合样本的 in-sample diagnostics，不应当作泛化表现。Validation 指标没有用于改变模型配置。

## 7. 隔离的 oracle diagnostics

Oracle diagnostics 只覆盖完整 `local_calibration` 300 行和 `validation` 200 行。表中 `r_abs` 为 Pearson corr(`abs_residual`, `sigma_true`)，`rho_abs` 为对应 Spearman correlation，`r_sq` 为 Pearson corr(`residual^2`, `sigma_true^2`)，`r_bias` 为 Pearson corr(`abs(y_pred-mu_true)`, `sigma_true`)。

| 场景 | subset | n | r_abs | rho_abs | r_sq | r_bias | mean abs bias | RMSE(`y_pred`,`mu_true`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| S1 | local_calibration | 300 | 0.3543 | 0.2845 | 0.3536 | 0.3619 | 0.3036 | 0.4101 |
| S1 | validation | 200 | 0.2131 | 0.2380 | 0.1657 | 0.0853 | 0.3426 | 0.5402 |
| S2 | local_calibration | 300 | 0.2721 | 0.2200 | 0.2835 | 0.2255 | 0.3211 | 0.4175 |
| S2 | validation | 200 | 0.2449 | 0.2106 | 0.2376 | 0.1520 | 0.3750 | 0.5692 |
| S3 | local_calibration | 300 | 0.2608 | 0.2407 | 0.2324 | 0.0599 | 0.5205 | 0.6405 |
| S3 | validation | 200 | 0.2514 | 0.2511 | 0.2001 | -0.0147 | 0.5305 | 0.7367 |
| S4 | local_calibration | 300 | 未定义 | 未定义 | 未定义 | 未定义 | 0.3091 | 0.3945 |
| S4 | validation | 200 | 未定义 | 未定义 | 未定义 | 未定义 | 0.3575 | 0.5516 |

S4 的 `sigma_true` 恒为 1.3，所有以它为一侧变量的相关系数在数学上未定义，已在 JSON 中写为 `null`。单个 realized `abs_residual` 同时受随机 `epsilon` 和 predictor error 影响，是局部误差尺度的高噪声观测，因此它与 `sigma_true` 的相关不必接近 1；这些相关值只能用于理解生成器和 predictor，不能用于选择后续方法或参数。

## 8. Leakage 与 reproducibility 检查

`predictor_check_v1.json` 的 `overall_status` 为 `PASS`。具体证据包括：

- generator 状态和 8 个输入 CSV hashes 全部核对通过；
- 每个场景恰有 1000 行进入 `fit`，且访问日志只记录 train labels；
- prediction 为 2000 行，authorized residual 为 1500 行；local calibration roles 恰为 `scale_fit=200`、`tuning_calibration=100`；
- prediction/residual schemas 均排除 `y` 和 oracle 字段，residual 文件排除 final-calibration/test rows；
- truth 在 non-oracle outputs 保存并哈希后才读取，且只读取 local-calibration/validation；
- 两个独立临时目录的四组 prediction CSV hashes 全部相同，四组 residual CSV hashes 全部相同；本机四个 joblib hashes 也全部相同；
- 四个重载模型均在 `rtol=0, atol=1e-12` 下重现冻结 `y_pred`；
- `test_predictor_stage.py` 的 8 项契约测试全部通过。

唯一 warning 是 S4 常数尺度使相关系数未定义。这是设计属性，不是验证失败。

## 9. 分场景解释

S1 中 mean driver 与 residual-scale driver 均为 `x1`；当前 predictor 对 conditional mean 的误差仍会混入 residual，但 oracle diagnostic 显示 realized absolute residual 与 scale 有正向关联。S2 中 `x2` 不驱动 conditional mean，却是 variance/residual-scale driver；后续方法是否能利用这种分离结构仍是待检验问题，不能把 `x2` 描述为完全非因果。S3 的 `g_true` 为 oracle-only latent spatial truth，实际模型只能使用坐标和 noisy `g_proxy`，因此其较大的 prediction-bias diagnostics 与测量误差和拟合难度相容。S4 为 homoscedastic control，局部尺度没有真实异质性可恢复，其相关系数本来就没有定义。

这些诊断不说明 GCCM 或 GCCP 会优于普通 Euclidean 方法，也没有打开最终 calibration/test 表现。

## 10. 局限性

- 本阶段评估的是 realistic fitted predictor，不是 true conditional mean。
- Residual 同时包含 irreducible noise 和 point-prediction error。
- 较低的 `abs_residual` 与 `sigma_true` correlation 本身不能否定场景设计。
- 后续所有 uncertainty methods 必须使用本阶段冻结的 predictions 和 residuals。
- Final-calibration 和 test performance 仍保持未打开。
- 本阶段不能证明 GCCM 或 GCCP 有效。
- 当前只有一个 seed 的单次 pilot 数据，不能据此形成稳定的性能结论。

## 11. 精确下一任务

下一实现任务是在不读取 final-calibration/test labels 的条件下，冻结 residual-scale comparison 的配置与接口：只在 `scale_fit=200` 上拟合共同的 feature preprocessing 和正值 KNN local-scale estimator，固定相同的 `k`/scale-floor candidates、tie rule 与 search budget；为 learned 分支实现以 `abs_residual` 为 target 的 GCCM weight 学习，并与 ordinary standardized Euclidean 距离严格分开。随后只用 `tuning_calibration=100` 构造 provisional normalized scores，用 `validation=200` 按预注册规则选择参数。Oracle weighting 必须作为独立诊断，不得标成 learned GCCM/GCCP；该下一阶段仍不读取 final-calibration/test labels。
