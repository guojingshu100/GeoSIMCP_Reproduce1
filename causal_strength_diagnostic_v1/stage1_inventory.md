# Stage 1 inventory

## 文件/函数地图

|范围|位置|
|---|---|
|生成器配置与方程|`simulation_pilot_20260908/generator/config_v1.json:21-60`|
|生成器实现/校验|`simulation_pilot_20260908/generator/generate_scenarios.py:36-70,267-289,303-326`|
|预测与残差|`simulation_pilot_20260908/predictor/fit_predictors.py:252-318`|
|scale-fit 输入/目标|`simulation_pilot_20260908/scale_comparison/fit_local_scales.py:231-306,390-469`|
|GCCM Python runner|`simulation_pilot_20260908/gccm_runtime/run_gccm_scale_fit_v1.py:39-59,83-110`|
|GCCM R wrapper|`simulation_pilot_20260908/gccm_runtime/gccm_point_graph_wrapper_v1.R:1-21,47-75`|
|环境记录|`simulation_pilot_20260908/generator/config_v1.json:85-90`; `simulation_pilot_20260908/gccm_runtime/r_environment_v1.log:1-40`|

## 已核实事实

1. S2 的生成方程在 `config_v1.json:31-35`，实现校验在 `generate_scenarios.py:280-282,289`：均值 `mu_true=2*x1`，尺度 `sigma_true=h(x2)`，响应 `y=mu_true+sigma_true*epsilon`。S4 在 `config_v1.json:43-48`、实现 `generate_scenarios.py:286-289`：均值同为 `2*x1`，尺度为常数 `1.3`，噪声为 `epsilon~Normal(0,1)`。公共定义为 `h(z)=0.6+1.4/(1+exp(-1.2z))`（配置 `:21-24`；实现 `:36-38`）；`x1,x2,epsilon` 均为独立标准正态，坐标为独立 `Uniform(0,1)`（配置 `:11-19`）。

2. 总切分在 `config_v1.json:50-60`：train/local_calibration/validation/final_calibration/test = 1000/300/200/200/300；local_calibration 内 scale_fit=200、tuning_calibration=100。实际契约由 `generate_scenarios.py:307-326` 校验。scale-fit 样本 ID 由 `fit_local_scales.py:231-245,277-306` 从预测和授权残差中按 `split=local_calibration` 且 `local_calibration_role=scale_fit` 过滤，要求恰好 200 个；runtime 再读取准备文件并按 `sample_id` 排序（`run_gccm_scale_fit_v1.py:41-57`）。

3. 冻结预测由 `fit_predictors.py:268-292` 生成：只用 train 标签拟合随机森林，固定树顺序求均值并写入 `predictor_outputs/{S2,S4}_predictions.csv`。绝对残差由授权 train、local_calibration、validation 标签减预测后取绝对值，在 `:294-318` 写入对应 `*_authorized_residuals.csv`；scale-fit 目标明确取其中的 `abs_residual`（`fit_local_scales.py:277-306`）。

4. 完整真实均值是可恢复的：生成器保存 truth 列 `mu_true`（`config_v1.json:74-83`），并在校验中按 S2/S4 公式重算（`generate_scenarios.py:267-289`）。本阶段未读取 truth 数值；冻结 manifest 仅记录 truth 文件已哈希而未解析（`frozen_input_manifest_v1.json:4-7,25`）。

5. R wrapper 固定方向为 `effect_xmap_cause`（`gccm_point_graph_wrapper_v1.R:12-15,49-55`），对六个特征逐一运行 `GCCMLattice`，取各 library size 的平均 rho。收敛条件和 terminal strength 在 `:57-65`；仅收敛特征的 strength 求和后归一化成 `weight=strength/total`，再设 `gamma=6*weight`（`:67-71`），写入 `weights.csv`（`:72-75`）。

6. 已记录 Python 3.13.5、NumPy 2.1.3、Pandas 2.2.3、SciPy 1.15.3、scikit-learn 1.6.1（`generator/config_v1.json:85-90`）。R 为 4.6.1；foreach 1.5.2、doParallel 1.0.17、iterators 1.0.14、codetools 0.2-20、parallel 4.6.1（`r_environment_v1.log:1-40`）。本阶段未安装或升级。

## 未解决事项

- `HANDOFF.md` 未在工作区定向搜索结果中发现；其内容无法核实。
- UNVERIFIED：用户所称“现有 S2/S4 simulation code”的入口调用链、以及运行时实际使用的 historical core 版本；Stage 2 需检查 `generate_scenarios.py:run_pipeline`（约 `:400-520`）和 `run_gccm_scale_fit_v1.py:118-150` 的路径配置。
- UNVERIFIED：`scale_fit_preparation.csv` 中各派生 scale 列的精确拟合算法；Stage 2 需检查 `fit_local_scales.py:353-469` 的计算段。

## Stage 2 最小检查集合

`generator/config_v1.json`；`generator/generate_scenarios.py` 的 `run_pipeline`/`generate_scenario`；`predictor/fit_predictors.py` 的主循环；`scale_comparison/fit_local_scales.py` 的 `load_scale_fit_features`、`load_scale_fit_target` 与输出段；`gccm_runtime/run_gccm_scale_fit_v1.py` 的配置/校验段；R wrapper 的主循环与权重段。
