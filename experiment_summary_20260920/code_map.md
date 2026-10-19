# 代码导航

> 路径均相对于项目根目录 **F:\GeoSIMCP_Reproduce**。Notebook 的“主函数”列同时列出已识别的关键函数或核心单元格用途。

| Method | Code path | Main function / class | Purpose |
|---|---|---|---|
| GeoCP（历史） | **geoconformal-main\Codes_Manuscript_GeoSIMCP\Codes_GeoSIMCP\GeoConformal\GeoConformalSpatialRegression.py** | **GeoConformalSpatialRegression**；**analyze**, **_kernel_smoothing**, **_weighted_quantile** | 计算空间核权重、加权残差分位数和 GeoCP 区间 |
| GeoCP（现代包） | **geoconformal-main\geoconformal\geocp\core.py** | **GeoConformalPrediction.conformalize** | 现代 GeoCP conformalization 主流程 |
| Spatial / joint weights | **geoconformal-main\geoconformal\geocp\weights.py** | **spatial_kernel_weights**, **joint_geo_feature_weights** | 生成空间核与空间—属性联合权重 |
| GeoSIMCP-EUC | **geoconformal-main\Codes_Manuscript_GeoSIMCP\Codes_GeoSIMCP\GeoConformal\GeoSIMCP.py** | **GeoSIMConformalSpatialRegression** | 欧氏属性距离 + 空间距离的联合局部校准 |
| GeoSIMCP-MND | **geoconformal-main\Codes_Manuscript_GeoSIMCP\Codes_GeoSIMCP\GeoConformal\GeoSIMCPzhao.py** | **GeoSIMCPzhao** | 最大归一化属性差异 + 空间距离的联合局部校准 |
| GCCP Seattle | **geoconformal-main\GCCP_Seattle\03_gccp\03_Seattle_GCCP_causal_similarity.ipynb** | **causal_feature_distance**, **evaluate_gccp_inner** | 使用 Seattle GCCM 权重构造因果属性距离，调节并评估 GCCP |
| GCCM PM2.5 | **geoconformal-main\GCCP_PM2.5\03_gccm\run_pm25_gccm_train_only_v1.R** | R 主流程 | 仅在训练数据上估计 PM2.5 属性的 GCCM causal scores |
| GCCP PM2.5 | **geoconformal-main\GCCP_PM2.5\04_gccp\pm25_gccp_b.ipynb** | **weighted_q_matrix** + grid cells | 计算空间核/因果核、加性混合权重及 PM2.5 GCCP-B 区间 |
| Simulation GCCM wrapper | **simulation_pilot_20260908\gccm_runtime\gccm_point_graph_wrapper_v1.R** | R wrapper main flow | 在模拟数据上运行 point-graph GCCM 因果强度估计 |
| CMI benchmark | **causal_strength_benchmark_v1\stage17_cmi_worker_v1.py** | **main** | 计算条件互信息、置换检验和特征级因果强度基准 |
| PM2.5 causal-only CMI | **causal_only_cp_v1\run_stage14_realdata_causalonly_cmi_exploratory_v1.py** | **run_cmi** | 对 PM2.5 绝对残差尺度计算 CMI、BH 筛选和回退判断 |
| Stage-19B direct parity | **geo_causal_cp_v1\run_stage19b_direct_parity_v1.py** | **fit_feature_transform**, **feature_distance_sq**, **weight_quantiles**, **tune_method** | 在 S2/S4 上统一比较 Raw-CP、GeoCP、GeoSIMCP 与 GeoCausalCP |

## Key data and result navigation

| Category | Path | Contents |
|---|---|---|
| Seattle data | **geoconformal-main\example\seattle_sample_3k.csv** | 3,000 条 Seattle 房屋样本 |
| Seattle GCCM scores | **geoconformal-main\GCCP_Seattle\outputs\gccm\seattle_gccm_causal_weights_outer_train_v1.csv** | 8 个房屋属性的 GCCM 分数与排序 |
| Seattle performance | **geoconformal-main\GCCP_Seattle\outputs\gccp\seattle_main_experiment_comparison_v1.csv** | CP 至 GCCP 的 Coverage、Width、IS |
| PM2.5 data | **geoconformal-main\GCCP_PM2.5\00_data_audit\china_sites_2018_05_PM2.5_all.csv** | 1,415 个站点样本 |
| PM2.5 GCCM scores | **geoconformal-main\GCCP_PM2.5\outputs\gccm\pm25_gccm_causal_weights_v1.csv** | 7 个属性的 GCCM 分数与排序 |
| PM2.5 CMI | **experiments\real_data\20260918_stage14_pm25_causalonly_cmi_exploratory_v1\cmi_feature_results.csv** | raw CMI、p/q 值与选择结果 |
| PM2.5 performance | **geoconformal-main\GCCP_PM2.5\outputs\pm25_formal_frozen_test_summary_v1.csv** | CP 至 GCCP-B 的冻结测试指标 |
| Simulation data | **simulation_pilot_20260908\data\S1–S4_observed.csv / S1–S4_truth.csv** | S1–S4 观测数据及生成真值 |
| Stage-19B results | **experiments\geo_causal_cp\20260919_stage19b_direct_geosimcp_parity_v1\stage19b_summary.md** | S2/S4 Track A/B 汇总和方法比较 |

## Reading order

1. 从 **GeoConformalSpatialRegression.py** 理解空间加权 conformal 基线。
2. 阅读 **GeoSIMCP.py** 与 **GeoSIMCPzhao.py**，比较 EUC 和 MND 属性距离。
3. 阅读 Seattle GCCP notebook，理解历史 GCCM 权重如何进入因果距离。
4. 阅读 PM2.5 GCCM R 脚本与 GCCP-B notebook，注意其采用加性核混合。
5. 阅读 Stage-14 CMI 与 Stage-17 CMI worker，理解 residual-scale CMI、置换检验及 BH 筛选。
6. 阅读 Stage-19B 脚本，理解当前较统一的 GeoSIMCP / GeoCausalCP parity 实验和 honest atom 问题。

