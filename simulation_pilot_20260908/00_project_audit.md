# GCCP 项目接手与生成器审查

## 1. 审查范围与结论

本报告仅依据 `F:\GeoSIMCP_Reproduce` 内的现有文件和只读检查结果。审查过程中没有执行历史 notebook、没有重新训练模型、没有读取后重新评估或调参 PM2.5 Frozen Test，也没有安装或升级依赖。

项目内未找到 `AGENTS.md`，也未找到文件名包含 `HANDOFF` 的文档。根目录现有的 `GCCP_文件清单_20260908_132601.txt` 是文件清单，不含研究交接说明。通用包说明位于 `geoconformal-main/README.md`，主要介绍 GeoCP、GeoSIMCP 和 GeoBCP，并不是本次 GCCP 模拟实验的专门协议。

核心结论如下：

1. PM2.5 的 baseline、GCCM、GCCP-B、冻结配置和最终测试结果均存在，且 notebook 保存输出显示这些阶段确实运行过。
2. 报告中的历史数值可以从 `pm25_formal_frozen_test_summary_v1.csv` 精确核实：Additive-EUC Interval Score 为 `26.930547520983055`，GCCP-B 为 `25.5196755953003`。
3. 预期的四场景模拟生成器不存在。`GCCP_Simulation/00_design/simulation_design.ipynb` 只有一个空代码单元；`GCCP_Simulation_Step1.ipynb` 不存在，其余模拟目录为空。
4. 因生成器缺失，无法从项目确认 S1--S4 的方程、观测变量、oracle truth、种子和实际 split。本轮没有生成模拟数据，也没有创建 `generator_check.json`。
5. 历史 PM2.5 GCCM 权重是针对 detrended `PM2.5` 学得的，并非针对 `|residual|` 或其他误差尺度目标学得。这一点对当前研究问题非常关键。

## 2. 重要目录和文件

| 类别 | 路径 | 作用与状态 |
|---|---|---|
| 通用代码 | `geoconformal-main/geoconformal/` | 当前发布包的 GeoCP、GeoSIMCP、GeoBCP 实现。并非 PM2.5 GCCP-B notebook 的唯一实现来源。 |
| 作者旧代码 | `geoconformal-main/Codes_Manuscript_GeoSIMCP/` | 历史论文代码和大量 notebook；包含 GeoCP/GeoSIMCP 的旧实现与旧模拟，但不是本次指定的 S1--S4 生成器。 |
| PM2.5 输入数据 | `geoconformal-main/GCCP_PM2.5/00_data_audit/china_sites_2018_05_PM2.5_all.csv` | 实际输入表，1415 行、24 列，含站点、坐标、PM2.5 和环境特征。 |
| PM2.5 数据审查/作者复现 | `GCCP_PM2.5/00_data_audit/pm25_data_audit.ipynb`、`01_author_reproduction/pm25_author_reproduction.ipynb` | 两个文件当前大小和 cell 结构相同，均保存了 6 个已执行 cell；包含作者式 RF/GeoCP/GeoSIMCP 复现。 |
| PM2.5 baseline | `GCCP_PM2.5/02_baselines/pm2.5_formal_baselines.ipynb` | 14 个 cell，前 13 个有 execution count 和输出；建立 formal split、RF、CP、GeoCP、GeoSIMCP-EUC/MND、Additive-EUC 和 Joint-HAV-EUC。 |
| PM2.5 GCCM | `GCCP_PM2.5/03_gccm/pm25_gccm_causal.ipynb` | 前 3 个 cell 已执行；构造 Train-only Delaunay 图、空间去趋势、调用 R GCCM、形成因果权重。 |
| GCCM R wrapper | `GCCP_PM2.5/03_gccm/run_pm25_gccm_train_only_v1.R` | 保存下来的 R runner；当前硬编码旧 `E:` 路径。 |
| GCCM R 核心 | `GCCP_Seattle/02_gccm/GCCM_updated 2/R/GCCM/GCCM4Lattice.r` | 提供 `GCCMSingle4Lattice`、`generateEmbedings`、`GCCMLattice`、`simplex_projection` 等函数。 |
| PM2.5 GCCP-B | `GCCP_PM2.5/04_gccp/pm25_gccp_b.ipynb` | 8 个 cell，前 7 个已执行；因果距离审查、validation 搜索、test 前冻结、最终一次性 Frozen Test 和事后诊断。 |
| PM2.5 结果 | `GCCP_PM2.5/outputs/` | 61 个 CSV、3 个 JSON、2 个 TXT；`outputs/gccm/` 保存 GCCM 曲线/诊断/权重，`outputs/gccp/` 保存 GCCP-B validation 搜索与几何诊断。 |
| Seattle 研究分支 | `geoconformal-main/GCCP_Seattle/` | 更早的 Seattle baseline/GCCM/GCCP 实验和结果，可作为 GCCM 工程参考，不能当作 S1--S4 生成器。 |
| 模拟目录 | `geoconformal-main/GCCP_Simulation/` | 仅有 `00_design` 至 `04_gccp` 和 `outputs` 的目录骨架；没有可运行生成器或数据。 |
| 环境记录 | `geoconformal-main/requirements_geosimcp_repro.txt` | 历史环境冻结，含失效的 editable 路径 `-e e:\geosimcp_reproduce\geoconformal-main`。 |

## 3. 已实现、已运行与未来工作

### 3.1 已实现且有运行证据

- `pm2.5_formal_baselines.ipynb` 的 STEP 3--11 已保存输出。它创建并验证四分 split、训练 `RandomForestRegressor`、计算 calibration absolute residual、实现 weighted quantile、有效样本量、Interval Score，并完成多种 validation grid。
- `pm25_gccm_causal.ipynb` 的输出记录 `R return code: 0`、`Raw rows: 69174`、`Mean rows: 84`，证明旧电脑上 GCCM 实际运行成功。
- `pm25_gccp_b.ipynb` 的输出记录 GCCP-B validation、冻结 manifest、七种方法在访问 test labels 前已构造完成，以及最终一次性读取 142 个 test labels。
- `pm25_formal_frozen_test_completion_v1.json` 明确记录状态 `FINAL_FROZEN_TEST_COMPLETED`，方法为 CP、GeoCP、GeoSIMCP-EUC、GeoSIMCP-MND、Joint-HAV-EUC、Additive-EUC、GCCP-B。

### 3.2 已保存的历史结果

`pm25_formal_frozen_test_summary_v1.csv` 中的最终测试结果为：

| 方法 | Coverage | Mean Width | Interval Score | Median n_eff |
|---|---:|---:|---:|---:|
| CP | 0.894366 | 18.267214 | 27.221021 | 141.000000 |
| GeoCP | 0.873239 | 17.553033 | 25.593470 | 81.720548 |
| GeoSIMCP-EUC | 0.866197 | 16.959461 | 26.972864 | 83.059654 |
| GeoSIMCP-MND | 0.866197 | 17.272900 | 25.658859 | 69.927360 |
| Joint-HAV-EUC | 0.866197 | 16.960444 | 26.973847 | 82.512900 |
| Additive-EUC | 0.859155 | 17.021409 | 26.930548 | 80.072098 |
| GCCP-B | 0.866197 | 17.121002 | 25.519676 | 64.046931 |

因此用户提供的约 `26.9305` 与 `25.5197` 均得到精确验证。GCCP-B 相比 Additive-EUC 的 test Interval Score 低约 `1.410872`，coverage 高约 `0.7042` 个百分点，mean width 高约 `0.099593`。这只是一个冻结数据划分上的结果。两种方法都没有达到 0.90 的名义覆盖率，不能据此声称 GCCP 必然优于控制方法。

Validation 上也不存在 coverage `>= 0.90` 的 GCCP-B 配置；保存结果采用 `fallback_global_min_interval_score`。Additive-EUC validation coverage 为 `0.823944`、Interval Score 为 `31.899938`；GCCP-B 分别为 `0.852113` 和 `31.307393`。

### 3.3 已准备但未执行的代码

未发现针对本次 S1--S4 设计的已准备代码。`simulation_design.ipynb` 不是“已写好但未运行”，而是空 notebook。`GCCP_Simulation` 其他目录为空。

PM2.5 三个主要 notebook 的末尾存在空 cell，但其主要历史流程已经执行。项目内未找到 normalized residual-scale conformal 的实现；搜索不到 `normalized score`、`local scale`、`residual scale` 或 `sigma_hat` 对应实现。

### 3.4 仅为拟议的未来工作

S1--S4、五分数据角色、normalized scale estimator、oracle scale 诊断和 Monte Carlo 比较只出现在本次任务说明中，当前项目文件没有对应实现或冻结配置。它们应视为待实现研究设计，而不是既有实验。

## 4. PM2.5 历史方法和数据流

### 4.1 数据与 split

输入表有 1415 个唯一站点。所用字段为 `PM2.5`、`dem`、`sp`、`tp`、`blh`、`r`、`aod_sat`、`ndvi`、经纬度以及 `proj_x/proj_y`。这些字段在当前 CSV 中均无缺失且数值有限。

`pm25_formal_protocol_v1.json` 冻结 `seed=42` 和四分 split：

- Train 990；
- Calibration 141；
- Validation 142；
- Frozen Test 142。

`pm25_formal_split_assignment_v1.csv` 的 1415 个 `original_index` 不重复，恰好覆盖 `0..1414`，四组数量与协议一致。按 notebook 中的字符串规则重新计算，split SHA256 为 `785436f18bcc08a5a6b94916d3e544af0e0706e959b7c930300f7a91bddf04f3`，与协议一致。

这个历史 split 没有独立的“local calibration”和“final calibration”：同一个 Calibration residual pool 用于各局部 weighted quantile，Validation 用于参数选择，随后 Frozen Test 用于最终评估。它与拟议的五分 `1000/300/200/200/300` 不同。

### 4.2 predictor 与 nonconformity score

正式 predictor 为 `RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)`，输入为 7 个环境变量加 `lon/lat`。代码先在 Formal Train 内做 3-fold CV 选择，再固定该配置。Calibration nonconformity score 是：

`a_i = |y_i - formal_rf.predict(X_i)|`。

历史 local conformal level 为 `k=ceil((141+1)*(1-0.1))=128`，`tau=128/141=0.9078014184`。

### 4.3 GeoCP、GeoSIMCP、GCCM 与 GCCP-B

- GeoCP 用 test/validation 点到 Calibration 的地理距离和 Gaussian kernel，对 Calibration absolute residual 做局部 weighted quantile。
- GeoSIMCP 把地理距离与普通 feature distance 组合。formal notebook 同时实现 EUC 与 MND；部分作者式 GeoSIMCP 使用 raw lon/lat degree，而严格 Additive/Joint 控制使用 Haversine km，应避免混称。
- PM2.5 GCCM 只使用 Formal Train。代码以 `proj_x/proj_y` 建 Delaunay graph，取 Ring1--Ring3，对 detrended `PM2.5` 和七个 detrended feature 运行 `GCCMLattice`。冻结参数为 `E=3`、`tau=1`、`b=4`、library sizes `[10,50,100,200,400,988]`。
- GCCM 的预设方向为 `effect_xmap_cause`，即用 PM2.5 manifold 重构候选 X。通过 gain、Spearman、terminal Pearson p-value 和 Fisher-z CI 条件后，terminal rho 归一化为 `causal_weight`，再令 `gamma_k = 7 * causal_weight_k`。
- GCCP-B 首先在 Calibration-only `StandardScaler` 空间构造
  `d_causal(i,j)=sqrt(sum_k gamma_k*(z_ik-z_jk)^2)`，再使用
  `W=lambda*K_geo+(1-lambda)*K_causal`，对同一组 Calibration residual 做 weighted quantile。
- Additive-EUC 使用相同的加法架构，但把 `K_causal` 换成普通标准化 Euclidean feature kernel。它是当前历史比较中最直接的控制。

全部七个 PM2.5 feature 在保存的 GCCM 规则下都通过 convergence。权重最高的是 `aod_sat`（`causal_weight=0.287181`，`gamma=2.010264`），最低是 `ndvi`（`0.048376`，`0.338629`）。

关键科学边界：这里 GCCM 的 effect 是 detrended `PM2.5`，不是 absolute residual 或 noise scale。因此历史 GCCP-B 检验的是“目标相关因果权重能否帮助局部化 prediction error quantile”，它没有证明同一组变量解释 residual magnitude。对当前研究问题，新的 primary GCCM 必须明确以误差尺度目标为学习对象，或者把 outcome-target GCCM 清楚标成单独的 secondary diagnostic。

## 5. 四场景生成器检查

指定文件检查结果：

| 预期文件 | 结果 |
|---|---|
| `pm2.5_formal_baselines.ipynb` | 存在，位于 `GCCP_PM2.5/02_baselines/`。 |
| `pm25_gccm_causal.ipynb` | 存在，位于 `GCCP_PM2.5/03_gccm/`。 |
| `pm25_gccp_b.ipynb` | 存在，位于 `GCCP_PM2.5/04_gccp/`。 |
| `simulation_design.ipynb` | 存在，但仅 628 字节，只有一个空 code cell，无输出。 |
| `GCCP_Simulation_Step1.ipynb` | 项目内不存在。 |

所以以下内容均无法从当前项目确认：

- S1--S4 的数学方程；
- observed variables 的名称和生成关系；
- hidden/oracle truth 的字段；
- 样本量是否为 2000；
- generator seed、replicate seed 和 split seed；
- 是否已经采用 `1000/300/200/200/300`；
- 跨场景是否使用相同 `sample_id`、相同 split 和 paired random numbers；
- noise scale 是否严格为正；
- observed data 与 oracle truth 的保存边界。

项目中的旧 GeoSIMCP notebook 含其他 synthetic grid/noise 示例，但其设计不是指定的 S1--S4。根据任务约束，本轮没有拿它替代缺失生成器。

## 6. 冻结配置完整性

`pm25_formal_frozen_test_manifest_v1.json` 保存了 split、predictor、各方法参数、GCCM 权重和 test policy。保存的 hash 文件及 completion JSON 记录 manifest SHA256 为 `d2090944...d7553`。

当前 Windows 文件的原始字节 SHA256 是 `c07f73d3...e69637`，表面上不匹配。检查显示该 JSON 当前为 CRLF 行尾；仅把 CRLF 规范化为 LF 后，SHA256 正好恢复为 `d2090944...d7553`。所以逻辑内容和历史冻结 hash 可由换行转换解释，但严格 byte-for-byte 验证在当前副本上失败。后续报告应同时记录“原始字节 hash 不匹配”和“LF 规范化后匹配”，不能直接写成无条件 hash verified。

## 7. 当前电脑的执行就绪情况

### 7.1 Python

当前可用解释器是 `D:\conda\python.exe`，版本 Python 3.13.5。当前可导入：

- `numpy 2.1.3`
- `pandas 2.2.3`
- `scipy 1.15.3`
- `scikit-learn 1.6.1`
- `matplotlib 3.10.0`
- `jupyter`
- `nbformat 5.10.4`

当前不能导入 `geopandas` 和 `xgboost`。从项目根包导入 `geoconformal` 会因 `geoconformal/utils.py` 顶层 `import geopandas as gpd` 而失败。

历史 notebook kernelspec 是 `Python (geosimcp_repro)`，记录的 Python 是 3.10.20；当前 Jupyter 只列出 `python3` kernel，Conda 环境列表中没有 `geosimcp_repro`。`requirements_geosimcp_repro.txt` 与当前包版本也不完全相同，并包含旧 `E:` editable install 路径。

对于一个只依赖 NumPy/Pandas/SciPy/sklearn 的新 generator，现有 Python 组件可能够用；但由于 generator 不存在，无法验证其真实 imports。

### 7.2 R/GCCM

当前 `PATH` 中没有 `Rscript`，`C:\Program Files\R` 也不存在。历史 notebook 指定 `C:\Program Files\R\R-4.6.1\bin\Rscript.exe`，当前路径失效。因此现有 R GCCM 无法在本机直接重跑。项目内能找到 R 源码和 wrapper，但无法确认 `parallel`、`foreach`、`doParallel` 等 R 包状态。

### 7.3 绝对路径

三个 PM2.5 notebook 和保存的 R runner 广泛硬编码 `E:\GeoSIMCP_Reproduce\geoconformal-main`，当前实际位置是 `F:\GeoSIMCP_Reproduce\geoconformal-main`，且旧 `E:` 根不存在。按本次要求没有修改这些路径。

## 8. 第一轮比较可复用组件

可复用但应从 notebook 抽成独立、可测试函数的组件包括：

- `interval_score`：位于 baseline 和 GCCP-B notebook，公式实现一致；
- `raw_gaussian_kernel`、`stable_gaussian_weights`：距离到 Gaussian weight；
- `weighted_quantile_harmonized`、`weighted_q_matrix`：weighted residual quantile；
- `neff_matrix_stable`：局部权重有效样本量；
- Haversine 距离、Calibration-only `StandardScaler` 和 GCCM-weighted Euclidean distance 构造；
- Delaunay 邻接、Ring1--Ring3 和线性空间 detrending 的 STEP 12A 逻辑；
- R 中的 `generateEmbedings`、`GCCMLattice`、`GCCMSingle4Lattice` 和 `simplex_projection`；
- GCCM convergence diagnostics 中的 `pearson_r_pvalue`、`fisher_z_ci` 以及冻结判据。

不能直接复用为“新 GCCP”的内容：历史 PM2.5 的手工保存 `gamma`、针对 PM2.5 outcome 学到的权重、旧 test 结果，以及旧 synthetic notebooks。Oracle 权重也只能用于诊断，不能冒充 learned GCCM。

## 9. 当前最小阻断与下一步

最小且首要的阻断是缺少 S1--S4 generator 及其数学设计，而不是环境迁移。没有方程就无法科学地确认 observed/hidden 变量、噪声尺度、proxy 机制和 pairing，也不能合法生成 `generator_check.json`。

下一步应先恢复或编写唯一的四场景 generator：把 S1--S4 的明确方程、参数、observed/oracle 字段、seed 派生规则和五分 split 写入冻结 config；只生成一个小型 pilot replicate，并执行有限值、正尺度、split、ID pairing 和 oracle separation 检查。完成这一关后，再实现 normalized-EUC 与 learned-GCCM scale comparison。
