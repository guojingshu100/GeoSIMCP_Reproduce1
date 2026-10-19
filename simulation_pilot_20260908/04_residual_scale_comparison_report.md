# Residual-scale comparison 阶段报告

## 1. 本阶段结果

本阶段先审查历史 GCCM 对 S1--S4 的科学与计算适用性。结论是：历史 irregular-point graph workflow 在概念上可适配，但当前不能可复现地运行 learned GCCM。因此 `scale_comparison_check_v1.json` 的总体状态为 `BLOCKED`，而已获准完成的 shared non-GCCM preparation 状态为 `PASS`。

没有生成 GCCM weights，没有执行 EUC-versus-GCCM validation comparison，也没有计算任何 final-calibration 或 test 结果。没有用 correlation、feature importance、mutual information 或 regression weights 冒充 GCCM。

## 2. GCCM 适用性审查

当前 `s_x,s_y` 是独立 Uniform 坐标，样本是 irregular spatial points，不是规则 raster/lattice；每个位置只有一次横截面观测。历史 `GCCM.r::GCCM` 接收二维 raster matrices，因此不能直接使用。

历史 `GCCM4Lattice.r` 的 `generateEmbedings(neighborMatrix,x,E)` 与 `GCCMLattice(...)` 可以接收 graph adjacency。PM2.5 和 Seattle 项目均先在不规则点上构造 Delaunay graph，再把一个标量 field 的空间 rings 汇总成 embedding。因此，“一个位置一个观测”对项目已有 point-graph adaptation 是足够的；它依靠空间图邻域，而非每点时间重复。

当前无法实际运行有两个原因：

1. PATH 中找不到 `Rscript`，无法执行历史 R core，也无法核验 `foreach`、`doParallel` 等 packages；
2. `scale_fit` 只有 200 个点。历史 PM2.5/Seattle library grids 面向约 990/2088 点，不能原样使用；`E=3,tau=1,n=200` 的 terminal maximum 为 198，但中间 `lib_sizes` 尚未预注册。

完整证据和最小修复方案见 `scale_comparison/00_gccm_applicability_audit.md`。

## 3. GCCM cause/effect 定义

研究假设是 observed feature `X_j` 是否影响 residual scale target `A=abs_residual`。按历史项目采用的 CCM reverse mapping：

- cause field：scale-fit 上的 standardized `X_j`；
- effect field：同一批点上的 `abs_residual`；
- manifold：effect field `abs_residual` 的 graph-ring embedding；
- reconstruction target：cause field `X_j`；
- direction：`effect_xmap_cause`，即 `abs_residual xmap feature`。

若以后运行 learned GCCM，必须保留这一定义。把 `y`、predictor importance、普通 correlation 或 regression coefficient 当作 GCCM target/weight 都会改变科学问题。

## 4. 已冻结的共同协议

特征固定为 `s_x,s_y,x1,x2,g_proxy,x_noise`。所有分支共享同一个 `StandardScaler` 规则：只用 `scale_fit=200` 拟合，`with_mean=True,with_std=True`。Scale target 固定为未经变换的 `abs_residual`。

KNN local scale 使用邻居 `abs_residual` 的无权算术均值。候选为 `k=[10,20,30,50,75,100]`。Scale-fit diagnostics 使用按 `sample_id` 排除自身的 leave-one-out prediction。邻居按 squared distance 升序排列，距离相同时按 reference `sample_id` 升序排列，实际邻居数严格等于 k。

正 floor 只使用 scale-fit target：

`floor=max(float64 machine epsilon, linear 5% quantile of scale-fit abs_residual)`。

方法和 search budget 已记录为：

| 方法 | 候选数 | 冻结定义 | 当前状态 |
|---|---:|---|---|
| Raw-CP | 1 | scale=1，score=`abs_residual` | 代码准备完成；未跑 validation |
| Normalized-Global | 1 | scale=scale-fit mean absolute residual | 代码准备完成；未跑 validation |
| Normalized-EUC | 6 | shared standardized features 上普通 EUC，六个 k | scale-fit-only 准备完成；未跑 validation |
| Normalized-GCCM | 6 | 同一 KNN/floor/k，仅以 `gamma_j=p*w_j` 改变 feature distance | GCCM weights 被阻断，未运行 |

EUC 和 GCCM 的 local-scale search budget 均为 6。GCCM 的历史参数证据支持 `E=3,tau=1,b=4` 和 Delaunay point graph；`library_sizes` 在 config 中保持 `null/BLOCKED`，没有事后发明网格。

## 5. 预注册的 tuning 与 validation 规则

若 GCCM blocker 解除，`tuning_calibration=100` 将计算 `abs_residual/estimated_scale`，并以 `ceil((100+1)*(1-0.10))=91` 取得第 91 小的 provisional score。Validation interval 固定为 `[y_pred-q*scale,y_pred+q*scale]`。

目标 coverage 为 0.90，绝对 tolerance 已在查看 validation 结果前冻结为 0.02，即 eligible range `[0.88,0.92]`。Eligible candidates 中先最小化 mean Interval Score，再最小化 mean width，再选择较大 k。若无 eligible candidate，则依次最小化 absolute coverage deviation、Interval Score、mean width，再选择较大 k。

这些函数已经实现并通过单元测试，但本阶段没有加载 tuning-calibration 或 validation labels，也没有执行选择。

## 6. Shared non-GCCM preparation

每个场景只读取 200 个 scale-fit rows 的六个 observed features 和 `abs_residual`，保存一个 scaler 和一个 scale-fit preparation CSV。由于四场景 feature matrix 完全配对，四个 scaler 的 SHA-256 相同。

| 场景 | Scale floor | Global scale | Target min | Target median | Target max |
|---|---:|---:|---:|---:|---:|
| S1 | 0.073502 | 1.137131 | 0.003605 | 0.904925 | 5.217725 |
| S2 | 0.102705 | 1.134644 | 0.005225 | 0.921785 | 4.840855 |
| S3 | 0.070665 | 1.135253 | 0.020645 | 0.897406 | 5.769036 |
| S4 | 0.081174 | 1.134899 | 0.002321 | 0.980473 | 4.780624 |

Scale-fit leave-one-out EUC scale 与自身 `abs_residual` 的 non-oracle Pearson correlations 为：

| 场景 | k=10 | k=20 | k=30 | k=50 | k=75 | k=100 |
|---|---:|---:|---:|---:|---:|---:|
| S1 | 0.1701 | 0.1952 | 0.2218 | 0.1860 | 0.2327 | 0.2359 |
| S2 | 0.1735 | 0.1574 | 0.1772 | 0.0503 | 0.0912 | 0.1041 |
| S3 | 0.0388 | 0.0573 | 0.0283 | -0.0312 | -0.0251 | -0.0043 |
| S4 | 0.0674 | 0.0233 | 0.0109 | -0.1418 | -0.1200 | -0.1217 |

这只是 scale-fit LOO engineering diagnostic，不是 validation 结果，不用于选择 k。所有 24 组 KNN estimates 均 finite、strictly positive，actual neighbour count 恰为 k，floor activation count 全部为 0。完整 min/median/max、距离范围和 correlations 位于 `scale_comparison_check_v1.json`。

## 7. Validation 结果状态

S1--S4 的 validation coverage、mean/median width、Interval Score、normalized-score distributions 和 selected parameters 均为“未计算”。原因是 primary comparison 的 learned GCCM branch 尚不可运行。只运行 EUC 并查看 validation 后再决定 GCCM library grid 会破坏预注册顺序，因此本阶段主动停止。

Raw-CP 与 Normalized-Global 若使用同一个正 global constant，会在 symmetric split conformal 下代数上产生相同区间：normalized quantile 缩放会抵消 global scale。两者仍按要求保留为独立结构控制，但以后结果应明确报告这一等价性。

## 8. Leakage prevention

访问日志逐场景记录：feature loader 只选择 `local_calibration/scale_fit` 200 行，只加载元数据和六个 features；`y` 未进入 DataFrame。Target loader 只选择相同 200 行并只加载 `abs_residual`，没有加载 tuning-calibration 或 validation residual values。

本阶段没有读取或重新哈希 generator observed/truth CSV 的完整内容，没有读取或哈希任何 truth 文件。Final-calibration/test labels 未访问；oracle diagnostics 未运行。Predictor 输出 hashes 与上一阶段冻结记录全部一致。

## 9. 自动测试与复现

`test_scale_comparison_stage.py` 的 12 项测试全部通过，包括 exact subset access、prohibited-label isolation、oracle exclusion、scale-fit-only scaler、EUC/GCCM equal search budget、deterministic ties、positive scales、double-run hashes、91st conformal order statistic、alpha=0.10 interval construction、constant-sigma handling 和 coverage-constrained selection。

两次独立 temporary-directory runs 的四个 preparation CSV hashes、四个 scaler joblib hashes 和 manifest hash 全部相同。正式 source、config、audit、README 和 test hashes 已写入 check 文件。

## 10. 局限与警告

- `overall_status=BLOCKED` 表示 primary scientific comparison 尚未完成；non-GCCM preparation 本身为 `PASS`。
- 尚未确认 R packages，也没有执行 GCCM smoke test。
- 200 点 GCCM library grid 仍是明确的预注册空缺。
- 当前没有 validation 方法排名，也没有证据表明 Normalized-EUC 或 GCCP 有效。
- 没有 oracle scale diagnostics；S4 constant-sigma 的 test 只用人工常数向量验证返回 undefined/null 的处理逻辑。
- 只有一个 paired seed，未来即使 pilot 完成也不能替代 Monte Carlo uncertainty assessment。

## 11. 精确下一任务

下一任务是恢复或定位一个可调用的 `Rscript`，只做历史 `GCCM4Lattice.r` 及 `foreach/doParallel` 的只读 smoke test；在不打开 tuning/validation labels 的前提下，由研究者预注册适合 `n=200,E=3,tau=1` 的完整 `lib_sizes`。随后编写版本化 R wrapper：仅接收 scale-fit Delaunay graph、effect field `abs_residual` 和六个 standardized cause fields，保存原始 rho curves 与 convergence diagnostics。只有 wrapper 和 weights 通过复现校验后，才同时运行冻结的 EUC/GCCM/Raw/Global pilot validation comparison。
