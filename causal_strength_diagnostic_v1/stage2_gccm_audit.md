# Stage 2 GCCM 调用链与空间 embedding 审计

## 证据表

|审计点|文件 / 函数|行号与证据|
|---|---|---|
|生成|`simulation_pilot_20260908/generator/generate_scenarios.py` / `create_base_random_variables`、`generate_scenario`|46-70、119-177|
|生成配置|`simulation_pilot_20260908/generator/config_v1.json`|11-24、31-48|
|scale-fit 特征|`simulation_pilot_20260908/scale_comparison/fit_local_scales.py` / `load_scale_fit_features`|231-270|
|scale-fit 目标|同上 / `load_scale_fit_target`|273-310|
|对齐与标准化|同上 / 主循环|389-430|
|Python→R|`simulation_pilot_20260908/gccm_runtime/run_gccm_scale_fit_v1.py` / `prepare_input`、`invoke`|36-70|
|R wrapper|`simulation_pilot_20260908/gccm_runtime/gccm_point_graph_wrapper_v1.R`|7-35、47-75|
|历史核心|`geoconformal-main/GCCP_Seattle/02_gccm/GCCM_updated 2/R/GCCM/GCCM4Lattice.r`|1-38、43-124、128-203、256-318|
|核心 hash|`simulation_pilot_20260908/gccm_runtime/run_gccm_scale_fit_v1.py`；`gccm_runtime_config_v1.json`|24-25、46-48|

## 1. 生成器语义：CONFIRMED

`generate_all_scenarios` 用配置中的 base seed 先生成一次共同变量，再把同一 base 和冻结 split 传入各场景（`generate_scenarios.py:180-190`）。`create_base_random_variables` 分别调用同一个 RNG 生成 `s_x,s_y,x1,x2,eta,x_noise,epsilon`，随后才计算 `g_true`（坐标函数）和 `g_proxy`（:46-70）；没有重采样、排序或变换把 x2/epsilon 接到坐标或共享空间场。`generate_scenario` 直接取 base 的 x2、epsilon（:130-133），S2 为 `mu_true=2*x1`、`sigma_true=h(x2)`（:138-140），S4 为同一均值、`sigma_true=1.3`（:144-148），只形成 `y=mu_true+sigma_true*epsilon`。因此代码层面 x2、epsilon 与坐标、彼此及 g_true 的生成过程独立；S4 的常数尺度也没有后续改变。配置 wiring 与这些分支一致（`config_v1.json:11-24,31-48`）。

## 2. GCCM 输入边界：CONFIRMED

`load_scale_fit_features` 先从冻结预测的元数据筛出 `local_calibration`/`scale_fit`，断言恰好 200 行，再从 observed 文件只读 `META_COLUMNS+FEATURES`，明确排除 y 与 oracle 列（`fit_local_scales.py:236-270`）。`load_scale_fit_target` 对授权残差做同样过滤，只读元数据与 `abs_residual`（:277-310）。主循环逐列比较两者元数据并断言一致（:389-406），所以 ID、特征、目标逐行对应。进入 GCCM 前仅对六个特征做 `StandardScaler`；target 原样保留为未缩放 `abs_residual`（:407-429）。runner 读取准备 CSV 的 200 行并按 ID 排序，坐标从冻结 scaler 逆变换得到，和同序目标/特征组装（`run_gccm_scale_fit_v1.py:41-60`）；R 输入断言 200 行及非负 target（wrapper:16-22）。三角剖分只使用坐标，embedding 只使用该输入中的 target，未导入其他 split；调用链的输入边界成立。

## 3. 方向与历史核心：CONFIRMED

runner 验证固定核心文件并复制为 `historical_core.r`（`run_gccm_scale_fit_v1.py:24-25,62-70`），对应 `GCCM4Lattice.r` SHA-256 `3552d871...38b12d`，配置也记录该 hash（`gccm_runtime_config_v1.json:46-48`）。wrapper 对每个特征调用 `GCCMLattice(emb, dat[["z_feature"]], ...)`（:47-55）。因此映射为：**输入 cause = z_feature；输入 effect = emb；embedding 变量 = `abs_residual` 的空间邻域 embedding；重构变量 = z_feature；选用 rho 字段 = `raw$rho` 的 library-size 平均值；feature weight = terminal strength 归一化值**。这是代码实际传参与核心注释（核心:130-136）共同支持的映射，不对方向做反转。

## 4. Embedding 与权重：CONFIRMED；适用性为谨慎解释

`laggedVar4Lattic` 从邻接表扩张路径，精确第 1、2、3 阶分别去掉上一阶集合及自身（核心:43-100）；所以 embedding 不含焦点自身，但高阶邻域会包含跨越路径可达的点，且每一阶可能是去重后的 exact ring。`generateEmbedings` 对每个 ring 取 `x[neighbors]` 的算术均值（:102-124）。邻接由 Delaunay 三角形边构造（runner:54-61）。核心的 `GCCMLattice` 把 `lib/pred=c(1,n)` 转成从 `1+(E-1)tau` 到 n 的可用区间（:142-171）；每个 prediction 点在查找近邻前临时从 library 移除（:269-301），避免同点 self match。library size 小于最大值时使用循环窗口，可能与 prediction 集重叠，但 focal 行仍被排除；高阶 embedding 值可通过邻居均值间接带入邻域信息。核心没有 detrending 步骤，传入的是已标准化 feature 与未缩放 residual embedding/target。

wrapper 的收敛规则为 gain>0、Spearman>=0.60、双侧 p<.05 且 Fisher-z CI 下限>0（wrapper:57-65）；terminal rho 作为 converged strength，权重为 strength/总 strength，`gamma=6*weight`（:67-75）。总 strength 为零时输出 `NA` weight、status=`NO_CONVERGED_FEATURES`，不使用均匀替代（:69-71；runner:109,145-147）。由于 x2 在生成器中没有空间结构，S2 的同位置 x2 值不会进入该点的 residual embedding；embedding 保留的是其他位置 x2 相关特征的邻域平均可能性，而不是 same-location x2/residual 配对关系。该事实说明 embedding 对“同位置尺度驱动”的表达存在理论错位风险，但没有实验依据把它认定为失败原因。

## 结论状态

- 生成器语义：CONFIRMED。
- 输入边界、200 个 scale-fit ID、对齐：CONFIRMED。
- 方向、核心路径与 hash：CONFIRMED。
- 空间 embedding 与权重规则：CONFIRMED；其对 S2 同位置关系的科学适用性需实验验证。
- 实际实现错误：未建立。当前发现是设计/解释层面的适用性风险，不足以指认 bug。
- S4 oracle-residual null：生成器支持“真实噪声尺度恒为 1.3”的 null；但冻结模型绝对残差还包含预测误差，不能仅凭生成器确认 `abs_residual` 的 null。故对 oracle-residual null 的完整表述为 **UNVERIFIED**。

## 后续最小 CMI adapter 接口

可复用 `load_scale_fit_features` 与 `load_scale_fit_target`，在同一场景分别取得 `META_COLUMNS=(sample_id,scenario,split,local_calibration_role)`、六个 `FEATURES=(s_x,s_y,x1,x2,g_proxy,x_noise)` 与 `abs_residual`（`fit_local_scales.py:29,231-306`）。adapter 至少应保留 `sample_id` 对齐键、六个 raw 或对应 `z_` 特征列和 `abs_residual`；坐标列 `s_x,s_y` 仅在需要空间条件时保留。不得从 observed 重新读取 y 或其他 split 标签。
