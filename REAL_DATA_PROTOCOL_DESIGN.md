# GCCP v3 real-data protocol design

日期：2026-10-12。版本：real-data design v1。

**方法定位冻结为 residual distribution driven spatial conformal calibration。** 主方法名称保留 ΦR-SLCP，ΦR 表示冻结预测器的条件残差分布画像；不作 causal semantic gain、因果效应识别或因果迁移优势声明。

本轮只完成方法及 Seattle / PM2.5 协议设计：不训练模型、不生成实际拆分、不计算真实数据性能、不执行正式实验，不修改任何已有实验、报告或结果。已读取历史说明、协议、notebook 源代码片段和 CSV 表头；未读取历史逐样本测试结果来选择本协议。

## 1. 研究问题、已有证据与真实数据状态

研究问题是：**条件残差分布相似性是否比地理/普通属性相似性，更有助于在可接受覆盖下改善局部预测区间的 width 与 Interval Score？** 不是验证因果语义。

依据：

- [v3 formal report](experiments/gccp_v3_formal_validation_20261012_v1/REPORT_zh.md)：相对固定 GeoSIMCP，S2/S3 通过预定联合效率门槛；S1/S4 不通过；S6 机制漂移覆盖失败。
- [causal value smoke](experiments/gccp_v3_causal_value_smoke_20261012_v1/REPORT_zh.md)：同信息的普通 residual profile 与 ΦR 输出一致，没有可分离的因果增量；经验覆盖与解析条件平均覆盖也不能混为一谈。
- [论文定位更新](experiments/gccp_v3_causal_value_smoke_20261012_v1/PAPER_POSITIONING_UPDATE_zh.md)：主线调整为任务对齐的 score-distribution geometry 与有效校准组合。

本地数据与适用任务如下，行数/可用字段在未来执行前仍须以不依赖标签值的审计复核。

| 数据 | 本地来源与已知结构 | 本协议任务 | 不支持的外推 |
|---|---|---|---|
| Seattle | `geoconformal-main/example/seattle_sample_3k.csv`；3k 房屋横截面样本；八个住房属性、UTM_X、UTM_Y、log_price；当前表头没有交易日期/parcel ID | 未观测房屋的横截面 log-price 区间预测；另报美元区间 | 未来交易时间预测、跨年份市场变化、已证实的同一房屋纵向泛化 |
| PM2.5 | `geoconformal-main/GCCP_PM2.5/00_data_audit/china_sites_2018_05_PM2.5_all.csv`；历史协议为 1415 站点、2018-05-31 单日快照 | 利用同日可用环境协变量进行未观测站点浓度的空间估计 | 多日预报、长期时间迁移、处理效应或污染成因识别 |

读取规范：Seattle 使用 UTF-8，首个未命名列只映射为原始行 ID，不作为预测特征；PM2.5 使用已核对表头的 GB18030，`监测点编码`保留为字符串站点 ID，`经度/纬度`映射为 lon/lat，`date`保留作单日审计。ID、编码与坐标有效性不通过时记录数据适配失败，不尝试用结果反推正确 schema。

**历史标签暴露约束：** [既有 feasibility README](experiments/real_data/20260918_stage14_realdata_feasibility_v1/README.md) 及其审计源码已记录两套历史 evaluation labels 暴露。对同一批数据重新分配 seed，不能将它们重新称作 never-accessed holdout。复用本地数据只能标为 **retrospective external-data validation / exploratory**，即使本轮运行严格遵守标签隔离。

真正未暴露的验证需要另一个有明确来源、ID、时间及 exposure history 的数据批次，在读取其标签前封存协议。当前没有核实存在这样的批次，不能承诺两套真正全新的确认性验证。本文同时给出可复用的隔离程序；不会为了得到“确认性”标签自动下载或更换数据。

## 2. 最终统计方法定义：ΦR-SLCP

### 2.1 预测目标与 score 单位

令 Z=g(Y)，点预测器 f̂(X,S) 只在 D_fit 上拟合。Seattle 的 g(Y) 直接使用文件中的 `log_price`；按历史源代码预定以 10^Z 还原美元价格，但 log 底和原始币种必须在执行前的数据字典中确认；无法确认时，美元辅助任务标为不可执行，不能猜测转换。PM2.5 使用原始浓度 Z=Y，单位须在元数据审计中确认为 µg/m³；若无法确认，停止并记录。

定义仅由 D_fit 标签确定的固定正尺度

\[
c=\max\{\operatorname{IQR}(Z_{\mathrm{fit}}),10^{-6}\},
\qquad E=(Z-\hat f(X,S))/c,\qquad R=|E|.
\]

IQR 使用线性插值的 .25/.75 分位数定义。c 对全方法、全拆分固定；不按查询或 test 残差重新缩放。它仅使固定 CDF 网格具有可比较的数值单位。最终半径乘回 c，所有主要结果在原 Z 单位评价；不跨数据集混合不同量纲的 width/IS。

### 2.2 条件残差分布画像

\[
\Phi_R(x,s)=\widehat F_R(\cdot\mid x,s),
\quad F_R(r\mid x,s)=P(|E|\le r\mid X=x,S=s),
\]

画像保留完整 CDF 水平，能够反映冻结预测器的偏差、噪声尺度及其模型可表达的空间异质性。它不是 treatment-effect difference。

**实数据画像估计器预定为 Gaussian signed-residual location-scale model**，详见第 4 节。这沿用模拟中的 Gaussian / folded-normal 分布族；扩展的是条件特征基与真实预测器输入接口，不假设真实误差确实正态。画像失配可能损害效率/局部覆盖，独立外层在交换性条件下的边际有效性不要求画像模型正确。

### 2.3 距离、权重和 ESS 规则

采用原 v3 网格 r_k=12k/2400，k=0,…,2400，

\[
d_R(i,j)=\max_k|\widehat F_R(r_k\mid x_i,s_i)-\widehat F_R(r_k\mid x_j,s_j)|,
\qquad w_j(x;h)\propto\exp[-d_R(x,j)^2/(2h^2)].
\]

实数据不人为构造 treatment；只有一个活动校准层 H≡0。所有 local 样本均可参与权重。空间位置可以影响画像估计器，但主方法不额外加入地理 kernel，保持 gamma_spatial=0。

候选 h 固定为 {.025,.05,.1,.2,.4}。归一化 p_j=w_j/Σw_j，

\[
\mathrm{ESS}=1/\sum_jp_j^2,\quad
p_{\max}=\max_jp_j,\quad B_R=\sum_jp_jd_R(x,j).
\]

选择最小可行 h，要求 ESS≥50、pmax≤.05、B_R≤.10，容差 1e−12。全部不可行时，保留 B_R 最小候选，首个候选打破平局并标记 low information；不合并、不删除、不扩展候选，不通过 outcome 调整 h。

这不是风险控制证书。估计画像的 B_R 不是未知真实 score CDF 的严格误差界。网格误差和拟合 Gaussian 尾部质量单独记录，不因 test 诊断而扩大网格。

### 2.4 局部分位数与独立 outer correction

\[
q_0(x,s)=Q_{.9}\!\left(\sum_{j\in D_{\mathrm{local}}}p_j(x,s)\delta_{R_j}\right).
\]

使用原稳定排序与累计权重首次达到 .9 的分位数规则；local learning 不加 infinity self atom。

冻结 q₀ 后，使用独立 D_cal：

\[
U_i=R_i-q_0(X_i,S_i),\quad
k=\lceil .9(n_{\mathrm{cal}}+1)\rceil,\quad
\Delta=U_{(k)},\quad q(x,s)=\max\{0,q_0(x,s)+\Delta\}.
\]

k>n_cal 时保留 Δ=∞。允许负 Δ；不将 Δ 限制为非负，不根据 test coverage 扩大区间。Z 尺度区间为

\[
C_Z(x,s)=[\hat f(x,s)-cq(x,s),\ \hat f(x,s)+cq(x,s)].
\]

无局部样本时沿用冻结模型的 .9 folded-normal 分位数回退并标记；模型不可用/非有限则报告实现失败，不用 test 信息补救。最低数值支持规则为 n_profile≥50、n_local≥1、n_cal≥9；n_local<50 必须如实标为不能满足原 ESS 门槛，不临时降低门槛。

若复用模拟的二元 T 接口，以全零标签表示唯一活动层；未使用的臂 1 只是接口兼容项，不是处理组。不能把未使用臂的空样本数计作有效校准组的最小 n，也不能把其未被任何查询采用的 infinity 当作实际区间 infinity。

### 2.5 冻结边界与覆盖条件

统计核心保持：连续 CDF 几何、原权重/ESS 决策、local quantile learning、独立秩校正。本协议新增并冻结的是**实数据适配规范**：真实 X/S、固定 score 单位、单活动层和条件特征基。没有修改历史 `scientific.py` 或旧实验；不宣称其针对模拟 SCM 的全部源代码可无修改直接处理真实表。

条件于独立学习阶段，若 D_cal 与查询 score U 可交换，则原 outer rank argument 给出边际 coverage≥.90。它不保证每个位置、固定已实现 calibration 集或未知 shift 下的 .90。空间阻隔不能自动建立独立性；存在空间依赖或分布差异时，真实数据覆盖是经验验证结果。

独立分位数学习与 conformal correction 有已有方法基础，不作为新覆盖定理声称；参见 [Romano 等，Conformalized Quantile Regression](https://proceedings.neurips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html)。关于交换性被破坏时不能直接沿用原保证，参见 [Barber 等，Conformal prediction beyond exchangeability](https://arxiv.org/abs/2202.13415)；本协议没有实现该文的非交换性修正。

## 3. 五个互斥数据角色与固定拆分

### 3.1 数据角色

| 集合 | 目标比例 | 允许使用标签的目的 | 不允许的用途 |
|---|---:|---|---|
| D_fit | 45% | 拟合一次点预测器；学习 target 单位 c | 不能参与最终 calibration rank |
| D_profile | 15% | 冻结预测器下的 signed-residual 分布学习 | 不能作为 test 或最终 calibration |
| D_local | 10% | 在已冻结 feature/CDF 权重下学习 q₀ | 不能选择画像模型、h 候选或基于 coverage 调参 |
| D_cal | 15% | 原始 CP/GeoCP/GeoSIMCP 阈值或 SLCP 的 Δ | 不得拟合 f、画像、特征尺度或选择局部权重 |
| D_test | 余下约 15% | 最终一次性评价 | 不得选择任何模型、特征、参数、分组或方法 |

独立行且无分组约束时，前四集合取 floor(N×比例)，余数全部给 test。示意预算如下，**不是本轮已经生成的拆分**：

| N | fit | profile | local | cal | test |
|---:|---:|---:|---:|---:|---:|
| 3000 | 1350 | 450 | 300 | 450 | 450 |
| 1415 | 636 | 212 | 141 | 212 | 214 |

没有额外验证集来搜索最佳配置；本协议参数直接事前固定。未来若要模型选择，必须另立只在 D_fit 内进行的嵌套设计，不能挪用 D_cal/test 或把本方案运行后改写为调优结果。

### 3.2 主任务：横截面随机留出

主拆分 seed 固定 **2026101501**。先确定无标签值参与的 cohort/重复单位规则，再对唯一 group ID 做确定性随机排列；所有方法共享相同角色清单。

- Seattle：优先使用可核实的 property/parcel ID；现有表头未提供，不能把行号当成 parcel。当前适配使用原始稳定行 ID，并将八属性及 UTM 坐标完全相同的记录归入同组。重复房屋识别不完整的限制必须披露。
- PM2.5：以站点编码为 group ID；若未来数据有重复日期，不能把同站点不同时刻误当独立行随机拆分。本单日任务同站点全部进入同一角色。
- 按排列后的 groups 填充前四角色的累计目标样本数；使边界最接近目标且不切开 group，距离相同选择较小累计数；test 取余。记录实际计数，不能为凑出示意数字拆开重复组或重新选择 seed。

在同一数据集内，f、画像和 c 对所有方法一致；不复用历史 outer-test predictions，因为新的五角色设计要求重新建立独立的 predictor lineage。点预测器只能在未来获得执行授权后新建目录训练。

### 3.3 预定压力任务：空间阻隔留出

压力拆分 seed 固定 **2026101502**，无论主任务结果好坏均按计划报告，不作为挑选最优拆分的备选。

- Seattle：EPSG:32610 UTM 平面上，以固定原点 (0,0) 建立 10 km×10 km 网格，group 是网格块与重复单位约束的并集。
- PM2.5：经纬度采用固定全球原点 (−180°,−90°) 的 2°×2°网格；这是角度网格，不宣称每块都恰好 200 km。相同站点不能跨角色。
- 对唯一块/group 用上述同一分配规则决定角色；cal 与 test 位于不同块，实际比例允许偏离行数预算。块太大导致无法构成五个非空角色或低于最低数值支持时，报告该压力任务不可执行，不换块大小或 seed 来求成功。

这个任务检验向未校准区域的空间外推压力。它可能改变 calibration/test 分布，因此不能援引随机留出下的交换性保证。原 ΦR-SLCP 不新增 block conformal、importance weighting 或动态 recalibration。

### 3.4 纳入、缺失与不可用查询

在读取标签数值前，按稳定 ID、坐标合法性和标签是否存在的布尔标志确定 cohort。缺失/无效坐标不能形成所有空间方法的共同任务，须在分配角色前记录数量及原因；标签缺失只依据可用性标志记录，不依据取值极端程度筛样。

合法协变量缺失由 D_fit 中位数插补，增加固定缺失指示列；统计量只由 D_fit 计算。任一必需特征在 D_fit 全缺失，则停止该配置并报告 schema 不满足，不根据其他集合补统计量。真实极端值、低 ESS、覆盖失败、超出训练范围、区间 infinity 均保留；锁定 test cohort 后，不因方法结果删除查询。

## 4. 真实数据残差画像如何估计？

### 4.1 共用冻结预测器

两套数据均预定同一 RF 规格，完整参数如下；压力任务也使用同一模型种子。

```python
RandomForestRegressor(
    n_estimators=300, criterion="squared_error", max_depth=None,
    min_samples_split=2, min_samples_leaf=5, min_weight_fraction_leaf=0.0,
    max_features=1.0, max_leaf_nodes=None, min_impurity_decrease=0.0,
    bootstrap=True, oob_score=False, n_jobs=1, random_state=2026101501,
    verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None,
    monotonic_cst=None,
)
```

依赖固定为当前只读环境中已核对的 Python 3.11.16、numpy 1.26.4、scipy 1.15.3、scikit-learn 1.6.1；本轮只检查版本与构造函数签名，没有实例拟合。未来执行需在新目录保存环境锁文件；不自动升级或以新版本默认值改变上述配置。

不复用历史最优配置或 notebook 中根据 validation 筛选的参数。Seattle 历史 XGBoost 与本方案 RF 不同，因此本研究是新的预定 predictive-calibration comparison，不是声称复现旧 predictor 的性能。

| 数据 | 点预测和画像可用协变量 | 明确排除 |
|---|---|---|
| Seattle | bathrooms、sqft_living、sqft_lot、grade、condition、waterfront、view、age，及 UTM_X / UTM_Y | log_price、price 及其派生量作为输入；旧残差、区间结果、GCCM/CMI 权重；未证实预测时可用的变量 |
| PM2.5 | dem、sp、tp、blh、r、aod_sat、ndvi，以及经纬度的固定球面坐标变换 | PM2.5 本身、其他站点测试浓度、同日浓度生成的插值/邻域均值、站点名称/ID作为预测特征、旧模型残差或 causal 权重 |

PM2.5 的 AOD/气象变量必须确认来自预测任务允许使用的同日外部资料，而不是用 held-out 站点浓度生成的产品。若预测时无法取得，就不能将此实验写为实时预报；不能在看过 test 后另选更有利的特征集合。

### 4.2 条件特征与拟合目标

每个普通属性的插补/缺失指示、中心（D_fit 中位数）和尺度（D_fit IQR；为零则置 1）先冻结。Seattle 地理输入为 UTM 米坐标；PM2.5 地理输入为从经纬度确定性生成的三维球面坐标（第 5 节）。地理维度同样按 D_fit 中位数/IQR 标准化，记为 z_geo；普通属性及缺失指示记为 z_A。测试超出训练范围时不裁剪，记录 OOD。

固定设计基为

\[
B(x,s)=[1,z_A,z_{\mathrm{geo}},z_{\mathrm{geo}}^2,
\{z_{\mathrm{geo},u}z_{\mathrm{geo},v}:u<v\},\tanh(2z_{\mathrm{geo}})].
\]

只在 D_profile 上，用真正 out-of-fit 的 E_i=(Z_i−f̂_i)/c 拟合

\[
E\mid X,S\approx N(\delta(X,S),\sigma^2(X,S)),
\quad\delta=B\beta,\quad\sigma=\exp(B\gamma).
\]

目标为 mean[log σ+.5(E−δ)²/σ²]。沿用原确定性初始化（OLS mean，log-SD 截距=log RMS，其余为零）与单次 L-BFGS-B：maxiter=2000、maxls=50、ftol=1e−12、gtol=1e−6；要求成功且最大梯度≤1e−4。没有多起点、优化到成功或根据最终结果更换模型。数值失败记录为失败单元。

对 r≥0，

\[
\widehat F_R(r\mid x,s)=
\mathcal N\!\left(\frac{r-\hat\delta(x,s)}{\hat\sigma(x,s)}\right)
-\mathcal N\!\left(\frac{-r-\hat\delta(x,s)}{\hat\sigma(x,s)}\right).
\]

其中 𝒩 为标准正态 CDF。这同时表达预测偏差与尺度，但不能表达任意多峰、偏态或重尾误差。结果不佳时如实报告；distributional forest / quantile forest 等替代 learner 只能另立协议，不能在本次 test 上试到最好再替换。

不得直接使用 D_fit 的 in-sample residual 学画像，否则画像可能学到过于乐观的训练误差。D_profile 标签本来就允许用于分布学习；它不是最终 calibration。不能将“localizer 使用 D_profile 学出的 CDF”误称为 calibration-label 泄漏，也不能把 D_cal 混入其中。

## 5. 论文 baseline、同预算对照与命名

### 5.1 必需四个名称的准确含义

| 名称 | 定义与标签池 | 与主方法的关系 |
|---|---|---|
| CP | 冻结 f 下 D_cal 的原始 R；精确 ceil[.9(n_cal+1)] split-CP rank | 全球残差阈值 |
| GeoCP | D_cal 的 R；仅地理 kernel；原 weighted step quantile，加查询 self-weight=1 的 infinity atom | 固定地理相似性实例 |
| GeoSIMCP | D_cal 的 R；地理/普通属性联合 kernel；同一 weighted step quantile 与 self atom | 固定普通属性相似性实例 |
| Residual-profile SLCP | 本文第 2–4 节的条件 score CDF、连续 local quantile 与独立 outer correction | **就是 ΦR-SLCP 的非因果名称/等价审计项，不是独立算法** |

因此论文有四个不同计算程序：CP、GeoCP、GeoSIMCP 和 ΦR-SLCP（Residual-profile SLCP）。如表格仍同时列后两个名称，必须注明 alias、预期差异为零；不计为第五种独立方法，不声称胜过同一算法。不能为了制造新增 baseline 而削弱普通 residual profile 的协变量、模型或数据预算。

### 5.2 固定地理/属性 kernel 实例

Seattle 以 UTM_X/Y 除 1000 得到 km 坐标。PM2.5 以合法经纬度构造

\[
v(s)=6371.0088(\cos\varphi\cos\lambda,\cos\varphi\sin\lambda,\sin\varphi)\quad\mathrm{km},
\]

角度先转弧度。PM 的 d_geo=||v_i−v_j|| 为球面弦长，是预定的地理 proximity，不称为精确道路距离或精确椭球测地距离。这样不依赖未核实的 proj_x/proj_y CRS。

令 d_ref 为 D_fit 中所有正地理 pairwise distances 的中位数；没有正距离则固定为 1 km 并记录坐标退化。地理距离 d_s=d_geo/d_ref。属性距离采用

\[
d_a=\sqrt{p^{-1}\sum_{k=1}^p(z_{A,i,k}-z_{A,j,k})^2},
\]

其中 p 是冻结的属性/缺失指示数；坐标不重复混入 d_a。

\[
K_{\mathrm{GeoCP}}=\exp[-d_s^2/(2\cdot .4^2)],
\qquad
K_{\mathrm{GeoSIMCP}}=
\exp\!\left[-\frac{.5d_s^2+.5(d_a+10^{-8})^2}{2(.4+10^{-8})^2}\right].
\]

h_geo=.4、lambda=.5、Euclidean feature metric、固定预处理与原 self atom 规则均事前冻结，无 validation/test 最优搜索。weighted step quantile 沿用 formal 的 .9 与 1e−9 累计边界容差。上述真实坐标/属性单位适配是新的预定实例；不声称是作者所有超参数配置的最优复现，也不假定任意查询 kernel 加 infinity atom 后就有分布无关 coverage 保证。

### 5.3 必须报告的预算控制

主方法除最终 D_cal 外还使用 D_profile 与 D_local 标签；直接 CP/GeoCP/GeoSIMCP 没有用尽这些角色。仅比较直接 baseline 不能把差异全归因于 profile geometry。

因此预定补充：

- **GeoCP-SLCP control、GeoSIMCP-SLCP control**：同一 D_local、同一 q₀ 学习与 outer Δ；仅 kernel 换为第 5.2 节固定 kernel。总角色预算及独立 outer 程序一致；带宽选择仍不同，不能称为纯几何单因素对比。
- **CP-all-residual-budget**：在 f 和 D_fit 预处理固定、参数未利用这些残差选择的前提下，合并 D_profile+D_local+D_cal 的 R 做一次全球 split CP。它使用与主方法相同的总非 fit 标签预算；不把已经训练过 f 的 D_fit 残差加入。该辅助 CP 不影响其他方法的数据隔离。

以上为事前预算对照，不能根据主比较结果好坏决定是否报告。

## 6. Seattle 房价实验协议

1. 任务：同分布横截面房屋价格预测与预定空间块外推；不作未来市场预测。当前来源的 property identity 与交易日期缺失必须写入限制。
2. 主 target：文件 log_price，点预测 RF 与 Gaussian residual adapter 如上。禁止把 log_price/price 作为属性相似性输入。
3. 角色预算与两个 split：第 3 节；所有算法共享同一 test cohort。不能复用旧 seed42 的 test 作为未暴露的新 holdout，也不能读取旧测试误差来选区域或截尾。
4. 主指标：log-price 尺度 coverage、width、IS；美元尺度按端点 [10^L,10^U] 还原，另报美元 width/IS/coverage。这一单调还原保留覆盖，但区间不再对称，美元效率排序不必与 log 尺度一致；不择优选择尺度作为主结果。
5. 不对美元区间做上限裁剪；指数还原溢出视为美元尺度 infinity 并保留。log 尺度有限不等于美元区间有限。
6. 子群：固定地理块，以及由 D_fit 的预测值 f̂ 四分位阈值定义的 test 预测价格层。分层只用冻结预测，不用 test 真价格寻找难样本或选择报告。

现有数据的 label exposure classification 固定为 retrospective/exploratory。新房屋样本若要更强外部验证，须先核实原始采样、价格单位、交易时间和 property-level ID，再按未读取 outcome 的规则封存，不在本轮实施。

## 7. PM2.5 实验协议

1. 任务：2018-05-31 同日站点横截面估计；训练/画像/local/cal/test 站点互斥。不得把单日数据包装成时间预报或跨日 transport 结果。
2. 主 target：原始 PM2.5 浓度；主 width 和 IS 单位为 µg/m³。固定七个环境属性，不根据既有特征重要性、残差关联或 test 表现增删变量。
3. feature availability/provenance gate：AOD、气象、NDVI/DEM 的生成时间与数据来源应独立于 held-out 站点 PM2.5 标签。任何用全体站点浓度估计的 covariate product 必须标为泄漏风险并停止该配置；不默默继续使用。
4. 不使用站点 test 浓度作为邻站、图平滑或空间 lag 特征。若未来研究真正允许已观测邻站浓度，需重新定义预测信息集和拆分，不能混入当前比较。
5. 主随机留出与预定 2°空间块压力任务均按第 3 节报告；同一站点不能因分拆或样本复用跨角色。
6. 子群：固定 2°块，以及以 D_fit 的预测浓度四分位阈值定义的低/中/高预测浓度层。不能用 test 真污染值划分后只展示覆盖较好的组。
7. 没有未来日期数据时，不设计或声称时间 holdout 已存在。未来时间验证必须另外封存过去的 fit/profile/local、独立部署时 calibration、未来 test 的日期和 feature availability，并重新说明非交换性覆盖边界；不从本协议推断已有保证。

现有 1415 站点快照仍为 retrospective/exploratory；重新 seed 不消除历史标签暴露。当前也没有证明所有 PM 外部产品的生成过程不含浓度标签，这属于未来只读 provenance audit 的先决检查，不是已通过的事实。

## 8. Test 泄漏防线：执行时的强制顺序

| 阶段 | 可读标签 | 必须先冻结的内容 |
|---|---|---|
| 0：只读元数据/预封存 | 仅标签存在标志，不读取数值 | cohort IDs、exposure status、来源、单位、feature availability、group/block、角色清单、代码与配置 |
| 1：point predictor | D_fit | 插补/尺度、c、RF 参数及最终模型；全部角色的预测 |
| 2：profile learning | D_profile | β/γ、条件基、CDF 网格、所有规则与模型 diagnostics |
| 3：local quantile learning | D_local | 对 cal/test 的权重、h、ESS、B_R、q₀ 及其哈希；localizer 从不接收 Y_cal/test |
| 4：final calibration | D_cal | 原始 baseline 阈值、SLCP 的 Δ、每个 test 查询最终区间 |
| 5：evaluation | D_test，最后一次解封 | 全部方法、区间、query IDs、评价代码与预定子群已封存 |

test 协变量只允许用于已冻结模型的逐查询预测/权重以及事前的坐标/ID 分块。不能用 test covariate 全局统计量更新插补、特征尺度、地理 d_ref、画像基、候选 h 或 subgroup 阈值；使用 test labels 做误差地图后不得返回改变任何方法。

未来数据适配应先建立 feature-only 视图和按角色封存的 label store；模型/localizer 入口不接收完整带 test Y 的 DataFrame。最终 evaluator 单独解封 D_test。若原始 CSV 本身包含全部标签，应由受控适配入口处理，并对标签导出/读取阶段留日志，不能把“没有显式使用 test Y”当成已建立强隔离的证明。这些工程措施只能减少本轮意外泄漏，不能撤销已发生的历史暴露。

执行前预定三类检查：

- 替换虚拟 Y_cal：画像、weights/h/q₀ 完全不变；Δ 随 calibration 标签改变是正常校准行为，不能要求最终半径不变。
- 替换虚拟 Y_test：全部模型、权重、阈值和最终区间完全不变；仅最终评价可以改变。
- 按 ID 检查所有角色互斥、重复单位不跨角色；预处理来源只能为 D_fit；记录每次标签读取阶段与封存哈希。

若检测到实际 cal/test 标签提前使用，停止该配置；不把修复后在同一 test 上再次调参的结果当作原封存验证。历史 exposure 与本轮 dataflow isolation 分别报告，不能用后者掩盖前者。

## 9. 评价指标、CDF 诊断与结果判读

### 9.1 主要预测区间指标

\[
\widehat{\mathrm{Coverage}}=n^{-1}\sum_i\mathbf1\{L_i\le Z_i\le U_i\},
\quad \mathrm{Width}_i=U_i-L_i,
\]

\[
\mathrm{IS}_{.1,i}=U_i-L_i+20(L_i-Z_i)\mathbf1\{Z_i<L_i\}
+20(Z_i-U_i)\mathbf1\{Z_i>U_i\}.
\]

全体 test 查询共同评价：coverage、mean/median width、mean IS、infinity rate、失败率。均值 width/IS 保留无穷，finite-only 统计只能作为有明确标签的附加描述；失败占位 infinity 的 coverage 不能记作有效成功。

同时报告实际 n_cal、local ESS 的均值/分位数/最小值、pmax、B_R、low-information rate、OOD rate、同一地理块内的 q₀/最终半径变化、与 uniform-local q₀ 的差异。预定子群逐个报告样本数和 coverage；空组保留。单活动层没有 v2 的分组合并，但也不保证所有位置都具有足够局部信息。

### 9.2 真实 score CDF 不可观测：不能照抄 oracle 指标

每个实数据位置通常只有一个 outcome；未知真实 F_R(r|x,s)，因此**不能报告“真实 CDF matching error 已恢复”**。以下只作可观测代理诊断：

1. **区域 CDF discrepancy**：在事前固定的地理块/预测层 g 中，比较 test score 的经验 CDF 与算法预先给出的 score-CDF 预测的组平均，记录网格 sup 差及组样本数。SLCP 的预测为 D_local 原始 R 的局部加权经验 CDF；CP/GeoCP/GeoSIMCP 为相应 D_cal 池 CDF。它包含有限样本噪声和组内混合，不能当作逐位置真 CDF 距离，也不使用 test 结果重新校准。
2. **固定网格 CDF Brier diagnostic**：在每个 query 计算 [0,12] 上预测 CDF 与 1{R_test≤r_k} 的平均平方差。明确标为有限网格诊断，不称作完整 CRPS；R_test>12 时继续保留，并另报超网格比例。
3. **profile-only check**：画像模型的 PIT 经验分布与尾部质量，test 解封后仅用于诊断模型失配，不据此替换 learner。Gaussian 的模型 Lipschitz/尾部界只约束拟合 CDF 的离散近似，不能约束未知真实 CDF 的估计误差。

区域 CDF 诊断不以 n 很小的组来选择赢家，也不事后合并失败组。辅助 CP-all-residual-budget 的 CDF 池按其明确标签池构造；同一算法 alias 的预测/指标必须相同。

### 9.3 不能只看覆盖，也不能用更窄掩盖欠覆盖

主比较为 ΦR-SLCP vs GeoSIMCP，并同时给出其余所有 baseline / 预算对照，不按 test 结果选择最弱对手。每个数据集、每种预定 split 单独报告。

双方经验 coverage≥.90、无失败/infinity 且 mean width 与 mean IS 均下降时，只记录该配置的联合数值改善；单一切分不提供稳定优越性的结论。覆盖不足、只有一个效率指标改善或方法失败都不记为联合成功，不降低到 .89，不删除困难查询。

不计算把空间相关查询视为独立试验的显著性来制造强证据。本次设计只有一个主随机拆分与一个空间压力拆分；它们不是独立 replicate。报告主 split 的逐查询配对 width/IS 差与各预定地理块的配对均值；没有经额外假设核实的空间不确定性模型时，不给“总体覆盖至少 90%”或严格总体显著性声明。两个数据集不按原始量纲混合总分，压力任务不与主任务合并掩盖失败。

## 10. 论文中如何描述定位调整

建议方法标题为 **Residual-distribution-driven spatial conformal calibration (ΦR-SLCP)**；GCCP 可保留为研究框架历史名称，但正文必须给出当前计算定义，不再展开为已证实的 causal gain。

可直接采用的表述：

> 我们以冻结预测器的条件残差分布定义空间校准相似性。通过连续 CDF 距离、独立局部分位数学习及外层 conformal correction，方法将校准邻域与预测误差分布对齐。此前的因果解释版本与同信息的普通条件残差版本在预定模拟中一致，因此本文将贡献定位于残差分布驱动的预测校准，不声称因果语义带来额外收益。

对真实数据应写：

> Seattle 与单日 PM2.5 实验评价横截面空间预测区间的覆盖和效率。复用数据具有历史标签暴露记录，故分析属于回顾性外部数据验证；本轮标签隔离不能把它重新变为未暴露 holdout。覆盖声明限于相应统计假设和实际评价结果，不推断因果效应或未知机制漂移下的普遍保证。

创新讨论应同时包含与直接 GeoSIMCP 的比较，以及同预算/同 outer 程序的对照；不将已有 SLCP 校正原则、Gaussian 分布模型或与自身 alias 的零差异单独作为新贡献。

## 11. 执行前冻结与交付规则

本文冻结数学程序、两个数据任务、角色比例、两个 split seeds、固定模型族/超参数、baseline 与评价规则。**尚未生成实际行级拆分、训练模型或运行真实数据。** 未来机器可执行配置需忠实实现本协议，并在首次拟合前记录：

- 数据文件/字段/单位/来源与 ID 哈希；明确历史暴露分类，不能将原历史目录当作未暴露 test。
- cohort 与重复规则、全部 group/block 和五角色清单、实际样本数、最低支持检查；只读 feature provenance audit。
- 完整 RF/sklearn/scipy/numpy 参数和版本、Gaussian adapter 基函数顺序、优化器、score 单位及 kernel 参数。
- 复制本协议、实现源码、独立验证器、报告规则的 SHA256；新目录独立输出，旧实验全部只读。
- 未来准备 smoke 只能验证适配、标签隔离、分位数及数值支持；不能访问正式 test 标签来选择 profile learner 或最佳参数。任何设计变更须在 test 解封前另立版本，并保留旧版本及变更理由。

建议未来目录分别为 `experiments/gccp_v3_realdata_seattle_v1` 与 `experiments/gccp_v3_realdata_pm25_v1`；本轮不创建或运行它们。

目前的结论是“真实数据验证协议已准备”，不是“方法已在真实数据验证有效”。已有模拟和实数据历史实验保持原样。
