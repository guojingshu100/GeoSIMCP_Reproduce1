# GCCP v3 causal value validation design

日期：2026-10-12。阶段：S7/S8 最小模拟设计与理论可识别性检查。**本文没有运行模拟、训练模型或访问真实数据；没有修改主方法、已有代码、协议或结果。** 文中的数值是预定设计参数，不是实验结果。

## 1. 研究结论先行：当前主方法下不能制造 causal semantic gain

现有 [formal report](experiments/gccp_v3_formal_validation_20261012_v1/REPORT_zh.md) 支持：相对预定 GeoSIMCP 实例，ΦR-SLCP 在 S2/S3 满足 width 与 Interval Score 的联合效率门槛。但 non-causal R-SLCP 与 ΦR-SLCP 完全一致，尚无因果语义增量证据。

本次源码核对进一步发现，[strategies.py](experiments/gccp_v3_formal_validation_20261012_v1/strategies.py) 的 `learn_localizers` 将同一个 residual localizer 状态同时赋给这两种方法。它们也共享分布拟合、局部分位数及外层校准。因此，在保持现有实现定义不变时，**S7/S8 即使引入混杂或 assignment shift，两者仍应逐查询相同**。这种实现相等本身不是独立估计器的实证等价检验；理论上是否相等，要另外检查识别条件。

不能通过让普通残差方法忽略混杂变量、给因果方法额外 oracle 信息，或偷偷添加 propensity adjustment 来得到“因果胜出”。本设计将以下问题分别验证：

1. 在混杂被充分控制时，完整条件残差画像是否仍等价？
2. assignment shift 后，冻结的完整条件画像是否仍对应目标 score 分布？
3. 画像稳定是否足以让 source 外层 CP 保持 target coverage？
4. 什么不同的预测目标才会产生 observational 与 interventional 残差分布的实质差异？

前三项属于当前主方法的 S7/S8 验证；第四项采用解析诊断说明，不将另一种画像定义冒充当前 ΦR。

## 2. 三个必须区分的分布

令 W 为全部已观测的处理前条件变量；本设计 W=S=(s₁,s₂)。冻结预测器为 f(W,a)，处理 a∈{0,1}，潜在残差为

\[
R(a)=|Y(a)-f(W,a)|.
\]

完整条件因果画像与完整条件观测画像分别是

\[
G_a(r\mid w)=P(R(a)\le r\mid W=w),
\qquad
H_{e,a}(r\mid w)=P(R\le r\mid W=w,T=a,E=e).
\]

在 consistency、给定 W 无未测混杂、positivity，以及跨环境潜在结果条件分布稳定时，

\[
H_{e,a}(r\mid w)=G_a(r\mid w)\quad\text{对每个环境 }e.
\]

证明：在 T=a 的观测中 consistency 给出 R=R(a)；条件可交换性允许从条件中去掉 T；机制稳定允许去掉 E。残差变换使用同一个冻结 f，因而保留这一分布等式。这是标准识别条件在 score 分布上的直接推导，背景见 [Hernán 与 Robins，Causal Inference: What If](https://miguelhernan.org/whatifbook)。

**混杂存在不等于混杂没有被控制。** P(T|W) 非常数，并不自动破坏上述等式。完整条件观测模型可以通过 outcome regression / g-computation 识别同一因果分布；不必显式使用 propensity weighting 才有这种识别解释。

第三个分布是按组或按总体汇总的 score CDF：

\[
F_{e,a}(r)=\int G_a(r\mid w)\,dP_e(w\mid T=a).
\]

改变 e(w)=P(T=1|W=w) 可以改变 P_e(W|T=a)，故 F_{e,a} 可以改变，尽管 G_a 完全稳定。这也是为什么仅按 T 做 source 校准不能普遍解决 assignment shift。

当前 ΦR 是 **当前 T 对应的 G_T 分布水平**，不是 G₁−G₀、不是均值处理效应，也不是对所有混杂变量积分后的总体 do 分布。必须保持这一含义。

## 3. 主方法及比较方法保持不变

沿用 [formal protocol](experiments/gccp_v3_formal_validation_20261012_v1/FORMAL_PROTOCOL_zh.md) 的数学程序与预定参数：

- ΦR：分 T 的 Gaussian outcome distribution，原八维基函数；当前 T 的 folded-normal score CDF。
- 距离：原 [0,12]、2401 点 CDF 网格上的 Chebyshev 距离；继续报告连续距离离散化与尾部界。
- 同 T 连续高斯权重；h∈{.025,.05,.1,.2,.4}；最小可行 h 满足 ESS≥50、pmax≤.05、B_R≤.10。
- 不可行时保留 B_R 最小候选并标记 low information；不扩大候选、不合并、不删除。
- D_local 学习加权 .9 分位数 q₀；独立 D_cal 使用 U=|Y−f|−q₀，按 T 取 ceil[.9(n_T+1)] 阶统计量 Δ_T。
- 最终半径 max(0,q₀+Δ_T)；允许负 Δ_T；外层样本不足时保留 infinity。
- 权重选择与 q₀ 工件先冻结，之后才读取最终 calibration labels。D_local 标签只用于局部分位数学习。

必需比较为 CP、GeoSIMCP、non-causal R-SLCP、ΦR-SLCP。CP 和 GeoSIMCP 的定义、固定核、原分位数规则均沿用 formal protocol。non-causal R-SLCP 仍是同变量、同分布模型、同样本预算的完整条件残差对照，不能改成未调整的边际残差模型。

保留原 GeoSIMCP-SLCP control 作为一个辅助公平性对照：它享有相同 D_profile/D_local/D_cal 预算和独立外层校准，避免将程序差异误认为几何差异；其固定带宽与 ΦR 的自适应选择仍不同，必须披露。它不是新增主方法，也不替代用户指定的四方法比较。

## 4. 共用 SCM：只改变处理分配，保持结果机制稳定

采用已完成 S2 的结果机制和预测器，避免同时引入新模型族或预测器变化：

\[
s_1,s_2\stackrel{iid}{\sim}\mathrm{Unif}[-1,1],
\quad U_T\sim\mathrm{Unif}(0,1),\quad\epsilon\sim N(0,1),
\]

三者独立；令 X=s₁、Z=s₂，W=S=(X,Z)，

\[
m(S)=s_1+.5s_2,\qquad
\sigma_a(S)=.6\exp\{a[.8+.6\tanh(2s_2)]\},
\]

\[
Y(a)=m(S)+\sigma_a(S)\epsilon,\qquad
T_e=\mathbf1\{U_T<e_e(S)\},\qquad Y=Y(T_e),\qquad f(S,T)=m(S).
\]

Z 同时影响 T 分配与潜在结果分布，是已观测处理前混杂变量；它同时进入两种残差画像。无未测混杂、无干扰，positivity 由预定 propensity 下界保证。全部环境都满足

\[
P_e(Y(do(a))\mid X,S)=N(m(S),\sigma_a^2(S)).
\]

此处 X=s₁ 已包含在 S 中，写作 P(T|X,S) 与 P(T|S) 等价，没有额外隐藏条件变量。真值 score CDF 为

\[
G_a(r\mid S)=
\begin{cases}
2\mathcal N(r/\sigma_a(S))-1,&r\ge0,\\
0,&r<0,
\end{cases}
\]

其中 𝒩 是标准正态 CDF。arm-specific mean 与 log-SD 均在原八维基函数模型内；因此只测试分配改变，不制造新的结果模型失配。

## 5. Scenario 7：observational confounding 与分配规则反转

定义两个观察环境：

\[
e_+(S)=.2+.6\operatorname{expit}(2s_2),\qquad
e_-(S)=.2+.6\operatorname{expit}(-2s_2)=1-e_+(S).
\]

两者均在 (.2,.8) 内，支持重叠；由于 s₂ 对称，两个环境的总体 P(T=1) **都为 .5**。因此这不是已有 S5 的单纯总体处理比例变化，而是 P(S|T) 的实质变化。

预定三项比较任务：

| 单元 | D_profile / D_local | D_cal | D_eval | 所检验的问题 |
|---|---|---|---|---|
| S7-IID | e₊ | e₊ | e₊ | 有观察性混杂但已充分调整；两种残差画像的等价负对照 |
| S7-source-cal | e₊ | e₊ | e₋ | 只用源标签时的分配迁移压力测试，不承诺 target 90% |
| S7-target-cal | e₊ | e₋ | e₋ | 不改算法，只换成独立目标校准样本；检验目标分布内效率 |

target-cal 与 source-cal 的标签预算相同，各使用一批 600 个校准标签，不把两批合并。所有方法在同一单元获得同一批 D_cal；目标 D_cal 的标签不进入画像拟合、权重选择或 q₀ 学习。target-cal 是一个单独的信息可用性条件，不宣称零标签迁移成功。

预期可证伪判断：完整条件 G_a 应保持稳定；non-causal 与 ΦR 应相同。若两者不同，先检查实现、输入、估计器或信息预算是否被改变，不能立即解释为因果增益。source-cal 可能接近或偏离 90%，两种结果都保留；不为了得到失败而加剧 shift。target-cal 恢复外层所需的目标 T 内可交换性，但不保证每个空间子群或每个已实现 calibration 集都达到 90%。

## 6. Scenario 8：intervention transport shift

source 使用相同观察机制 e₊，target 改为随机化政策

\[
e_{\mathrm{int}}(S)=.5.
\]

target 的 T 不再依赖 Z，但 Y(a)|S、P(S) 和 f 均不变。相对于 source，**总体处理比例仍为 .5，改变的是处理选择与混杂变量的关系**。

| 单元 | D_profile / D_local | D_cal | D_eval | 解释 |
|---|---|---|---|---|
| S8-source-cal | e₊ | e₊ | e_int | 冻结 source 校准迁移到随机干预环境；不额外加入 transport weights |
| S8-target-cal | e₊ | e_int | e_int | 画像/local learner 仍来自 source，目标独立校准恢复外层条件 |

采用随机化政策而非全体强制 T=1，以保留两处理臂和原校准程序，避免让组消失成为主要差异。T=1 强制干预不属于本次最小设计，不能事后加入以选择有利结果。

两个任务均预测目标环境**实际分配处理后的 Y**；不改成个体处理效应区间或未观测潜在结果区间。后两者需要另外的预测目标与识别/覆盖协议。

## 7. 为什么稳定因果画像不等于 source→target CP 保证

给定冻结 q₀，原外层所用的分布是

\[
P_e(U\le u\mid T=a)
=\int G_a(q_0(w,a)+u\mid w)\,dP_e(w\mid T=a).
\]

G_a 稳定并不能使这个混合分布自动稳定；S7/S8 改变了混合权重。即使 oracle q₀=cσ_a，其中 c=𝒩⁻¹(.95)，仍有 U=σ_a(|ε|−c)，其整个条件 T 分布一般随 σ_a 的组内混合而改变。

oracle q₀ 在每个 W 上恰好有 .9 覆盖，因而不做外层修正时可以迁移；这不等于有限样本随机 Δ_T 的原秩论证在 source→target 仍成立。真实估计 q₀ 还存在拟合与邻域误差。不能从本设计推出 coverage 必然失败，也不能推出必然有效。

覆盖条件如下：

| 校准情形 | CP | non-causal / ΦR-SLCP | 原直接 GeoSIMCP |
|---|---|---|---|
| S7-IID | 全体边际 split-CP 保证 | 条件于独立学习阶段、按 T 的边际保证 | 任意相似性核没有自动的同等分布无关保证；如实评价 |
| source-cal→shifted target | 原交换性论证不适用 | 除非额外满足 U|T 稳定等条件，否则无目标保证 | 无本设计可援引的目标保证 |
| 独立 target-cal→target | 全体目标边际保证 | 原独立外层按 T 的目标边际保证；source 学习误差主要影响效率 | 仍只报告实际覆盖，不借用 SLCP 定理 |

当 α=.1 时，外层每 T 至少 n_T=9 才能避免秩要求本身造成 infinity；这与 local learner 的 ESS≥50 是两件不同的事。模型非有限、拟合失败等仍另行记录。

本设计可记录已知密度比作为诊断：

\[
\rho_e(w,a)=
\left[\frac{e_e(w)}{e_+(w)}\right]^a
\left[\frac{1-e_e(w)}{1-e_+(w)}\right]^{1-a}.
\]

它衡量在 P(W) 不变时联合 (W,T) 的目标/源比例。**不将它乘进现有权重或外层分位数**；那会改变主方法。真正的 weighted conformal transport 需要相应的校准程序、密度比条件与证明，不能把画像相似性权重当作该密度比。参见 [Tibshirani 等，Conformal Prediction Under Covariate Shift](https://www.stat.cmu.edu/~ryantibs/papers/weightedcp.pdf)。

## 8. 哪种条件下才可能出现 causal residual 的实质价值？

考虑另外一种**粗条件/总体干预目标**，保留 C=s₁、对 Z=s₂ 积分。这个解析诊断不替换主方法的完整 S 条件画像。

源观察分布与总体干预分布分别为

\[
H^{\mathrm{coarse}}_{+,a}(r\mid C)
=\int G_a(r\mid C,z)p_+(z\mid C,T=a)\,dz,
\]

\[
G^{\mathrm{std}}_a(r\mid C)
=\int G_a(r\mid C,z)p(z\mid C)\,dz.
\]

由于 Z 同时影响分配和残差尺度，它们通常不相等。在本 SCM 的 a=1 臂，e₊(z) 和 σ₁(z) 随 z 增加，而 G₁(r|z) 对 z 递减，因此对任意有限 r>0 有

\[
H^{\mathrm{coarse}}_{+,1}(r\mid C)
<G^{\mathrm{std}}_1(r\mid C)
<H^{\mathrm{coarse}}_{-,1}(r\mid C).
\]

这可由单调函数协方差符号直接推出，无需抽样来搜索有效参数。a=0 的尺度固定，三者相等，是另一个负对照。

S8 的随机化 target 满足 p_int(z|C,T=a)=p(z|C)，所以目标粗条件 score CDF 恰为 G_std；source 粗条件 H₊ 则存在选择混合偏差。这展示了 **interventional estimand 与 observational mixture 的区别**。但 S7 的 target 仍为非随机分配，目标粗条件 CDF 是 H₋，不是 G_std；不能将“标准化 do 画像”无条件当成所有 target 政策的预测分布。

一般政策目标必须使用

\[
F^{\mathrm{target}}_a(r\mid C)
=\int G_a(r\mid C,z)p_{\mathrm{target}}(z\mid C,T=a)\,dz.
\]

这种因果价值需要：明确的干预/政策预测目标、足够的混杂测量或其他有效识别信息、稳定潜在结果机制、source→target 支持重叠，以及与目标一致的标准化/迁移操作。若条件不满足，因果标签没有修复能力。

公平对照还必须包含：**用相同完整条件观测模型，再进行相同 target 标准化的 residual estimator**。它也能恢复同一目标分布。若它与因果 estimator 一致，所证实的是正确调整/迁移操作的价值，不能宣称“只有因果模型才能做到”。故意仅提供未调整粗条件 residual baseline，只能证明调整的价值。

若 Z 不可观测、又没有其他识别信息，当前 observational outcome regression 无法一般识别 G_a；当前两方法依旧输出相同，但 causal 解释可能错误。直接用模拟 oracle 潜在结果赋予 ΦR、再与观测 baseline 比较，至多是 oracle 上界，不能算可实现的 causal gain。

**结论：当前全部冻结定义下，S7/S8 不能证明 ΦR 比同信息的完整条件 R-SLCP 更好。** 若今后要验证标准化、政策迁移或使用额外识别结构的增量，必须先另立画像估计协议；虽然可以复用 SLCP 校准核心，但不能声称完整主方法未改变。本轮不进行该扩展。

## 9. 最小未来验证规模与审计：仅设计，不执行

先做上述解析等价检查；该检查已经说明，用更多 replicate 搜索两种同算法输出的差异没有科学意义。

若后续要验证实现边界，建议只做一个预定 seed `2026101401` 的五个单元：S7-IID、S7-source-cal、S7-target-cal、S8-source-cal、S8-target-cal。每个单元沿用 800/400/600/2000，既保持原样本预算，也避免为了“最小”另行修改拟合门槛。这是单 seed 的边界 smoke 设计，**不支持显著性或总体优越性声明，不自动扩展到 20 replicate**。

同 seed 共享 source profile/local 数据；source-cal 两个 shift 任务共享源校准数据。相应 target-cal 与 target evaluation 使用独立随机流；S7/S8 可共享潜在随机原语以配对，但同一方法的 cal 与 eval 不能重用样本。所有方法共享同单元数据与冻结 f。未来独立目录建议 `experiments/gccp_v3_causal_value_smoke_20261012_v1`；本轮不创建实验目录或生成样本。

若未来需要正式推断，另行事前冻结 replicate 数、种子与多重比较规则；不基于上述一个 seed 的好坏选择继续、调参或改变场景。由于现有两残差方法结构上相同，**不建议单为证明它们不同而开展正式批次**。

未来需记录以下量，保留所有失败查询：

| 量 | 预定定义与用途 |
|---|---|
| 画像估计误差 | 估计当前 T CDF 与解析 G_T 的网格 sup 距离；另报尾部/离散化界 |
| CDF matching error | 目标真实 G_T 与相似性加权 source score CDF 混合的网格 sup 距离；与已有 formal 指标一致 |
| 两残差方法差异 | 逐查询 CDF、权重、h、ESS、q₀、Δ_T、最终半径的最大差；共享实现预期逐项相同 |
| Coverage | 全体经验覆盖与解析真覆盖；按 T、固定 s₂ 四等宽分箱×T 报告；空组明确标记 |
| Width / Interval Score | 全查询均值，IS=2q+20·max(R−q,0)；不截断 infinity，不做 finite-only 主结论 |
| ESS / support | local Kish ESS、pmax、B_R、low-information rate；另报 outer n_T 与 known-propensity overlap |
| Infinity / failures | 原始 infinity 比例与实现失败单独计数；失败占位的 coverage 不当作成功 |
| 信息保留 | 同 T 的 q₀/最终半径变异、与 uniform-T 的差、原预定画像置换响应 |
| 外层 shift | U|T 的 source/target 分布差作为评价诊断；不能反馈修改 q₀、Δ 或方法 |

calibration/evaluation 标签不能用于选择 localizer、h、分组、模型或场景强度；真 G、propensity 和潜在结果仅用于生成/独立评价。失败不能更换 seed、重拟合到成功或删除低支持查询。

预定判断规则：

1. 两 residual 方法在 S7/S8 相同：报告“当前方法没有可分离的 causal semantic gain”，不是实验失败后更换 baseline。
2. source-cal 覆盖不足：报告 assignment shift 改变组内 U 分布；不临时加入 transport correction。
3. target-cal 覆盖良好：支持目标独立校准程序，不证明零标签 causal transport。
4. 覆盖达标但 width/IS 无共同改善：不宣称有效率优势。
5. CDF matching 改善但最终区间不改善：说明画像对齐没有转化为校准收益，不能以中间指标代替主要结论。

## 10. 论文定位与 Seattle / PM2.5 决策

### 当前可支持的定位

建议主线为 **score-distribution-aware spatial conformal calibration**：以冻结预测器的残差 CDF 定义与校准目标一致的连续相似性，通过局部分位数学习和独立外层校正保持局部信息，同时明确覆盖的适用条件。

可写为：

> 我们研究使用冻结预测器残差分布几何的空间预测区间校准。该几何能够区分均值效应相同但误差尺度不同的位置；在独立局部学习与外层校准下保留连续局部信息。预定模拟表明，在尺度和基准噪声异质场景，相对固定 GeoSIMCP 实例可同时改善宽度和区间评分；其完整条件观测残差版本与因果解释版本一致，尚未显示额外因果语义收益。

“Causal residual profile”目前应作为满足识别条件时的解释或候选扩展，不作为已证实优于普通 residual learning 的主创新。独立外层 conformal correction 本身也不能单独作为新定理贡献。

还必须报告已有边界：S1/S4 未获得联合效率优势；S6 机制漂移导致覆盖失效；与相同预算/外层的 GeoSIMCP-SLCP 对照相比，S2/S3 为 IS 改善但 width 没有改善。不能将对直接 GeoSIMCP 的结果升级成“全面优于同预算方法”。

### 是否值得进入 Seattle / PM2.5？

**若论文必须以独有 causal semantic gain 为主贡献，目前不宜直接进入两套真实数据的正式实验。** 当前算法等价问题不是增加真实样本能够解决的；首先应确定是否存在可识别且有实际意义的干预/政策预测目标。

**若接受任务对齐的 residual-distribution calibration 为主贡献，值得下一阶段选择一套数据做有限、预注册的预测性外部验证。** 它检验真实空间相关、模型失配及效率能否迁移，不承担证明因果增益的任务。之后再决定是否扩展第二套数据，选择规则应事前确定，而非挑选效果好的数据集。

进入前需另行冻结：预测目标与预测时可用变量、空间/时间拆分、冻结预测器、独立 profile/local/cal/eval 数据预算、同预算强 residual baseline、空间依赖下覆盖解释、width/IS 联合门槛及全失败保留规则。若仍作 causal 声明，还需明确 T 的可干预含义、处理前混杂变量、识别假设与支持诊断；未核对数据内容前不假设 Seattle/PM2.5 已满足这些条件。

本轮只交付此设计；没有运行 S7/S8，也没有进入 Seattle/PM2.5。
