# GCCP v3 calibration design

日期：2026-10-11。阶段：理论与校准程序设计。

依据：[v2完整验证报告](experiments/gccp_v2_profile_validation_20261011_v1/REPORT_zh.md)、[退化原因解释](experiments/gccp_v2_profile_validation_20261011_v1/INTERPRETATION_zh.md)、[v2校准策略](GCCP_v2_calibration_strategy_report.md)。本次只新增本文档，不修改代码、旧协议或结果，不运行仿真或真实数据。以下是v3建议，不是已验证结果或已冻结实验协议。

## 1. 方法选择与 A–E 的直接回答

**建议论文主方法采用 ΦR 驱动的 split-localized conformal calibration：先以连续画像权重学习局部残差分位数，再对中心化残差做一次独立的90% split CP。** 下文简称 ΦR-SLCP。完整Guan LCP作为直接局部加权的理论参照；风险控制作为附加扩展，分别说明保证。

SLCP是已有的方法类别，不将其重命名为新发明。主张的新内容应放在干预残差分布几何、信息不足时的局部学习规则，以及画像误差到校准误差的联系。[Han等：Split Localized Conformal Prediction，式(9)–(10)](https://arxiv.org/pdf/2206.13092)

| 必答问题 | 建议与边界 |
|---|---|
| A. 如何避免Method B过度合并？ | 取消覆盖前提中的全树支持度认证；以重叠的连续邻域学习局部形状。最终覆盖由独立的外层秩校准承担，局部ESS不足不触发整棵树合并 |
| B. 如何让ΦR进入权重？ | 用当前T状态的残差CDF水平之间的距离构造核权重，再用这些权重计算查询相关的局部分位数q0(x) |
| C. coverage依赖什么？ | 预测器、画像、邻域和q0函数在最终calibration之前冻结；最终校准与查询同分布且交换；执行正确conformal阶次。局部/逐点覆盖还需分布匹配和估计精度条件 |
| D. 哪种最适合作为论文主方法？ | ΦR-SLCP：覆盖证明短，局部形状可明确保留，有限性条件透明，无需v2的两个95%分支。完整LCP作为重要比较，PAC风险控制不作为小预算下的硬门槛 |
| E. 相对GeoSIMCP创新在哪里？ | 从原始属性相似性转向冻结预测器的干预score分布相似性，并把这种几何放入有效校准程序；需要误差桥接和强消融，不能仅以“换距离+使用LCP”宣称新理论 |

本设计不保证v3一定改善width或interval score。它消除了v2中一个明确的结构障碍，效率仍须由新的预注册验证判断。

## 2. v2失败的是哪一层

v2给出三个不同结论：

1. 画像层：S2/S3中，ΦR距离与真实score CDF距离的Spearman为0.7493/0.9293，明显优于Φμ的0.0590/0.2429。
2. 覆盖层：ΦR_B四场景平均Coverage约95.2%–95.8%，本批次没有infinity。
3. 校准分辨率层：78/80个单元只剩两个T根组，ΦR与Φμ的最终区间几乎相同，增量效率未通过。

原因不是“99稍大”这样单一的常数问题。v2要求独立support数据同时证明所有候选节点有足够未来校准量，再对稀疏分支采用双95%预算。在常见M=14时，每节点需至少110个support观测；两个子叶合计至少220，而每T根通常只有约200。该保证结构会系统性压低分辨率。[本地实证与规则推导](experiments/gccp_v2_profile_validation_20261011_v1/INTERPRETATION_zh.md)

v3的改变是：**局部样本量负责局部估计的质量，独立外层conformal负责边际覆盖。** 不再要求每个局部邻域先取得高概率尾部样本量认证才允许画像发挥作用。

## 3. 画像、目标与连续权重

### 3.1 使用分布水平，不只使用干预差值

记X=(S,Z,T)，冻结预测器为f_hat，绝对残差为：

\[
R=|Y-\widehat f(X)|.
\]

干预score分布和当前状态画像定义为：

\[
G_a(r\mid s,z)
=P\{|Y(a)-\widehat f(s,z,a)|\le r\mid S=s,Z=z\},
\qquad \Phi_R(x)=G_t(\cdot\mid s,z).
\]

保存G0、G1以作机制分析，但主校准距离使用当前T下的CDF水平。只用G1−G0会再次丢失S3中的基准噪声；只用均值效应也不能表示S2的尺度或S4的预测器偏差。

主距离建议为：

\[
\widehat d_R(x,x')
=\sup_{r\ge0}|\widehat G_t(r\mid s,z)
-\widehat G_{t'}(r\mid s',z')|.
\]

网格实现须单独记录离散化和尾部误差，不把有限网格KS冒充精确连续KS。画像单位固定，不以经验treatment-effect标准差重新缩放。

### 3.2 连续核

在同T内定义：

\[
w_j(x;h_R,h_S)
=\mathbf1\{T_j=t\}
\exp\left[-\frac{\widehat d_R(x,X_j)^2}{2h_R(x)^2}
-\frac{\gamma\,d_S(x,X_j)^2}{2h_S^2}\right],
\qquad
p_j(x)=\frac{w_j(x)}{\sum_k w_k(x)}.
\]

主机制验证建议gamma=0，即让残差分布相似性直接决定邻域；地理核作为预定消融。把地理距离乘进去可能排除“地理较远、score机制相同”的样本，反而增加稀疏性。真实空间依赖是否需要地理约束属于额外建模问题，地理权重本身不创造交换性。实现时应稳定归一化log权重，避免将浮点下溢误当成数学上的空邻域。

同T限制暂沿用v2，便于解释覆盖目标；是否跨T共享只能作为后续独立设计。不能为了补样本临时改变目标总体。

这些相似性权重通常不是covariate-shift密度比。不能直接借用密度比加权CP的保证。[Tibshirani等：Conformal Prediction Under Covariate Shift](https://arxiv.org/abs/1904.06019)

## 4. 方案一：完整 ΦR-LCP

这是把ΦR直接用于最终校准权重的方案，需要完整LCP构造。以下用c=1−alpha=0.9表示目标覆盖；原文的alpha表示覆盖水平，须注意符号差别。

在某个T内有n个calibration观测，加上查询构成n+1个点。设置非负localizer H_ij，H_ii=1，归一化为：

\[
p_{ij}=H_{ij}/\sum_{k=1}^{n+1}H_{ik}.
\]

对候选score v，令V_i(v)=R_i（i<=n），V_(n+1)(v)=v，并定义：

\[
\widehat F_i^v=\sum_{j=1}^{n+1}p_{ij}\delta_{V_j(v)},
\qquad
\widehat F_*^\infty
=\sum_{j=1}^n p_{*,j}\delta_{R_j}+p_{*,*}\delta_{+\infty}.
\]

从相应加权经验CDF的可能取值集合Gamma中，取最小beta(v)使：

\[
\frac1{n+1}\sum_{i=1}^{n+1}
\mathbf1\{V_i(v)\le Q_{\beta(v)}(\widehat F_i^v)\}
\ge c.
\]

预测score集合为：

\[
\mathcal C_R(x)=\{v\ge0:v\le Q_{\beta(v)}(\widehat F_*^\infty)\},
\qquad
C(x)=\{y:|y-\widehat f(x)|\in\mathcal C_R(x)\}.
\]

该构造和反演来自Guan的split-score LCP。实际实现应遵循其处理并列和边界的算法，不能把beta(v)固定为0.9。[Guan：LCP，Lemma 3.1及Algorithm 1](https://arxiv.org/html/2106.08460v2)

**覆盖条件：** IID/相应交换性、独立训练的score、localizer与算法的排列规则。允许localizer依据增广样本的无序X集合设计，但必须同等处理每一行，不能只在查询行上偷偷调带宽。更简单的首版是从独立设计数据冻结完整h(x)函数。

**有限性：** 对某个实际分位水平beta，有限加权阈值要求：

\[
p_{*,*}\le1-\beta.
\]

名义覆盖0.9或ESS达标不能保证调整后的beta满足此条件。完整LCP仍可能给出infinity，不能截断或换成普通90%分位数后继续引用LCP定理。

**定位：** 它最直接体现连续校准相似性，适合作为核心理论参照和预定比较；但有限性、水平调整和计算负担使其不是当前最稳妥的唯一主程序。

RLCP是另一种有效构造：先从正确归一化kernel产生随机局部中心，再以该中心计算权重。它有边际及放松局部保证，但不能以“随机带宽”代替其完整采样与归一化步骤。CDF画像空间中的核也不自动是可采样概率密度。本轮不把任意ΦR核直接称为已满足RLCP条件。[Hore与Barber：RLCP](https://arxiv.org/html/2310.07850v5)

## 5. 推荐主方法：ΦR-SLCP

### 5.1 三个独立角色

保持与v2相同的基本数据总量作为下一轮设计参考：

| 数据 | 建议角色 | 参考数量 |
|---|---|---:|
| D_profile | 拟合/冻结预测器、结果分布与画像；预测器若已冻结则不重训 | 800 |
| D_local | 连续核局部残差分位数学习；原support预算改为实际学习局部形状 | 400 |
| D_cal | 只进行最终外层秩校准 | 600 |
| D_eval | 新的独立评价；不选择任何参数 | 后续协议确定 |

这些是新设计数据，不能重新使用已查看的v2数据作独立确认。风险控制扩展需要另有D_risk，不能重复使用D_cal还声称独立检验。

### 5.2 第一步：从连续画像邻域学习局部分位数

D_local中计算R_j。固定alpha=0.1，定义：

\[
\widehat F_{\rm local}(r\mid x)
=\sum_{j\in D_{\rm local}}p_j(x)\mathbf1\{R_j\le r\},
\qquad
q_0(x)=Q_{0.9}\{\widehat F_{\rm local}(\cdot\mid x)\}.
\]

带宽规则见第6节，不能利用D_cal或D_eval的标签。这里的q0是学习到的score形状函数，**本身没有90%覆盖承诺**。因此不需要为这个训练分位数加入查询的infinity原子。

当同T的D_local至少有一个有限残差、核有正质量时，q0有限。局部ESS不足时保留估计并标记信息不足；它不会自动使外层覆盖失效，也不要求删除该查询。

若同T的D_local为空，预定使用独立画像模型的有限0.9残差分位数；模型也不可用时使用固定q0=0并标记无局部信息。此类退化不能宣传为ΦR发挥作用，但仍进入完整评价。

### 5.3 第二步：中心化残差做一次90% conformal

在D_cal上计算：

\[
U_i=|Y_i-\widehat f(X_i)|-q_0(X_i).
\]

对每个T=t，设有n_t个U值，取：

\[
k_t=\left\lceil(n_t+1)\,0.9\right\rceil,
\qquad
\Delta_t=U_{(k_t)};
\quad k_t>n_t\text{ 时取 }+\infty.
\]

最终输出：

\[
q_3(x)=\max\{0,q_0(x)+\Delta_t\},
\qquad
C_3(x)=[\widehat f(x)-q_3(x),\widehat f(x)+q_3(x)].
\]

Delta可以为负；不强制改成非负，否则会额外保守。max(0,.)只是把负半径对应的空接受集扩大为单点集，不降低覆盖。不能在此后截断大区间。

### 5.4 覆盖证明与有限性

条件于D_profile、D_local，q0、画像和预测器均为固定函数。D_cal与未来查询在T=t内的U是交换的，故：

\[
P\{U_*\le\Delta_t\mid D_{\rm profile},D_{\rm local},T_*=t\}
\ge0.9.
\]

该事件蕴含Y*属于C3，因此取得同样的T条件边际覆盖。概率平均于随机D_cal与未来查询；不保证给定某个已实现D_cal的覆盖必定>=90%。该证明是split CP秩论证在局部中心化score上的直接应用，不是新覆盖定理。

只要q0及残差有限、同T最终校准数量至少9，外层阈值有限；不需要每个局部邻域99点，也没有两个95%候选的union-bound成本。T根最终校准量少于9时仍诚实保留infinity。

### 5.5 为什么外层不会像Method B一样抹掉画像

对同T的x、x'，当两个最终半径都为正：

\[
q_3(x)-q_3(x')=q_0(x)-q_0(x').
\]

外层只加一个公共校正量，保留由ΦR权重学得的局部差异；它没有用根组的一个原始残差分位数替换整个q0函数。

这是一项结构性质，不是效率保证。如果q0估计很差，Delta可能很大；若q0本来近乎常数，最终区间仍可能缺乏局部性。需要报告实际q0变化和画像影响，不能只证明权重公式里出现了ΦR。

## 6. Adaptive neighborhood与minimum effective sample size

### 6.1 不采用单一“ESS>=某数即有效”的规则

对归一化权重p_j，记录：

\[
N_{\rm eff}(x)=\frac1{\sum_jp_j(x)^2},
\quad p_{\max}(x)=\max_jp_j(x),
\quad B_R(x)=\sum_jp_j(x)\widehat d_R(x,X_j).
\]

三个量分别描述信息集中度、单点支配和分布混合偏差。仅扩大带宽使ESS足够，会重复v2丢掉画像差异的问题。

在直接LCP中另记录self质量p_infinity和实际beta，检查第4节有限性条件。ΦR-SLCP的内层是分位数学习，不具有该self原子；外层则使用标准等权CP的精确阶次。两者不能混用有限性判据。

### 6.2 建议的操作规则

下一版协议可将以下作为固定、非结果驱动的初始规则：

- 尾部操作预算b_tail=5，目标N_eff,min=ceil(b_tail/alpha)=50。
- p_max<=0.05，避免一个观测支配。
- B_R<=0.10，控制平均估计CDF距离；该0.10是工程容忍值，不表示额外误覆盖预算。
- 残差CDF距离的候选带宽为正的有限列表，例如{0.025,0.05,0.10,0.20,0.40}；这些是待下一版协议冻结的设计建议，本轮不选择或验证最佳值。

对每个x，只用D_profile和D_local的X/冻结画像检查候选带宽，选择同时满足上述条件的最小带宽。

若没有可行带宽，**不把整个T群合并、不删除查询、不扩大到无穷带宽强行通过**；选择候选中B_R最小的带宽，并按固定顺序解决并列，输出low_local_information。仍计算q0并参加同一个外层校准。样本不足带来的估计波动由诊断与后续评价披露。

“5个上尾名义ESS”只是可解释的操作预算，不是实际有5个独立尾部观测，也不是quantile精度或覆盖认证。与v2的99相比，50的变化来自单次90%外层结构；不是把旧方法的支持门槛事后调低。

### 6.3 更严格的信息需求如何定义

条件于独立画像训练数据和D_local的协变量，若权重不利用该池的Y选择，则在固定r处，独立残差指标的加权Hoeffding界为：

\[
P\{|\widehat F_{\rm local}(r\mid x)-\overline G_x(r)|>\eta\mid X_{1:m}\}
\le2\exp\{-2N_{\rm eff}(x)\eta^2\},
\]

其中overline G是相应真实CDF的加权混合。对L个预定网格点、J个候选带宽同时控制，可取：

\[
\eta(x,h;\delta)
=\sqrt{\frac{\log(2LJ/\delta)}{2N_{\rm eff}(x,h)}}.
\]

这是本设计对有界独立加权均值的直接推导。它不是对数据相关权重、相关空间误差或所有查询连续域都自动有效的界。

若任务需要CDF随机误差不超过eta_req，那么对应的信息要求为：

\[
N_{\rm eff,req}
\ge\frac{\log(2LJ/\delta)}{2\eta_{\rm req}^2}.
\]

例如仅一个固定r、一个带宽、delta=0.05、eta_req=0.05，这个保守界已要求约738的ESS。不能把50或99宣传成同等严格认证。对随机分位数使用该界，至少需要覆盖相应的整个网格，并处理网格之间和尾部的误差。

因此应分别报告：**操作ESS目标、随机误差界是否有信息、最终秩覆盖**。不把高概率局部精度认证设为主方法的几何准入门槛。

## 7. 画像误差怎样影响校准目标

以下G_x必须是与被校准总体一致的真实条件score CDF。将它等同于第3节的当前T干预CDF，需要相应识别条件；若两种目标不一致，还须加入目标分布失配项，不能直接套用下面的界。当前主覆盖目标是同分布未来观测，不是任意未观测潜在结果或新干预政策下的覆盖。

假设在相关支持域内存在真正的统一估计界：

\[
\sup_x\|\widehat G_x-G_x\|_\infty\le\epsilon_{\rm profile}.
\]

由三角不等式，真实加权混合与查询CDF的偏差满足：

\[
\|\overline G_x-G_x\|_\infty
\le B_R(x)+2\epsilon_{\rm profile}.
\]

结合第6节随机误差，固定查询的局部CDF误差可分解为：

\[
\underbrace{B_R(x)}_{\text{邻域混合偏差}}
+\underbrace{2\epsilon_{\rm profile}}_{\text{画像估计误差}}
+\underbrace{\eta(x)}_{\text{有限样本误差}}
+\underbrace{e_{\rm grid,tail}(x)}_{\text{数值与尾部误差}}.
\]

这解释了为什么比起“处理效应接近”，校准更需要score分布接近：区间覆盖直接取决于G_x(q)，而不是tau_mean。

该分解不直接给出PhiR-SLCP的逐点90%保证。外层Delta的偏移也会影响局部覆盖；将CDF误差换成分位数误差还需要目标分位数附近的密度下界、平滑性及经验分位数跳跃控制。未知epsilon不能用拟合内误差替代，更不能据此宣称分布无关局部coverage。

## 8. Risk-Controlled LCP应怎样定义

### 8.1 先区分三种“控制”

| 说法 | 实际内容 |
|---|---|
| ESS-aware | 信息量诊断/带宽规则，没有独立的风险保证 |
| Conformal Risk Control，CRC | 在有界、单调、适当连续的损失族及交换性下控制期望风险 |
| 高概率局部风险控制 | 对预定区域/局部重加权总体，在独立风险样本上给出1−delta的风险认证，需要额外数据及选择校正 |

CRC的期望风险控制不能直接改称“给定每个位置，90%覆盖以95%置信度成立”。本设计中的Risk-Controlled扩展明确采用第三种含义，并限制在有限的预定画像邻域。[Angelopoulos等：Conformal Risk Control](https://arxiv.org/abs/2208.02814)

### 8.2 推荐扩展：保留主区间，必要时只扩张

先用第5节形成C3；其全部训练和calibration数据记作D_base。独立于D_base另设D_risk。

在D_base确定有限个画像锚点a=1,...,M和核函数h_a(x)∈[0,1]，包含相应T约束。预定有限个区间扩张函数lambda_l(x)>=0，l=1,...,J：

\[
C_l(x)=[\widehat f(x)-q_3(x)-\lambda_l(x),
\widehat f(x)+q_3(x)+\lambda_l(x)].
\]

候选函数和锚点都必须在查看D_risk之前确定；不把无限多个查询各当作无需校正的独立检验。无论选择哪个候选，它逐点包含C3，因此不会降低主方法的边际覆盖，不需要重新给两条路径各分5%错误率。

### 8.3 可审计的局部风险上界

定义固定锚点邻域总体的风险：

\[
\mathcal R_a(l)
=\frac{E[h_a(X)\mathbf1\{Y\notin C_l(X)\}]}{E[h_a(X)]},
\qquad E[h_a(X)]>0.
\]

在m个独立D_risk观测上估计分子A_hat(a,l)和分母B_hat(a)。取：

\[
b=\sqrt{\frac{\log\{M(J+1)/\delta\}}{2m}},
\]

\[
U_{a,l}=
\begin{cases}
\min\{1,(\widehat A_{a,l}+b)/(\widehat B_a-b)\},
&\widehat B_a>b,\\
1,&\widehat B_a\le b.
\end{cases}
\]

对M×J个分子和M个分母使用单侧Hoeffding界及union bound，可直接得到：

\[
P_{D_{\rm risk}}\{\forall a,l:\mathcal R_a(l)\le U_{a,l}
\mid D_{\rm base}\}\ge1-\delta.
\]

这是一种保守但明确的设计推导；成熟实现可采用有效的Learn-then-Test检验提高功效，不能直接把ESS当成二项样本量塞进Clopper-Pearson。[Angelopoulos等：Learn then Test](https://arxiv.org/abs/2110.01052)

如有候选满足全部预定U<=0.1，按事先固定的最小扩张顺序选取。若没有，保留C3并标记local_risk_not_certified；不删该查询，不额外宣称局部风险保证。若业务必须取得此高概率局部认证，可能需要更多独立风险数据或允许极宽乃至无限区间；不能把信息缺口隐藏为“适应性”。

该保证针对固定画像重加权总体，不是每个x的条件风险。小邻域的B_hat很小，上界可能无信息；风险控制并不能免费解决局部数据不足。

### 8.4 为什么不把它立即设为论文主方法

v2已经表明，高概率支持/风险认证若成为局部化的硬门槛，会再次吞掉分辨率。主方法先保留清晰的有限样本边际保证和连续局部形状；局部风险控制作为需要额外数据的扩展和诊断，证据不足时明确未认证。

## 9. 覆盖保证的完整条件

| 层面 | 必要条件/限制 |
|---|---|
| ΦR-SLCP边际覆盖 | 所有score学习与参数选择独立于最终D_cal；D_cal和查询交换；正确阶次及完整输出 |
| T条件覆盖 | 在每个固定T中执行外层CP，未来查询来自同一T条件总体；不跨T临时补样本 |
| 画像的因果解释 | 一致性、无干扰、合法干预、重叠以及所需的无混杂/识别假设；随机化仿真只覆盖简单情形 |
| 局部效率与近似局部覆盖 | score分布匹配、画像精度、足够局部信息；分位数稳定性还需局部密度等条件 |
| PAC邻域风险认证 | 独立D_risk、预定有限候选/锚点、同时选择校正、正邻域质量 |
| 真实空间迁移 | 空间相关、时间变化、干扰或covariate shift可能破坏交换性；需要独立论证，地理核不能代替它 |

画像模型失配通常影响q0质量及效率；在主方法的独立拆分条件成立时，不要求画像估计正确才能取得T条件边际覆盖。相反，错误使用D_cal标签学习或选择q0会破坏这个简单证明。

默认不使用未经论证的cross-fitting、在最终校准集上调带宽直到Coverage达标，或测试后按最好width选择方法。

## 10. 相对GeoSIMCP的真正创新与必要消融

本仓库GeoSIMCP权重实现以地理和属性距离构造：

\[
d_{\rm joint}^2=\lambda d_S^2+(1-\lambda)d_{\rm attr}^2,
\qquad w=\exp\{-d_{\rm joint}^2/(2h^2)\}.
\]

依据：[本地权重实现](geoconformal-main/geoconformal/geocp/weights.py)及[作者仓库](https://github.com/pengtum/geoconformal)。不同论文/代码版本的细节不应混称为同一个精确算法；本节比较相似性目标，不声称所有GeoSIMCP变体都缺少有效性处理。

GCCP的潜在贡献是：

1. **任务目标改变。** 用冻结预测器的干预score分布水平定义可校准相似性，明确均值、尺度、基准噪声和预测器偏差的不同角色。
2. **几何与有效校准连接。** 让画像权重先学习局部分位数，再通过独立秩校正保留局部形状；不以任意加权分位数冒充分布无关CP。
3. **信息预算与误差桥接。** 同时控制/诊断邻域混合距离、权重集中度和随机误差，说明为什么“效果相同”不足，以及信息不足时哪些保证仍成立。
4. **可反驳的因果价值主张。** 在观测条件残差CDF与干预残差CDF相同的随机化环境，不能声称因果建模天然有额外收益。要证明因果价值，需要另有识别场景、干预/环境迁移目标及相应基线。

已有LCP、SLCP和分布型CP已经研究局部加权或条件分布校准；普通条件残差模型也是强基线。论文不能把这些组件重新命名为GCCP原创。[Chernozhukov等：Distributional Conformal Prediction](https://pmc.ncbi.nlm.nih.gov/articles/PMC8640792/)

下一轮必须预注册以下比较，而非运行后补选最有利对照：

- 同一SLCP程序下：Spatial、attribute、Φμ、ΦD、ΦR。
- Global90、T-only90、T-only95，以及原冻结Method B。
- ΦR驱动的完整calibrated LCP，用来检验两种有效局部化实现。
- 普通条件残差分布相似性，以及直接用估计G_x分位数构造的conformal基线；区分“因果语义”“学到残差分布”“选择相似邻居”各自作用。
- 同T内打乱画像对应关系的预定负对照，检验画像—查询联系是否实际影响q0和最终区间。

除coverage、width、interval score、infinity、ESS、CDF匹配外，应新增：q0的查询间变化、外层Delta、低局部信息比例、带宽选择分布、B_R、最终区间对画像替换的变化，以及保守校正是否抵消了局部收益。

## 11. 本阶段交付与后续门槛

当前选择为：**ΦR-SLCP作为v3主候选；完整ΦR-LCP作为理论参照与比较；独立样本的局部风险控制作为可选扩展。**

下一阶段如获授权，应先冻结数据拆分、邻域规则、候选带宽、操作ESS、失败输出、方法列表和种子，再做实现检查和新仿真。不得从v2已查看样本选择v3最优参数，也不直接进入Seattle或PM2.5。

本文没有运行新实验，也没有证明v3已经获得效率收益。v2保留为完整的负面开发证据；v3是否解决局部预测区间问题仍是待验证命题。
