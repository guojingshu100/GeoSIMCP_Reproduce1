# GCCP final method specification

日期：2026-10-12。版本：final-method v1。状态：**FROZEN**。

**最终方法名称：ΦR-SLCP。论文定位：Residual distribution driven spatial conformal calibration。**

本文件冻结最终论文方法定义，不运行实验、不生成真实数据拆分、不修改代码及历史结果。**真实数据实验前禁止修改本文件规定的统计方法、参数、数据角色、baseline 和评价规则。** 真实数据的任务、输入字段、点预测器完整配置及拆分细则继续执行 [REAL_DATA_PROTOCOL_DESIGN.md](REAL_DATA_PROTOCOL_DESIGN.md)，其封存 SHA256 为：

```text
1469ce66ed5574df9ad3e923da44e3529e8533b6be7541faf5fd71e6ba73d9a4
```

本文件统一四类角色和论文方法名称，不改变该协议的五个互斥物理子集。该协议第 11 节关于另立设计版本的宽泛表述，在本次冻结中收紧：不得在真实数据执行前自行改版或替换方法。未来若用户明确启动新方法研究，应另立版本并保留本封存件，不能追溯改写本方法或根据结果选择版本。

## 1. 研究对象与名称

给定协变量和空间位置 \(V=(X,S)=v\)，研究冻结点预测器的条件预测误差分布。ΦR 表示 **residual distribution profile**，SLCP 表示本规范的局部分位数学习与独立外层校正程序。

GCCP 保留为研究框架历史名称；最终算法名固定为 **ΦR-SLCP**。Residual-profile SLCP 是同一方法的描述性别名，不是额外 baseline。

研究问题：条件残差分布定义的校准相似性，能否相对地理和普通属性相似性，在覆盖可接受时改善局部预测区间的宽度及 Interval Score？

**不再宣称 causal semantic gain。** 本方法不识别 treatment effect，不以反事实差分定义画像，不宣称因果迁移或未知机制漂移下的覆盖优势；不得将同信息普通残差画像与 ΦR 的相同输出包装为因果增量。

## 2. 数据角色与允许的信息

四类角色冻结为 train、profile-learning、calibration、test。profile-learning 内部保留两个互斥子集，不能复用为同一个标签池。

| 四类角色 | 物理集合与原协议对应 | 目标比例 | 标签允许用途 |
|---|---|---:|---|
| train | \(D_{\mathrm{train}}=D_{\mathrm{fit}}\) | 45% | 拟合点预测器、预处理和 score 尺度 |
| profile-learning | \(D_{\mathrm{PL}}=D_G\mathbin{\dot\cup}D_Q\)；\(D_G=D_{\mathrm{profile}}\)，\(D_Q=D_{\mathrm{local}}\) | 15%+10%=25% | \(D_G\) 拟合残差画像；\(D_Q\) 学习局部分位数 |
| calibration | \(D_{\mathrm{cal}}\) | 15% | 最终 conformal 校正；直接 baseline 的校准阈值 |
| test | \(D_{\mathrm{test}}\) | 余下约 15% | 区间全部冻结后进行最终评价 |

五个物理集合按独立单位互斥；重复房屋/站点和空间块遵守原协议的分组约束。没有分组约束时前四个物理集合分别取比例的 floor，test 接收余数；有分组时执行原协议的最接近累计目标规则，不能拆开单位或重新抽 seed 凑样本量。

train 中学习的插补、特征中心/尺度及点预测器供所有方法共用。主方法不使用 train 的 in-sample residual 拟合画像。\(D_Q\) 标签只进入固定权重下的局部分位数；\(D_{\mathrm{cal}}\) 标签不能进入画像、距离、带宽选择或 \(q_0\)。test 标签最后解封。

## 3. Score 与 residual profile：\(G(r\mid v)\)

令 \(Z=g(Y)\) 为原协议预定的预测尺度，\(\widehat f(v)\) 只在 train 上拟合。定义

\[
c=\max\{\operatorname{IQR}(Z_{\mathrm{train}}),10^{-6}\},\qquad
E=\frac{Z-\widehat f(V)}{c},\qquad R=|E|.
\]

IQR 为线性插值的 .75 分位数减 .25 分位数；\(c\) 一经训练即固定，对所有角色和所有方法一致。

条件于已经冻结的训练状态，目标画像为

\[
G(r\mid v)=\Pr\{R\le r\mid V=v\},\quad r\ge0;
\qquad G(r\mid v)=0,\quad r<0.
\]

算法画像为 \(\Phi_R(v)=\widehat G(\cdot\mid v)\)。它保留条件 score CDF 的水平，刻画当前预测器的偏差和噪声尺度。它是预测器相关的分布对象，不能在更换预测器后原样复用。

### 3.1 冻结的画像估计器

只在 \(D_G\) 上使用 out-of-fit signed residual \(E_i\) 拟合 Gaussian location-scale 模型：

\[
E\mid V=v\approx N(\delta(v),\sigma^2(v)),\qquad
\delta(v)=B(v)^\top\beta,\quad \sigma(v)=\exp\{B(v)^\top\gamma\}.
\]

普通属性及固定缺失指示的标准化向量为 \(z_A\)，地理标准化向量为 \(z_S\)。插补、中心和 IQR 全部来自 train；特征 IQR 为零则置 1，测试越界值不裁剪。固定设计基为

\[
B(v)=[1,z_A,z_S,z_S^2,\{z_{S,u}z_{S,w}:u<w\},\tanh(2z_S)].
\]

Seattle 使用二维 UTM 地理输入，PM2.5 使用原协议的三维球面坐标。特征名单、坐标转换、预处理及 RF 完整配置均绑定上述封存协议，不另选模型或特征。

最小化 \(\operatorname{mean}[\log\sigma_i+\tfrac12(E_i-\delta_i)^2/\sigma_i^2]\)。沿用单次确定性初始化：OLS mean、log-SD 截距为 log RMS、其余为零；L-BFGS-B 参数固定 maxiter=2000、maxls=50、ftol=1e−12、gtol=1e−6。要求优化成功且最大绝对梯度≤1e−4，不做多起点或重复拟合到成功。

令 \(\Phi\) 为标准正态 CDF，则对 \(r\ge0\)，

\[
\widehat G(r\mid v)=
\Phi\!\left(\frac{r-\widehat\delta(v)}{\widehat\sigma(v)}\right)
-\Phi\!\left(\frac{-r-\widehat\delta(v)}{\widehat\sigma(v)}\right).
\]

这是一种冻结的分布估计器，不假设真实误差确实正态；多峰、偏态、重尾及未观测异质性可能导致画像失配。不得用 test 诊断替换 learner。

## 4. Residual distribution distance：\(d_R\)

连续 CDF 几何的理想距离为

\[
d_R^{\infty}(v,v')=\sup_{r\ge0}|\widehat G(r\mid v)-\widehat G(r\mid v')|.
\]

**实际程序冻结使用原有网格近似**，不将其改为新的距离估计器：

\[
r_k=12k/2400,\quad k=0,\ldots,2400,\qquad
d_R(v,v')=\max_k|\widehat G(r_k\mid v)-\widehat G(r_k\mid v')|.
\]

距离取值在 [0,1]，不以均值效应差代替完整 CDF 差。所谓连续相似性指 CDF 画像及软权重，不是宣称数值网格等于精确连续 supremum。拟合模型的网格/尾部诊断不能当作未知真实 CDF 的估计误差保证。

## 5. Local weighting：固定决策规则

对查询 \(v\) 和 \(j\in D_Q\)，定义

\[
w_j(v;h)=\exp\!\left[-\frac{d_R(v,V_j)^2}{2h^2}\right],\qquad
p_j(v;h)=\frac{w_j(v;h)}{\sum_{\ell\in D_Q}w_\ell(v;h)}.
\]

归一化使用减去最大 log-weight 的数值稳定计算。真实数据只有一个活动层 \(H\equiv0\)；不人为建立 treatment，不做 Mondrian 合并。主方法不叠加地理 kernel，固定 gamma_spatial=0；空间信息通过 \(G(r\mid X,S)\) 进入。

候选带宽按顺序固定为 \(\mathcal H=(.025,.05,.1,.2,.4)\)。定义

\[
\mathrm{ESS}(v;h)=\frac1{\sum_jp_j^2},\qquad
p_{\max}(v;h)=\max_jp_j,\qquad
B_R(v;h)=\sum_jp_jd_R(v,V_j).
\]

选择最小可行 \(h\)，同时满足 ESS≥50、\(p_{\max}\le.05\)、\(B_R\le.10\)，边界容差固定 1e−12。若没有可行候选，选择 \(B_R\) 最小的候选，平局取顺序中的首个，并标记 low information。不得扩展候选、降低门槛、合并成根组、删除查询或用 coverage 选择带宽。

带宽选择只使用冻结画像和协变量，不使用 \(D_Q\) 的 score 值，也不使用 calibration/test 标签。ESS 是局部学习权重的有效样本量，不是外层秩校正样本量；\(B_R\) 是拟合画像中的匹配诊断，不是风险控制证书。

## 6. Local quantile estimation

固定 \(\alpha=.10\)。在已选择的权重下，定义

\[
\widehat H_v(r)=\sum_{j\in D_Q}p_j(v)\mathbf1\{R_j\le r\},\qquad
q_0(v)=\inf\{r:\widehat H_v(r)\ge1-\alpha\}.
\]

沿用稳定 score 排序和累计权重首次达到 .9 的规则；本学习步骤不加入查询 infinity atom。先冻结画像及权重，再读取 \(D_Q\) score 学习 \(q_0\)；这不等同于使用最终 calibration 标签。

最低数值支持沿用 \(n_G\ge50\)、\(n_Q\ge1\)、\(n_{\mathrm{cal}}\ge9\)。\(n_Q<50\) 时如实标记原 ESS 门槛不可满足，不改门槛。沿用原协议无局部样本时的模型 folded-normal .9 分位数回退并标记，但该状态不满足正式任务最低支持，不能计作通过支持检查的实验。模型非有限或优化失败报告失败，不更换 seed 或模型补救。

## 7. Independent outer conformal correction

在 \(\widehat f,c,\widehat G\)、带宽决策及 \(q_0\) 全部冻结后，才读取独立 \(D_{\mathrm{cal}}\) 标签：

\[
A_i=R_i-q_0(V_i),\quad i\in D_{\mathrm{cal}},\qquad
k=\left\lceil(n_{\mathrm{cal}}+1)(1-\alpha)\right\rceil.
\]

令 \(A_{(k)}\) 为第 \(k\) 个顺序统计量，定义

\[
\Delta=\begin{cases}A_{(k)},&k\le n_{\mathrm{cal}},\\+\infty,&k>n_{\mathrm{cal}},\end{cases}
\qquad q(v)=\max\{0,q_0(v)+\Delta\},
\]

\[
C_Z(v)=[\widehat f(v)-c\,q(v),\ \widehat f(v)+c\,q(v)].
\]

秩由精确 .9 有理数计算，避免浮点 ceil 偏差。允许负 \(\Delta\)，只在最终半径下截断为零；不把 \(\Delta\) 强制非负，不设半径上限，不用 test coverage 调整校正。若 g 可逆且单调递增，原目标区间为 \([g^{-1}(L_Z),g^{-1}(U_Z)]\)，须遵守预先固定且已核实的单位转换。

外层使用全部实际 \(D_{\mathrm{cal}}\)，不是按查询权重再次加权的秩，不按画像重新分组。\(\alpha=.10\) 时 \(n_{\mathrm{cal}}\ge9\) 可满足有限秩条件；有限输入和满足支持并不自动保证局部效率。

### 7.1 覆盖声明的边界

条件于独立 train/profile-learning 阶段，若 calibration 和新查询的 \(A=R-q_0(V)\) 可交换，则 split-conformal 秩论证给出

\[
\Pr\{Z_{\mathrm{new}}\in C_Z(V_{\mathrm{new}})\}\ge .90.
\]

概率对 calibration 和新查询的随机性取平均；这是边际覆盖，不是每个位置、每个子群或固定已实现 calibration 集上的条件覆盖。画像模型正确性不是该边际论证的必要条件，但画像失配会损害局部匹配和效率。

空间相关、协变量/机制漂移或不独立的数据流可能破坏交换性。本方法没有额外 importance weighting、block conformal 或非交换性风险控制保证。外层校正和 Gaussian 分布族本身不作为新的覆盖定理或独立创新声明。

## 8. 论文 baseline：四个不同程序

所有主表统一使用以下四个名称，采用共同预测器、预处理、score 单位、角色清单及 test cohort。

| 名称 | 校准程序 | 主要区别 |
|---|---|---|
| CP | \(D_{\mathrm{cal}}\) 原始 R 的 split-conformal 精确秩阈值 | 全球同一半径 |
| GeoCP | \(D_{\mathrm{cal}}\) 原始 R；固定地理 kernel 加查询 self-weight=1 的 infinity atom | 地理相似性 |
| GeoSIMCP | \(D_{\mathrm{cal}}\) 原始 R；固定地理/普通属性 kernel 与相同 self atom | 地理和普通属性相似性 |
| ΦR-SLCP | 本规范第 3–7 节 | 残差 CDF 相似性、局部分位数、独立外层校正 |

CP 阈值为 \(R_{(\lceil .9(n_{\mathrm{cal}}+1)\rceil)}\)，秩超过样本量则为 infinity。GeoCP/GeoSIMCP 对查询 v 的加权分布为

\[
\frac{\sum_{i\in D_{\mathrm{cal}}}K(v,V_i)\,\delta_{R_i}+\delta_{+\infty}}
{\sum_{i\in D_{\mathrm{cal}}}K(v,V_i)+1},
\]

取 .9 step quantile，累计边界容差沿用 1e−9；其中 \(\delta_a\) 为点质量。保留原始 kernel 及固定 self-weight 的相对尺度，不能只归一化 calibration 权重而改变 self atom。任意局部 kernel 加 atom 不自动具有 CP 的分布无关覆盖保证。

令 \(d_s=d_{\mathrm{geo}}/d_{\mathrm{ref}}\)，\(d_{\mathrm{ref}}\) 为 train 正地理成对距离中位数；无正距离时为 1 km 并标记。\(d_a=\sqrt{p^{-1}\sum_{b=1}^p(z_{A,b}-z'_{A,b})^2}\)，不重复包含坐标。冻结

\[
K_{\mathrm{GeoCP}}=\exp[-d_s^2/(2\cdot .4^2)],\qquad
K_{\mathrm{GeoSIMCP}}=\exp\!\left[-\frac{.5d_s^2+.5(d_a+10^{-8})^2}{2(.4+10^{-8})^2}\right].
\]

Seattle 地理距离为 UTM 平面 km；PM2.5 使用半径 6371.0088 km 的三维球面坐标欧氏弦长，不能称为精确测地距离。上述是事前固定的 baseline 实例，不声称是原作者最佳调参复现。

原真实数据协议要求的 GeoCP-SLCP、GeoSIMCP-SLCP 和 CP-all-residual-budget 继续作为**必须报告的补充预算对照**，不替代四方法主表、不按结果决定是否报告。主方法多使用 profile-learning 标签，故相对直接 baseline 的收益不能全部归因于 CDF 几何；预算及 outer 程序的贡献必须区分。

## 9. Evaluation：三个主要指标

对全部已冻结 test 查询，令区间为 \([L_i,U_i]\)，目标为 \(Z_i\)：

\[
\widehat{\mathrm{Coverage}}=\frac1n\sum_{i=1}^n\mathbf1\{L_i\le Z_i\le U_i\},
\qquad \mathrm{Width}_i=U_i-L_i,
\]

\[
\mathrm{IS}_{.10,i}=U_i-L_i
+20(L_i-Z_i)\mathbf1\{Z_i<L_i\}
+20(Z_i-U_i)\mathbf1\{Z_i>U_i\}.
\]

主要报告 **Coverage、mean Interval Width、mean Interval Score**；同时保留原协议的 median width。主尺度固定为 Seattle 文件 log_price 与 PM2.5 原始浓度；币种/log 底及浓度单位按原协议核实，不跨数据集混合 width/IS。

主比较固定为 ΦR-SLCP vs GeoSIMCP；四方法与补充预算对照完整报告。只有双方经验 Coverage≥90%、无失败/infinity，且 mean Width 与 mean IS 都下降，才记录该任务的联合数值改善。覆盖达标但效率无改善必须如实报告；单切分的数值改善不构成普遍或显著优越性的证明。

Infinity rate、失败率、ESS、外层样本数、\(B_R\)、low-information rate、OOD 和局部信息保留继续作为诊断。保留全部困难查询及无穷宽度/IS，finite-only 统计只能附加展示；失败占位区间不能被计作有效覆盖成功。不得删除失败查询、挑选子群、降低覆盖门槛或依据 test 结果调整参数。

真实数据逐位置真 score CDF 不可观测；只能报告原协议事前固定的区域 CDF/有限网格 Brier/PIT 等代理诊断，不把它们称为 oracle CDF matching error。

## 10. 真实数据执行约束与标签防线

主随机拆分 seed=2026101501；预定空间压力拆分 seed=2026101502。Seattle 10 km UTM 网格、PM2.5 2°经纬网格、特征名单、RF 参数及依赖版本均按封存协议固定。两种拆分分别完整报告，不因主任务结果决定是否运行压力任务，也不把它们当作独立 replicate。

执行时的不可逆标签顺序为：

1. 先按 ID、坐标、标签存在标志和来源审计冻结 cohort、角色及配置；不读取标签值挑 cohort。
2. 只读取 train 标签，冻结预测器、预处理与 c。
3. 只读取 \(D_G\) 标签，冻结画像；根据画像/协变量计算权重及带宽。
4. 读取 \(D_Q\) 标签，冻结 cal/test 的 \(q_0\)、权重和诊断。
5. 读取 calibration 标签，计算 baseline 阈值与 \(\Delta\)，冻结所有 test 区间。
6. 最后解封 test 标签，只评价，不返回改变方法。

替换 calibration 标签必须不改变画像、权重、带宽及 \(q_0\)，允许改变 \(\Delta\)；替换 test 标签必须不改变全部模型和区间。test 协变量仅用于预定 ID/坐标分配及逐查询计算，不能更新全局预处理、画像、距离尺度或参数。

原 Seattle/PM2.5 数据的历史 evaluation-label 暴露状态继续披露：复用属于 retrospective/exploratory 验证，重新 seed 不会产生从未暴露的 holdout。不得将本轮标签隔离等同于撤销历史暴露。

实现只能忠实翻译本封存定义。数据来源、单位、特征可用性或最低支持审计失败时，停止并报告对应任务不可执行；不得借此自行更换模型、核函数、带宽、拆分或 seed。未来执行必须保存本规范及绑定协议、数据/配置/代码与角色清单的哈希；旧实验和报告保持只读。

## 11. 论文贡献与冻结结论

论文方法描述固定为：

> 我们提出 ΦR-SLCP，以冻结预测器的条件残差分布定义空间校准相似性，通过 CDF 距离形成连续局部权重，在独立样本上学习局部分位数，再使用独立 calibration 集进行 conformal correction。方法将邻域与预测误差分布对齐；覆盖声明依赖相应交换性条件，效率优势须通过覆盖、宽度和 Interval Score 联合评价。本文不声称因果语义带来额外收益。

创新定位是 **残差分布驱动的空间校准几何与完整校准程序的结合及其验证**，不是普通 treatment-effect similarity、已证实的 causal gain、新的外层覆盖定理或与自身别名的比较。

本次交付仅冻结方法。没有运行实验、读取真实数据标签作评价、训练模型或修改代码。真实数据实验前，以上方法定义与预定规则禁止修改；尚未取得真实数据有效性结论。
