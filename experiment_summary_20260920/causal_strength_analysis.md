# 因果强度分析

## 1. Two causal-strength definitions in the project

本项目目前存在两类不可直接混用的“因果分数”。

### GCCM causal score

GCCM（Geographical Convergent Cross Mapping）沿用收敛交叉映射思想，检查候选变量能否在空间邻域和不同 library size 下稳定地恢复目标变量信息。现有真实数据流程以结果变量为目标；模拟研究的相关流程也曾以绝对残差为目标。

代码将通过收敛判据后的终点 cross-map 相关系数记为因果强度：

$$
S_j^{\mathrm{GCCM}}=
\begin{cases}
\rho_j(L_{\max}), & \text{通过收敛门控}\\
0, & \text{未通过收敛门控}
\end{cases}
$$

其中 $\rho_j(L_{\max})$ 是最大 library size 下候选变量 $X_j$ 对目标的 cross-map skill。该分数强调空间交叉映射的收敛性，不是条件互信息值。

### CMI residual-scale score

最新 CMI 流程不是直接估计特征对原始响应值的因果效应，而是检验特征与基础模型绝对残差尺度之间的条件依赖：

$$
A_{\mathrm{model}}=|Y-\widehat{Y}|
$$

$$
I(X_j;A_{\mathrm{model}}\mid Z)
$$

这更贴近 conformal 区间的局部尺度问题：如果 $X_j$ 在控制空间坐标及其他协变量后仍与绝对残差相关，则它可能有助于确定局部预测区间宽度。

因此：

- **GCCM causal score**：基于空间 cross mapping 的终点相关强度，并受收敛门控约束。
- **CMI residual-scale score**：候选特征对绝对残差尺度的条件互信息，经置换检验和 BH 多重检验校正。
- 两类分数的量纲、目标变量、显著性处理和解释均不同，不能放在同一数值列中比较大小。

## 2. Conditional Mutual Information

条件互信息定义为：

$$
I(X;Y\mid Z)=
\iiint p(x,y,z)
\log\frac{p(x,y\mid z)}
{p(x\mid z)p(y\mid z)}
\,dx\,dy\,dz
$$

一般记号中：

- $X$：candidate feature，待评价的候选特征；
- $Y$：prediction target；
- $Z$：conditioning variables，用于控制空间位置或其他协变量。

在本项目的 **residual-scale CMI** 实现中，为避免概念混淆，应进一步明确：

- $X=X_j$：候选属性；
- $Y=A_{\mathrm{model}}=|y-\widehat y|$：基础模型的绝对残差，而不是原始 PM2.5 或房价；
- $Z$：空间坐标以及除 $X_j$ 外的其他属性。

CMI 由 k-nearest-neighbor 型估计器计算，并通过置换检验生成 $p$ 值；多个特征的 $p$ 值再使用 Benjamini–Hochberg（BH）程序得到 $q$ 值。

## 3. Causal weight normalization

若特征同时满足：

$$
\mathrm{CMI}_j>0,\qquad q_j<0.05
$$

则进入因果特征集合 $E$，其归一化权重为：

$$
\omega_j=
\frac{\mathrm{CMI}_j}
{\sum_{k\in E}\mathrm{CMI}_k}
$$

不满足条件的特征权重为 0。若 $E$ 为空，现有 real-data causal-only CMI 流程回退到 Raw CP，而不是强行构造因果距离。

历史 Seattle GCCM 流程也将非负 GCCM 分数归一化用于属性距离，但其原始分数不是 CMI，不应称为 CMI 权重。

## 4. Seattle: GCCM causal scores

来源：

**geoconformal-main\GCCP_Seattle\outputs\gccm\seattle_gccm_causal_weights_outer_train_v1.csv**

| Variable | GCCM causal score | Rank |
|---|---:|---:|
| grade | 0.3105836877 | 1 |
| view | 0.2961501470 | 2 |
| sqft_living | 0.2709033700 | 3 |
| sqft_lot | 0.1713034675 | 4 |
| waterfront | 0.1644551465 | 5 |
| bathrooms | 0.1432196440 | 6 |
| age | 0.0555661413 | 7 |
| condition | 0.0000000000 | 8 |

说明：

- Seattle 表中的数值是 outer-train 范围内形成的 GCCM 分数。
- condition 未通过相应门控或终点强度为 0，因此排在最后。
- 历史 GCCP 使用这些分数构造因果加权属性距离。

## 5. PM2.5: GCCM causal scores

来源：

**geoconformal-main\GCCP_PM2.5\outputs\gccm\pm25_gccm_causal_weights_v1.csv**

| Variable | GCCM causal score | Rank |
|---|---:|---:|
| aod_sat | 0.5822393682 | 1 |
| sp | 0.3146099851 | 2 |
| dem | 0.2910348061 | 3 |
| r | 0.2754979853 | 4 |
| blh | 0.2524015262 | 5 |
| tp | 0.2135712280 | 6 |
| ndvi | 0.0980783120 | 7 |

说明：

- AOD 的 GCCM 分数最高，之后依次为地表气压、高程、相对湿度、边界层高度、tp 和 NDVI。
- 该排序描述 GCCM cross-map skill，不代表这些变量对残差尺度的 CMI 排序。

## 6. PM2.5: residual-scale CMI

来源：

**experiments\real_data\20260918_stage14_pm25_causalonly_cmi_exploratory_v1\cmi_feature_results.csv**

| Variable | Raw CMI | BH q value | Selected |
|---|---:|---:|---|
| blh | 0.0241900692 | 0.434 | No |
| tp | 0.0214636498 | 0.210 | No |
| dem | 0.0182877809 | 0.434 | No |
| sp | 0.0176105389 | 0.434 | No |
| r | 0.0145870327 | 0.210 | No |
| aod_sat | 0.0138433692 | 0.650 | No |
| ndvi | -0.0105888160 | 0.650 | No |

结果解释：

1. 在 BH $q<0.05$ 且 raw CMI $>0$ 的预设规则下，没有变量被选中。
2. 因此本次 PM2.5 residual-scale CMI 没有产生可用于因果距离的非零归一化权重。
3. 探索性 causal-only 方法按规则回退到 Raw CP。
4. 该结果不否定变量与原始 PM2.5 的关系；它只说明在当前样本、残差定义、条件集和检验功效下，没有检测到足够稳定的“额外残差尺度信息”。

## 7. Relevant scripts and outputs

| Method | Inputs | Target | Script | Output |
|---|---|---|---|---|
| Seattle GCCM | 8 个房屋属性、空间坐标 | 房价相关目标 | GCCP Seattle notebook / GCCM workflow | Seattle GCCM weights CSV |
| PM2.5 GCCM | dem, sp, tp, blh, r, aod_sat, ndvi、空间位置 | PM2.5 | **geoconformal-main\GCCP_PM2.5\03_gccm\run_pm25_gccm_train_only_v1.R** | PM2.5 GCCM weights CSV |
| General CMI benchmark | 候选特征、条件变量 | 绝对残差尺度 | **causal_strength_benchmark_v1\stage17_cmi_worker_v1.py** | CMI feature/result files |
| PM2.5 real-data CMI | 7 个属性；条件集含坐标和其他属性 | $|y-\widehat y|$ | **causal_only_cp_v1\run_stage14_realdata_causalonly_cmi_exploratory_v1.py** | cmi_feature_results.csv |

## 8. Audit conclusion

目前仓库已经包含：

- Seattle 和 PM2.5 的 GCCM 特征强度与排序；
- PM2.5 residual-scale CMI 的原始分数、BH q 值和选择结果；
- 将 causal weights 用于局部预测区间的历史 GCCP / GeoCausalCP 实现。

但需要保持结论边界：PM2.5 的 GCCM 排序与 residual-scale CMI 检验回答的是不同问题。前者不能替代后者的显著性结论，后者的“无特征入选”也不能推翻前者对原始目标关系的排序。

