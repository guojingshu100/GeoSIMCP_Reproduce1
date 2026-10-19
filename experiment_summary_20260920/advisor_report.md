# GCCP / GeoSIMCP 阶段性汇报

## 一、研究背景

本研究面向空间回归中的预测不确定性量化。标准 Conformal Prediction（CP）使用全局校准残差分位数，具有边际覆盖保证，但无法反映不同空间位置的可靠性差异。GeoCP 在校准阶段加入空间核，GeoSIMCP 进一步联合空间相似度与属性相似度，使与测试点更接近的校准样本获得更高权重。

评价指标为：

$$
\mathrm{Coverage}=\frac1n\sum_i\mathbf1\{L_i\le y_i\le U_i\}
$$

$$
\mathrm{Width}=\frac1n\sum_i(U_i-L_i)
$$

$$
\mathrm{IS}_\alpha=(U-L)+\frac2\alpha(L-y)\mathbf1(y<L)+\frac2\alpha(y-U)\mathbf1(y>U)
$$

研究目标是在首先满足目标 Coverage 的条件下，降低 Width 和 Interval Score。

## 二、GeoSIMCP 存在的问题

GeoSIMCP 的联合距离与权重为：

$$
d_{\mathrm{joint}}^2
=
\lambda d_{\mathrm{geo}}^2+
(1-\lambda)d_{\mathrm{attr}}^2,\qquad
w_i=\exp\left(-\frac{d_{\mathrm{joint},i}^2}{2h^2}\right)
$$

主要问题：

1. EUC 或 MND 默认各变量同等重要，可能受无关变量和高维距离稀释影响。
2. 与条件均值相关的变量不一定决定残差尺度，而 conformal 区间真正需要定位的是非一致性分数分布。
3. 局部权重过度集中时，有效校准样本不足；严格加入 test-point $+\infty$ atom 后可能出现无限宽区间。

## 三、GCCP 创新点

**GeoSIMCP = spatial similarity + attribute similarity**

**GCCP = spatial similarity + attribute similarity + causal similarity**

最新因果距离定义为：

$$
d_{\mathrm{causal},i}^2=
\sum_{j\in E}\omega_j(z_{ij}-z_{*j})^2
$$

$$
d_{\mathrm{joint},i}^2=
\lambda d_{\mathrm{geo},i}^2+
(1-\lambda)d_{\mathrm{causal},i}^2
$$

目标是筛除与局部误差尺度无关的变量，并按因果强度重加权剩余变量，减少属性距离噪声，使预测区间针对局部不确定性自适应。

代码位置：

- Seattle GCCP：**geoconformal-main\GCCP_Seattle\03_gccp\03_Seattle_GCCP_causal_similarity.ipynb**
- PM2.5 GCCP-B：**geoconformal-main\GCCP_PM2.5\04_gccp\pm25_gccp_b.ipynb**
- Stage-19B：**geo_causal_cp_v1\run_stage19b_direct_parity_v1.py**

## 四、因果强度计算方法

### 1. GCCM

GCCM 检验候选变量的空间交叉映射能力及其随 library size 的收敛。通过门控时使用最大 library size 下的相关系数：

$$
S_j^{\mathrm{GCCM}}=\rho_j(L_{\max})
$$

否则记为 0。

Seattle GCCM 排名前三：grade 0.3106、view 0.2962、sqft_living 0.2709。  
PM2.5 GCCM 排名前三：aod_sat 0.5822、sp 0.3146、dem 0.2910。

### 2. Residual-scale CMI

当前 CMI 检验候选变量与基础模型绝对残差之间的条件依赖：

$$
I(X_j;A_{\mathrm{model}}\mid Z),\qquad
A_{\mathrm{model}}=|y-\widehat y|
$$

其中 $Z$ 包含空间位置和其他属性。经置换检验与 BH 校正后，满足 raw CMI $>0$ 且 $q<0.05$ 的变量进入集合 $E$：

$$
\omega_j=
\frac{\mathrm{CMI}_j}{\sum_{k\in E}\mathrm{CMI}_k}
$$

PM2.5 当前 CMI 结果中没有变量通过 BH 门槛，因此 causal-only 流程回退 Raw CP。GCCM 原始目标分数与 residual-scale CMI 回答的问题不同，不能混为同一因果证据。

相关代码：

- **causal_strength_benchmark_v1\stage17_cmi_worker_v1.py**
- **causal_only_cp_v1\run_stage14_realdata_causalonly_cmi_exploratory_v1.py**
- **geoconformal-main\GCCP_PM2.5\03_gccm\run_pm25_gccm_train_only_v1.R**

## 五、实验结果

### Seattle

| Method | Coverage | Width | IS |
|---|---:|---:|---:|
| CP | 0.9089 | 35.2126 | 57.7319 |
| GeoCP | 0.9156 | 35.7454 | 55.9272 |
| GeoSIMCP-EUC | 0.9156 | 35.7454 | 55.9272 |
| GeoSIMCP-MND | 0.8933 | 32.3386 | 49.0971 |
| GCCP | 0.8889 | 32.8870 | 49.0295 |

GCCP 的 IS 最低，但 Coverage 为 0.8889，不能据此认为其有效优于 GeoSIMCP-MND。

### PM2.5

| Method | Coverage | Width | IS |
|---|---:|---:|---:|
| CP | 0.8944 | 18.2672 | 27.2210 |
| GeoCP | 0.8732 | 17.5530 | 25.5935 |
| GeoSIMCP-EUC | 0.8662 | 16.9595 | 26.9729 |
| GeoSIMCP-MND | 0.8662 | 17.2729 | 25.6589 |
| GCCP-B | 0.8662 | 17.1210 | 25.5197 |

GCCP-B 在相同 Coverage 的局部方法中 IS 最低，但局部方法均明显低于 0.90。

### Stage-19B

S2 中 GeoSIMCP-EUC 为 0.9010 / 4.6380 / 5.8894，GeoCausalCP 为 0.9032 / 4.6903 / 5.9225。因果方法覆盖率略高，但效率未优于 EUC。  
S4 中 GeoCausalCP 为 0.9078 / 4.6142 / 5.6369，与 GeoCP 一致；Raw-CP 为 0.9010 / 4.4904 / 5.5989，负对照中未出现因果收益。

阶段性结论：GCCP 已完成与 GeoSIMCP 的初步比较，但尚未在有效覆盖约束下表现出稳定优势。

## 六、目前问题

1. Seattle、PM2.5 和 Stage-19B 使用不同的因果分数、核组合和 conformal 原子规则。
2. Seattle 与 PM2.5 的局部高效结果伴随欠覆盖。
3. Stage-19B 尚缺 S1、S3 的同口径 fresh parity。
4. Track B 加入 honest $+\infty$ atom 后部分局部方法产生无限区间。
5. PM2.5 residual-scale CMI 没有变量通过 BH 门槛。
6. Stage-17B LOCI 缺少 signed residual；Stage-17C 早期并行线程错误，恢复结果仅 37/48 checkpoints。
7. PM2.5 文件日期全部为 2018-05-31，与“2018 年年均”的文字描述不一致。
8. 部分历史脚本保留旧绝对路径，影响跨环境复现。

## 七、下一步计划

1. 统一四类方法的模型、数据划分、超参数选择、加权分位数和 $+\infty$ atom 处理。
2. 以 Coverage 可行为首要约束，再在可行方法间比较 Width 与 IS。
3. 补齐 S1、S3，并统一 S1–S4 的随机种子、重复次数、失败和回退统计。
4. 将因果问题明确限定为“对绝对残差尺度的增量信息”，避免把均值相关性误当作尺度信息。
5. 对 CMI/GCCM 权重做 bootstrap 或跨 seed 稳定性分析，并报告特征选择频率。
6. 预先定义空特征集、低有效样本量和无限区间的回退机制。
7. 在真实数据中只使用训练数据估计因果强度，冻结后再进入 calibration/test。
8. 核实 PM2.5 时间口径，清理旧绝对路径并形成统一可复现实验入口。

