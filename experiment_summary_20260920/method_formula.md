# 方法公式与实现位置

## 1. Common conformal foundation

设基础回归模型为 $\widehat f$，校准集非一致性分数为：

$$
r_i=|y_i-\widehat f(x_i)|
$$

标准 CP 使用全局分位数 $q_{1-\alpha}$，预测区间为：

$$
C_\alpha(x_*)=
[\widehat f(x_*)-q_{1-\alpha},\;
 \widehat f(x_*)+q_{1-\alpha}]
$$

局部方法的差别主要体现在：对每个测试点 $x_*$，如何给校准残差 $r_i$ 分配权重并计算加权分位数。

## 2. GeoCP

### Spatial kernel

对校准点位置 $s_i$ 和测试位置 $s_*$：

$$
d_{\mathrm{geo},i}=\lVert s_i-s_*\rVert_2
$$

$$
K_{\mathrm{geo},i}
=
\exp\left[
-\frac{1}{2}
\left(\frac{d_{\mathrm{geo},i}}{h}\right)^2
\right]
$$

其中 $h$ 为带宽。GeoCP 以 $K_{\mathrm{geo},i}$ 为校准残差权重并求局部加权分位数。

实现位置：

- **geoconformal-main\Codes_Manuscript_GeoSIMCP\Codes_GeoSIMCP\GeoConformal\GeoConformalSpatialRegression.py**
- 主类：**GeoConformalSpatialRegression**
- 入口/核心：**analyze**, **_kernel_smoothing**, **_weighted_quantile**
- 现代实现：**geoconformal-main\geoconformal\geocp\core.py** 中的 **GeoConformalPrediction.conformalize**
- 权重：**geoconformal-main\geoconformal\geocp\weights.py** 中的 **spatial_kernel_weights**

## 3. GeoSIMCP

### Attribute distance: EUC

$$
d_{\mathrm{attr},i}^{\mathrm{EUC}}
=
\sqrt{\sum_{j=1}^{p}(z_{ij}-z_{*j})^2}
$$

### Attribute distance: MND

$$
d_{\mathrm{attr},i}^{\mathrm{MND}}
=
\max_j
\frac{|x_{ij}-x_{*j}|}
{\operatorname{range}(X_j)}
$$

历史代码以“最小相似度转距离”的等价方式实现，并截断归一化差异的上限。

### Joint spatial and attribute distance

$$
d_{\mathrm{joint},i}
=
\sqrt{
\lambda d_{\mathrm{geo},i}^2+
(1-\lambda)d_{\mathrm{attr},i}^2
}
$$

### Weight formula

$$
w_i
=
\exp\left[
-\frac{1}{2}
\left(
\frac{d_{\mathrm{joint},i}}{h}
\right)^2
\right]
$$

实现位置：

- EUC：**geoconformal-main\Codes_Manuscript_GeoSIMCP\Codes_GeoSIMCP\GeoConformal\GeoSIMCP.py**
- 主类：**GeoSIMConformalSpatialRegression**
- MND：**geoconformal-main\Codes_Manuscript_GeoSIMCP\Codes_GeoSIMCP\GeoConformal\GeoSIMCPzhao.py**
- 主类：**GeoSIMCPzhao**
- 现代联合权重：**geoconformal-main\geoconformal\geocp\weights.py** 中的 **joint_geo_feature_weights**

## 4. GCCP / GeoCausalCP

### Historical Seattle causal distance

Seattle 历史实现先将 GCCM 分数归一化为 $w_j$：

$$
d_{\mathrm{causal},i}
=
\sqrt{
\sum_{j=1}^{p}
p\,w_j
(z_{ij}-z_{*j})^2
}
$$

乘子 $p$ 使等权情形 $w_j=1/p$ 退化为普通欧氏距离。

$$
d_{\mathrm{joint},i}^{2}
=
\lambda d_{\mathrm{geo},i}^{2}
+
(1-\lambda)d_{\mathrm{causal},i}^{2}
$$

$$
w_i^{\mathrm{GCCP}}
=
\exp\left[
-\frac{1}{2}
\frac{d_{\mathrm{joint},i}^{2}}{h^2}
\right]
$$

实现位置：

- **geoconformal-main\GCCP_Seattle\03_gccp\03_Seattle_GCCP_causal_similarity.ipynb**
- 主函数：**causal_feature_distance**, **evaluate_gccp_inner**

### CMI-weighted causal distance

Stage-19B 使用通过显著性筛选的 residual-scale CMI 特征集合 $E$：

$$
\omega_j=
\frac{\mathrm{CMI}_j}
{\sum_{k\in E}\mathrm{CMI}_k}
$$

$$
d_{\mathrm{causal},i}^{2}
=
\sum_{j\in E}
\omega_j(z_{ij}-z_{*j})^2
$$

$$
d_{\mathrm{joint},i}^{2}
=
\lambda d_{\mathrm{geo},i}^{2}
+
(1-\lambda)d_{\mathrm{causal},i}^{2}
$$

$$
w_i^{\mathrm{GeoCausalCP}}
=
\exp\left[
-\frac{d_{\mathrm{joint},i}^{2}}{2h^2}
\right]
$$

实现位置：

- **geo_causal_cp_v1\run_stage19b_direct_parity_v1.py**
- 关键函数：**fit_feature_transform**, **feature_distance_sq**, **weight_quantiles**, **tune_method**

### Historical PM2.5 GCCP-B

PM2.5 历史 notebook 使用空间核与因果核的加性混合：

$$
W_i
=
\lambda K_{\mathrm{geo},i}
+
(1-\lambda)K_{\mathrm{causal},i}
$$

这与 Stage-19B 的“先合并平方距离，再计算单一高斯核”不同。跨实验比较时应注明具体定义。

实现位置：

- **geoconformal-main\GCCP_PM2.5\04_gccp\pm25_gccp_b.ipynb**
- 关键函数：**weighted_q_matrix** 及 notebook 中的参数网格单元格

## 5. Weighted quantile

$$
q_w(\tau)
=
\inf\left\{
q:
\frac{\sum_iw_i\mathbf 1(r_i\le q)}
{\sum_iw_i}
\ge\tau
\right\}
$$

预测区间为：

$$
[\widehat f(x_*)-q_w,\widehat f(x_*)+q_w]
$$

历史 notebook 与新实验引擎对 conformal test-point $+\infty$ atom 的处理不完全一致。Stage-19B Track B 的 honest atom 版本中，部分局部方法产生无限宽区间，因此历史结果与严格版本不能视为完全同口径。

## 6. Causal-strength formulas and code

### Conditional mutual information

$$
I(X;Y\mid Z)=
\iiint p(x,y,z)
\log\frac{p(x,y\mid z)}
{p(x\mid z)p(y\mid z)}
\,dx\,dy\,dz
$$

residual-scale 实验中，$Y=|y-\widehat y|$。

实现：

- **causal_strength_benchmark_v1\stage17_cmi_worker_v1.py**，入口 **main**
- **causal_only_cp_v1\run_stage14_realdata_causalonly_cmi_exploratory_v1.py**，入口 **run_cmi**

### GCCM

GCCM 通过不同 library size 下的 cross-map skill 检查空间交叉映射收敛性，并将通过门控的最大 library size 终点相关系数作为 causal score。

实现：

- PM2.5：**geoconformal-main\GCCP_PM2.5\03_gccm\run_pm25_gccm_train_only_v1.R**
- 模拟 wrapper：**simulation_pilot_20260908\gccm_runtime\gccm_point_graph_wrapper_v1.R**

## 7. Evaluation formulas

$$
\mathrm{Coverage}
=
\frac{1}{n}
\sum_{i=1}^{n}
\mathbf 1\{L_i\le y_i\le U_i\}
$$

$$
\mathrm{Width}
=
\frac{1}{n}
\sum_{i=1}^{n}(U_i-L_i)
$$

$$
\mathrm{IS}_\alpha
=
(U-L)
+\frac{2}{\alpha}(L-y)\mathbf 1\{y<L\}
+\frac{2}{\alpha}(y-U)\mathbf 1\{y>U\}
$$

Coverage 应优先满足设计水平；Width 与 Interval Score 只能在覆盖率可比或满足约束时解释。

