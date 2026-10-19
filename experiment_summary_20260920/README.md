# GCCP / GeoSIMCP 实验审计总览

> 审计日期：2026-09-20  
> 项目根目录：`F:\GeoSIMCP_Reproduce`  
> 本目录仅汇总现有代码、数据和实验结果；未修改代码、未重跑实验、未改动任何原始结果文件。

## Research goal

本项目研究空间回归任务中的预测区间构建问题：在有限样本下保持边际覆盖率（Coverage）的同时，利用测试位置附近的空间、属性与因果信息缩短预测区间（Interval Width），并降低区间评分（Interval Score）。

当前核心问题是：与只利用空间和属性相似度的 GeoSIMCP 相比，引入因果相似度的 GCCP / GeoCausalCP 是否能在不牺牲有效覆盖率的前提下，得到更窄、评分更低、局部适应性更强的预测区间。

## Method relationship

| 方法 | 校准信息 | 核心思想 |
|---|---|---|
| CP | 全部校准残差等权 | 用残差的全局分位数构造预测区间 |
| GeoCP | 空间相似度 | 距离测试点越近的校准样本权重越高 |
| GeoSIMCP | 空间相似度 + 属性相似度 | 同时按地理位置和协变量相似性进行局部加权 |
| GCCP / GeoCausalCP | 空间相似度 + 属性相似度 + 因果相似度 | 用因果强度筛选或重加权属性距离，使校准邻域更贴近目标生成机制或误差尺度机制 |

方法关系可概括为：

- **GeoSIMCP**：spatial similarity + attribute similarity。
- **GCCP**：spatial similarity + attribute similarity + causal similarity。
- 当前实现并不完全统一：Seattle 使用 GCCM 权重构造因果属性距离；PM2.5 的历史 GCCP-B 使用空间核与因果核的加性混合；Stage-19B 使用 CMI 筛选后的因果距离，并与空间距离构造联合核。

## Current research hypothesis

若变量对预测目标或残差尺度具有稳定且可重复的因果关联，则在局部 conformal 校准中：

1. 应优先保留因果强度较高的特征；
2. 应在属性距离中按因果强度加权；
3. 应减少无关变量造成的“距离稀释”；
4. 有望在保持目标覆盖率的同时缩短区间并降低 Interval Score。

现有结果只对该假设提供了有限支持：Seattle 和 PM2.5 的 GCCP 结果在部分指标上略优于某些 GeoSIMCP 版本，但覆盖率低于 0.90；Stage-19B 中最小因果模块未在聚合结果上优于 GeoSIMCP-EUC。因此，目前不能声称 GCCP 已稳定优于 GeoSIMCP。

## Evaluation metrics

设测试样本预测区间为 \([L_i,U_i]\)，真实值为 \(y_i\)，样本数为 \(n\)，误覆盖水平为 \(\alpha\)。

### Coverage

$$
\mathrm{Coverage}=\frac{1}{n}\sum_{i=1}^{n}\mathbf{1}\{L_i\le y_i\le U_i\}
$$

用于衡量预测区间包含真实值的比例。若目标覆盖率为 90%，经验 Coverage 应尽量接近且不系统低于 0.90。

### Interval Width

$$
\mathrm{Width}=\frac{1}{n}\sum_{i=1}^{n}(U_i-L_i)
$$

在满足覆盖率约束的前提下，区间越窄通常越有信息量。不能脱离 Coverage 单独比较 Width。

### Interval Score

$$
\mathrm{IS}_\alpha(L,U;y)=(U-L)
+\frac{2}{\alpha}(L-y)\mathbf{1}\{y<L\}
+\frac{2}{\alpha}(y-U)\mathbf{1}\{y>U\}
$$

Interval Score 同时惩罚区间宽度和漏覆盖，数值越低越好。仍需结合 Coverage 判断，避免通过系统性欠覆盖获得表面上的低分。

## Report index

- [dataset_inventory.md](dataset_inventory.md)：Seattle、PM2.5、Simulation S1–S4 数据清单与划分。
- [causal_strength_analysis.md](causal_strength_analysis.md)：GCCM、CMI、因果权重及变量排序。
- [method_formula.md](method_formula.md)：CP、GeoCP、GeoSIMCP、GCCP 的数学公式与实现位置。
- [performance_comparison.md](performance_comparison.md)：真实数据与 Stage-19B 性能表。
- [code_map.md](code_map.md)：关键实现文件、主函数和用途导航。
- [advisor_report.md](advisor_report.md)：面向导师汇报的中文摘要。

