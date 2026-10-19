# 实验性能对比

> 评价原则：Coverage 优先；Width 和 Interval Score 只有在覆盖率满足目标或至少可比时才可直接解释。以下表格按现有结果文件原样汇总，没有重跑实验。

## Seattle

来源：**geoconformal-main\GCCP_Seattle\outputs\gccp\seattle_main_experiment_comparison_v1.csv**

| Method | Coverage | Width | Interval Score |
|---|---:|---:|---:|
| CP | 0.9089 | 35.2126 | 57.7319 |
| GeoCP | 0.9156 | 35.7454 | 55.9272 |
| GeoSIMCP-EUC | 0.9156 | 35.7454 | 55.9272 |
| GeoSIMCP-MND | 0.8933 | 32.3386 | 49.0971 |
| GCCP | 0.8889 | 32.8870 | 49.0295 |

- CP、GeoCP 和 GeoSIMCP-EUC 达到或超过 0.90 的经验覆盖率。
- GCCP 的 Interval Score 最低，比 GeoSIMCP-MND 低约 0.0676，但 Coverage 仅为 0.8889。
- 因此不能仅凭最低 Interval Score 宣称 GCCP 优于 GeoSIMCP；其小幅评分优势伴随欠覆盖。
- GeoCP 与 GeoSIMCP-EUC 的汇总数值完全相同，建议后续核查冻结超参数是否使属性部分失效或两者是否读取了同一结果。

## PM2.5

来源：**geoconformal-main\GCCP_PM2.5\outputs\pm25_formal_frozen_test_summary_v1.csv**

| Method | Coverage | Width | Interval Score |
|---|---:|---:|---:|
| CP | 0.8944 | 18.2672 | 27.2210 |
| GeoCP | 0.8732 | 17.5530 | 25.5935 |
| GeoSIMCP-EUC | 0.8662 | 16.9595 | 26.9729 |
| GeoSIMCP-MND | 0.8662 | 17.2729 | 25.6589 |
| GCCP-B | 0.8662 | 17.1210 | 25.5197 |

- 所有方法都低于 0.90，局部方法的欠覆盖更明显。
- GCCP-B 在局部方法中取得最低 Interval Score，并且宽度介于 GeoSIMCP-EUC 和 MND 之间。
- GCCP-B、GeoSIMCP-EUC 和 MND 的 Coverage 相同，可以进行局部横向比较；但它们都没有满足目标覆盖率。
- 该结果说明 GCCP-B 有局部效率信号，但不足以证明有效且稳定的总体改进。

## Simulation Stage-19B

来源：**experiments\geo_causal_cp\20260919_stage19b_direct_geosimcp_parity_v1\stage19b_summary.md**

以下为 Track A 直接 parity 汇总。该表采用已选择可行参数组合的实验口径，不是包含 test-point $+\infty$ atom 的严格 Track B 结果。

### S2

| Method | Coverage | Width | Interval Score |
|---|---:|---:|---:|
| CP (Raw-CP) | 0.8987 | 4.6242 | 6.0434 |
| GeoCP | 0.9083 | 4.8009 | 6.0670 |
| GeoSIMCP-EUC | 0.9010 | 4.6380 | 5.8894 |
| GeoSIMCP-MND | 0.8968 | 4.6430 | 6.0153 |
| GeoCausalCP | 0.9032 | 4.6903 | 5.9225 |

补充：CausalOnly-Full 为 Coverage 0.9001、Width 4.5906、Interval Score 5.8621。它属于 causal-only 对照，不计入指定的主方法比较。

### S4

| Method | Coverage | Width | Interval Score |
|---|---:|---:|---:|
| CP (Raw-CP) | 0.9010 | 4.4904 | 5.5989 |
| GeoCP | 0.9078 | 4.6142 | 5.6369 |
| GeoSIMCP-EUC | 0.9025 | 4.5680 | 5.6681 |
| GeoSIMCP-MND | 0.9014 | 4.5757 | 5.6897 |
| GeoCausalCP | 0.9078 | 4.6142 | 5.6369 |

补充：CausalOnly-Full 与 Raw-CP 相同，为 Coverage 0.9010、Width 4.4904、Interval Score 5.5989，符合 S4 同方差负对照中“不应由属性因果加权获得系统收益”的预期。

### Stage-19B interpretation

- S2 中 GeoCausalCP 的 Coverage 高于 GeoSIMCP-EUC，但 Width 和 Interval Score 均略高，未形成严格优势。
- S4 中 GeoCausalCP 与 GeoCP 汇总值一致，且没有优于 Raw-CP。
- Track A 有效选择数不完全一致：S2 中 GeoCP / EUC / MND / GeoCausalCP 分别为 35 / 40 / 37 / 40；S4 分别为 34 / 35 / 35 / 34。聚合比较可能受可行性筛选影响。
- Track B 加入 honest $+\infty$ atom 后，部分局部方法出现无限平均宽度和 Interval Score，揭示有限有效样本权重下的可行性问题。

## Does GCCP currently improve on GeoSIMCP?

结论：**目前尚不能证明 GCCP 稳定优于 GeoSIMCP。**

1. Seattle：GCCP 的 Interval Score 仅略低于 GeoSIMCP-MND，但 Coverage 更低且小于 0.90。
2. PM2.5：GCCP-B 在同等 Coverage 的局部方法中 Interval Score 最低，但所有局部方法均明显欠覆盖。
3. S2：GeoCausalCP 达到 0.90 以上覆盖率，但效率指标不如 GeoSIMCP-EUC。
4. S4：因果方法没有产生优势，符合负对照预期。
5. Seattle、PM2.5 与 Stage-19B 的因果分数、权重组合及 conformal 原子处理不同，不宜将数值直接视为同一算法版本的重复验证。

当前最稳妥的表述是：GCCP 已完成与 GeoSIMCP 的初步比较，并显示少量数据集依赖的效率信号，但尚缺少在有效覆盖约束下的稳定、可重复优势。

## Current status

### Completed experiments

- Seattle：CP、GeoCP、GeoSIMCP-EUC、GeoSIMCP-MND、GCCP 已有冻结测试结果。
- PM2.5：CP、GeoCP、GeoSIMCP-EUC、GeoSIMCP-MND、GCCP-B 已有冻结测试结果。
- Stage-19B：S2、S4 已有 direct parity 比较。
- Causal-only：模拟结果中存在 CausalOnly-Full；PM2.5 存在 residual-scale CMI 探索性流程。
- 因果权重已用于局部预测区间的距离/权重计算，但没有用于重新训练基础预测器。

### Unfinished, failed, or missing

- Stage-19B 尚未覆盖 S1、S3 的同口径 fresh direct parity。
- Stage-17B LOCI 因缺少 signed residual 而受阻。
- Stage-17C 早期版本出现 “cannot set number of interop threads after parallel work has started” 错误；恢复流程仅保留 37/48 checkpoints。
- PM2.5 residual-scale CMI 没有特征通过 BH 门槛，因果权重集合为空。
- PM2.5 本地数据日期与“2018 年年均”文字说明不一致。
- 部分旧脚本包含历史绝对路径，复现前需进行环境无关性核查。

## Recommended next steps

1. 统一 CP、GeoCP、GeoSIMCP 和 GCCP 的校准规则，明确是否加入 test-point $+\infty$ atom。
2. 以 Coverage 约束为第一筛选条件，再比较 Width 和 Interval Score。
3. 补齐 S1、S3，并对 S1–S4 使用相同种子数、可行性规则和全量失败记录。
4. 区分“目标均值因果强度”和“残差尺度因果强度”，优先评估后者对区间宽度的贡献。
5. 对真实数据只用外层训练集估计因果权重，并报告权重的 bootstrap / seed 稳定性。
6. 预先定义无显著因果变量时的回退规则，并报告回退频率。

