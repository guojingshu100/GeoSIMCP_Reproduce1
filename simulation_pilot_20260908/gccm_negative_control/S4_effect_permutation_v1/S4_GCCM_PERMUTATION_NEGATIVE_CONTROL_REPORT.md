# 1. 实验目的

只检验冻结 S4 GCCM 在 effect 与特征/位置对应关系被打乱后是否仍产生收敛；不优化 GCCP，不评价区间。

# 2. 冻结输入与设置

数据仅来自原 gccm_outputs/S4/restricted_input.csv 及既有 scale_fit 核验文件，共 200 行。effect=abs_residual；cause=z_s_x,z_s_y,z_x1,z_x2,z_g_proxy,z_x_noise；effect_xmap_cause。图 edges.csv、坐标、所有标准化数值、样本顺序保持原样。E=3、tau=1、b=4、R seed=42、cores=NULL；grid=[10,18,33,60,109,198]；200 有效 embeddings、198 terminal rows。mean rho=max(0,mean(raw rho))；gain>0、Spearman>=0.6、Pearson p<0.05、Fisher-z lower>0；通过后 strength=terminal rho，归一化为 weight。原 wrapper/core 按字节复制，没有增补 self-lag 或修改规则。

# 3. 置换设计

99 次，完整 seeds=2026090901..2026090999，运行前保存。每次 default_rng(seed).permutation(200)，只移动 effect 字段原始 CSV 数值文本。允许通常随机置换的固定点，不强制 derangement；没有重抽或挑选 seed。最多四个独立 R 任务，每任务内部顺序执行。第 1 次置换预先指定复现。原始冻结 S4 结果为 observed statistic，复现结果仅校验，不重新选择 observed。

# 4. 复现检查

总体实施状态 **PASS**。原样重跑的 raw/mean/convergence/weights/embedding 等结果与旧 S4 hashes 完全一致；99 次输入多重集、仅 effect 改变、特征/坐标/graph/settings/IDs 不变全部通过；独立 Python 从 raw 曲线重算评分通过。预指定第 1 次重复的输入和全部科学输出 hashes 一致。所有原文件前后 hashes 不变。R 版本、架构、locale、library 和 sessionInfo 保存在每个 run/sessionInfo.txt；Python 包版本及源码/配置 hashes 见 config_v1.json。

# 5. 原始 S4 GCCM 统计量

| 变量 | starting rho | terminal rho | gain | trend | historical p | Fisher lower | 通过 | strength | weight |
|---|---|---|---|---|---|---|---|---|---|
| s_x | 0.076811 | 0.246624 | 0.169813 | 1.0 | 0.000460761 | 0.110999 | True | 0.246624 | 0.454343 |
| s_y | 0.118656 | 0.296190 | 0.177534 | 1.0 | 2.26695e-05 | 0.163501 | True | 0.296190 | 0.545657 |
| x1 | 0.000000 | 0.000000 | 0.000000 | nan | 1 | -0.139441 | False | 0.000000 | 0.000000 |
| x2 | 0.000000 | 0.000000 | 0.000000 | nan | 1 | -0.139441 | False | 0.000000 | 0.000000 |
| g_proxy | 0.000000 | 0.095628 | 0.095628 | 0.94112394811432 | 0.180194 | -0.044406 | False | 0.000000 | 0.000000 |
| x_noise | 0.000000 | 0.000000 | 0.000000 | nan | 1 | -0.139441 | False | 0.000000 | 0.000000 |

# 6. 置换分布

terminal rho 的分布如下；完整 gain/strength/normalized weight 分布见 variable_distribution_summary.csv。

| 变量 | mean | p05 | median | p95 | max |
|---|---|---|---|---|---|
| s_x | 0.242870 | 0.000000 | 0.223952 | 0.571021 | 0.748812 |
| s_y | 0.239067 | 0.000000 | 0.204710 | 0.542627 | 0.727271 |
| x1 | 0.027009 | 0.000000 | 0.000000 | 0.113605 | 0.146853 |
| x2 | 0.037857 | 0.000000 | 0.007075 | 0.159786 | 0.279650 |
| g_proxy | 0.191473 | 0.000000 | 0.171372 | 0.430163 | 0.537951 |
| x_noise | 0.044097 | 0.000000 | 0.017878 | 0.166263 | 0.224588 |

每次置换的 maximum normalized weight 与通过变量个数在 permutation_summary.csv。最大权重分布：`{"n_available": 96, "mean": 0.6333734700669565, "sd": 0.2133899202389069, "min": 0.373168101076219, "p05": 0.3979588886114225, "p25": 0.46436397738325724, "median": 0.552152057477781, "p75": 0.7219385184391343, "p95": 1.0, "max": 1.0}`。全零 strength 的置换数为 3，若有则权重保持 NA，不赋零或等权；相应权重分布只对可定义结果汇总。

# 7. s_x 与 s_y 比较

上尾概率=(1+置换统计量>=原始值的次数)/(1+99)，包含 ties；它是 permutation negative-control tail probability。

| 变量 | 统计量 | 原始值 | null median | null p95 | 上尾次数/可用次数 | 尾概率 |
|---|---|---|---|---|---|---|
| s_x | terminal_rho | 0.246624 | 0.223952 | 0.571021 | 44/99 | 0.4500 |
| s_x | gain | 0.169813 | 0.124213 | 0.237114 | 32/99 | 0.3300 |
| s_x | strength | 0.246624 | 0.220243 | 0.571021 | 43/99 | 0.4400 |
| s_x | normalized_weight | 0.454343 | 0.357490 | 0.795981 | 33/96 | 0.3505 |
| s_y | terminal_rho | 0.296190 | 0.204710 | 0.542627 | 32/99 | 0.3300 |
| s_y | gain | 0.177534 | 0.131651 | 0.253965 | 26/99 | 0.2700 |
| s_y | strength | 0.296190 | 0.204710 | 0.542627 | 32/99 | 0.3300 |
| s_y | normalized_weight | 0.545657 | 0.376881 | 1.000000 | 23/96 | 0.2474 |

主要统计量是预注册的两个 terminal rho；gain、strength、weight 为补充，不能挑其中最有利者。权重如存在 NA，其尾概率为明确标记的可定义条件分布。六变量最大 terminal rho 的补充尾概率为 0.6500。

# 8. 负对照收敛频率

| 变量 | 通过次数/99 | 通过比例 |
|---|---|---|
| s_x | 64/99 | 64.65% |
| s_y | 70/99 | 70.71% |
| x1 | 2/99 | 2.02% |
| x2 | 7/99 | 7.07% |
| g_proxy | 58/99 | 58.59% |
| x_noise | 10/99 | 10.10% |

至少一个坐标通过：90/99；两个坐标都通过：44/99。通过变量个数分布：`{"0": 3, "1": 21, "2": 39, "3": 33, "4": 2, "5": 1}`。

# 9. 本实验支持什么

原始 S4 坐标强度没有同时与无对应关系的置换参考充分分离。当前 graph/sample 配置下，应暂停把 convergence/weight 解释为已识别的尺度因果信息。 归一化权重表示相对分配：只要有通过者就会归一到总和 1，大权重本身不等于强证据。

# 10. 本实验不支持什么

不能据此宣称 GCCM 一般错误，或已证明因果关系；也不能推断区间覆盖/宽度/IS。这只是固定 graph、200 点、单个 effect 边际分布下的无对应关系参考。原始 S4 的拟合残差包含 RF 预测误差，同方差真实噪声不保证实际残差尺度处处相同。置换后空间依赖和关联结构被破坏，本实验没有建立原始残差交换性或正式因果检验的条件。历史 Pearson/Fisher 数值不能当作新校准后的因果 p 值。没有读取 oracle 文件或使用隐藏 sigma 等真值作统计选择。

# 11. 下一阶段决策

原始 S4 坐标强度没有同时与无对应关系的置换参考充分分离。当前 graph/sample 配置下，应暂停把 convergence/weight 解释为已识别的尺度因果信息。 本实验结束后停止，等待研究者审阅。不自动运行 S1/S2/S3、residual-scale 比较、oracle 对照、final calibration 或 test；不修改任何门槛、图、权重规则或 predictor。本轮 final-calibration/test 标签保持未访问，未读取 tuning/validation 标签或 oracle 文件。
