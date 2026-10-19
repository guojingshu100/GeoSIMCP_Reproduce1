# 1. 本轮结果

总体状态：**PASS**。R 环境、必需包、历史核心、point-graph wrapper、200 点网格、S1–S4 scale-fit GCCM 及双独立目录复现均已完成。没有执行 Raw/Global/EUC/GCCM validation comparison，不评价区间收益。

## 2. R 环境与依赖

Rscript：`C:/Program Files/R/R-4.6.1/bin/Rscript.exe`；R 4.6.1，architecture x86_64，platform x86_64-w64-mingw32，locale C。PATH 仍未发现 Rscript，没有修改全局 PATH。R.home 返回对应安装目录的短路径。

library paths：`C:/Users/12068/AppData/Local/R/win-library/4.6` 和 `C:/Program Files/R/R-4.6.1/library`。从用户指定的 cloud.r-project.org 安装 foreach 1.5.2、doParallel 1.0.17 及必要依赖 iterators 1.0.14；codetools 0.2-20、parallel 4.6.1 原已可用。安装返回 0，包加载成功。命令和依赖版本见 install_required_packages_v1.R、package_install_v1.log。

首次 R 启动产生 C.UTF-8 locale warnings，后续仅对 R subprocess 设置 LC_ALL=C，无系统设置修改。正式 stderr 只有 iterators/parallel 包加载信息，没有 R 执行错误。完整 R.version、R.home、.libPaths、locale、sessionInfo、capabilities 见 r_environment_v1.log。

## 3. 历史来源与科学方向

实际 source 的 historical core 为 `GCCM4Lattice.r`，SHA-256 `3552d871539dc682519da04d07306a97e9138f30b22ecdbfcc2cf61efa38b12d`。generateEmbedings、laggedVar4Lattic、GCCMLattice、GCCMSingle4Lattice、simplex_projection、compute_stats 均在 smoke/正式运行中调用。规则 raster GCCM.r/basic.r 接口仅审查，没有用于不规则点数据。

方向严格为 effect_xmap_cause：effect/manifold=abs_residual，cause/reconstruction target=单个 standardized observed X_j。六字段为 s_x、s_y、x1、x2、g_proxy、x_noise。没有直接用 response、feature importance 或普通相关构造权重。

图接口能运行不等于因果识别假设得到证明。历史 graph-ring embedding 不包含 lag-0 的本点 field；跨空间独立的 x1/x2 未必可由邻域 residual manifold 重构。

## 4. Smoke tests 与真实 preflight

64 个独立临时 synthetic points 的 Delaunay graph、Ring1–Ring3、node/sample_id 对齐、64×3 finite embedding 与独立 Python BFS 比较均通过。Simplex、library loop 和两个独立输出 hashes 一致。

随后逐场景使用已有 scale_outputs 中 200 行 preparation CSV 与 frozen scaler。坐标由 standardized columns 经 inverse_transform 恢复，未打开 observed CSV。真实 preflight：S1–S4 都有 **200 个有效 embeddings**，Ring1–Ring3 均非空；R 结果与独立 BFS 一致。Restricted input 和 graph 均保存。

## 5. 正式冻结协议

E=3、tau=1、b=4、seed=42、cores=NULL（顺序）。历史 lib/pred 排除前 (E−1)tau=2 行，M=198。最小 L=10 沿用历史惯例且大于 b+1=5。预先记录的六级公式为 `floor(10*(198/10)^(i/5)+0.5)`，i=0..5；grid **[10,18,33,60,109,198]**。没有换终点、删重复或事后选网格。

随机 library draws=0。每个 L<198 枚举 198 个循环起点；末端一次。每 feature 991 raw rows，每场景 5946 raw rows。首次 estimation 之前冻结 JSON；每个正式与临时 run 保存完全相同的 execution config。

Convergence 保持历史四门槛：gain>0、Spearman>=0.60、terminal Pearson p<0.05、Fisher-z 95% lower>0。Mean curve=max(0,mean raw rho)。通过时 strength=terminal rho，否则 0；w=strength/sum，gamma=6w。全零不转等权。NA/Inf、空 ring、零方差或失败 simplex 明确停止；常数收敛曲线的 Spearman 不定义，判不收敛。

## 6. 六特征 strengths、weights 与 convergence

此处“通过”仅表示历史数值门槛通过。未通过的 strength/weight=0 是该协议规定的结果，不是程序未执行；terminal rho 另列，便于分辨。

| 场景 | Feature | Strength | Weight | Convergence | Terminal rho |
|---|---|---:|---:|---|---:|
| S1 | s_x | 0.300128 | 0.524312 | 通过 | 0.300128 |
| S1 | s_y | 0.000000 | 0.000000 | 未通过 | 0.129312 |
| S1 | x1 | 0.000000 | 0.000000 | 未通过 | 0.076422 |
| S1 | x2 | 0.000000 | 0.000000 | 未通过 | 0.000000 |
| S1 | g_proxy | 0.000000 | 0.000000 | 未通过 | 0.131259 |
| S1 | x_noise | 0.272295 | 0.475688 | 通过 | 0.272295 |
| S2 | s_x | 0.396978 | 0.477051 | 通过 | 0.396978 |
| S2 | s_y | 0.239470 | 0.287773 | 通过 | 0.239470 |
| S2 | x1 | 0.000000 | 0.000000 | 未通过 | 0.118676 |
| S2 | x2 | 0.000000 | 0.000000 | 未通过 | 0.000000 |
| S2 | g_proxy | 0.195701 | 0.235175 | 通过 | 0.195701 |
| S2 | x_noise | 0.000000 | 0.000000 | 未通过 | 0.000000 |
| S3 | s_x | 0.241882 | 0.235153 | 通过 | 0.241882 |
| S3 | s_y | 0.329094 | 0.319940 | 通过 | 0.329094 |
| S3 | x1 | 0.000000 | 0.000000 | 未通过 | 0.000000 |
| S3 | x2 | 0.000000 | 0.000000 | 未通过 | 0.046270 |
| S3 | g_proxy | 0.457638 | 0.444907 | 通过 | 0.457638 |
| S3 | x_noise | 0.000000 | 0.000000 | 未通过 | 0.028946 |
| S4 | s_x | 0.246624 | 0.454343 | 通过 | 0.246624 |
| S4 | s_y | 0.296190 | 0.545657 | 通过 | 0.296190 |
| S4 | x1 | 0.000000 | 0.000000 | 未通过 | 0.000000 |
| S4 | x2 | 0.000000 | 0.000000 | 未通过 | 0.000000 |
| S4 | g_proxy | 0.000000 | 0.000000 | 未通过 | 0.095628 |
| S4 | x_noise | 0.000000 | 0.000000 | 未通过 | 0.000000 |

完整逐库窗口原始 rho 位于 gccm_outputs/S*/raw_curves.csv；convergence.csv 保存 gain、Spearman、p、CI；weights.csv 保存六个 strengths、weights、gamma 和状态。

## 7. Mean convergence curves

| 场景 | Feature | L10 | L18 | L33 | L60 | L109 | L198 |
|---|---|---:|---:|---:|---:|---:|---:|
| S1 | s_x | 0.107197 | 0.140204 | 0.167188 | 0.248882 | 0.320316 | 0.300128 |
| S1 | s_y | 0.067698 | 0.091644 | 0.098931 | 0.112286 | 0.137413 | 0.129312 |
| S1 | x1 | 0.003213 | 0.009172 | 0.014068 | 0.010885 | 0.024341 | 0.076422 |
| S1 | x2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| S1 | g_proxy | 0.000000 | 0.001859 | 0.026381 | 0.052407 | 0.103287 | 0.131259 |
| S1 | x_noise | 0.000000 | 0.000000 | 0.016495 | 0.070997 | 0.137059 | 0.272295 |
| S2 | s_x | 0.145010 | 0.173900 | 0.231406 | 0.290719 | 0.337315 | 0.396978 |
| S2 | s_y | 0.076962 | 0.097326 | 0.120465 | 0.150974 | 0.208210 | 0.239470 |
| S2 | x1 | 0.000000 | 0.000000 | 0.000000 | 0.006316 | 0.064603 | 0.118676 |
| S2 | x2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| S2 | g_proxy | 0.037461 | 0.033359 | 0.051052 | 0.089592 | 0.121957 | 0.195701 |
| S2 | x_noise | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| S3 | s_x | 0.123703 | 0.159894 | 0.207104 | 0.233603 | 0.242753 | 0.241882 |
| S3 | s_y | 0.122786 | 0.166879 | 0.245195 | 0.281229 | 0.303676 | 0.329094 |
| S3 | x1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| S3 | x2 | 0.006533 | 0.011921 | 0.019996 | 0.014119 | 0.021592 | 0.046270 |
| S3 | g_proxy | 0.344239 | 0.390877 | 0.436330 | 0.454013 | 0.460513 | 0.457638 |
| S3 | x_noise | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.003752 | 0.028946 |
| S4 | s_x | 0.076811 | 0.104638 | 0.140039 | 0.193686 | 0.239661 | 0.246624 |
| S4 | s_y | 0.118656 | 0.152651 | 0.179719 | 0.222268 | 0.258350 | 0.296190 |
| S4 | x1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| S4 | x2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| S4 | g_proxy | 0.000000 | 0.000000 | 0.000000 | 0.011259 | 0.034932 | 0.095628 |
| S4 | x_noise | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## 8. 运行耗时、复现与测试

| 场景 | 正式运行秒 | 独立临时 A 秒 | 独立临时 B 秒 |
|---|---:|---:|---:|
| S1 | 37.45 | 37.35 | 38.48 |
| S2 | 37.56 | 36.90 | 36.30 |
| S3 | 37.01 | 36.88 | 36.38 |
| S4 | 37.37 | 36.64 | 36.03 |

正式输出及两个独立临时目录的 restricted input、edges、core、参数、execution config、embedding、raw curves、mean curves、convergence、weights SHA-256 全部一致。没有为 hash 一致而额外舍入。Runtime、临时路径等 execution metadata 不作为字节一致性目标，仍原样保存。

7 项自动测试通过：环境 smoke、冻结网格、restricted schema 与 IDs、Python 独立重算历史收敛、weights normalization/无等权 fallback、全部复现 hashes、非法 y 列及错位 node ID 的拒绝测试。Synthetic negative test 中的 y=0 仅是人工非法字段，不是读取任何数据标签。

## 9. Leakage 与文件保护

当前 runtime 阶段只读取 200 行 scale-fit preparation 和 frozen scaler，以及源码、配置/已有检查记录。没有读取或哈希混合 split 的 authorized_residuals 文件；没有读取 observed/truth CSV、tuning/validation/final/test 标签或 oracle files。所有 R 输入均由 Python 指定，R wrapper 不扫描项目数据。此限制由显式输入及代码/schema检查实现，不宣称另有操作系统级 R 文件沙箱。

Primary validation comparison=false。这些标记仅表示本阶段访问，不能改写更早 predictor 阶段已经完成的授权 validation/oracle diagnostics 历史。Historical PM2.5/Seattle、generator、predictor、scale-comparison 文件和 notebook 未修改。Runtime 原记录备份在 backup_20260909_101053。

## 10. 科学警告

S1 的 x_noise 通过，S4 的两个坐标通过；S1/S2 的 x1/x2 均未通过。必须如实报告，不能直接把历史 convergence 当成真实 residual-scale driver recovery。这些结果没有触发任何参数修改、方法筛选或权重替换。

S3 的 g_proxy 得到较大权重，但单次配对 seed 及空间共变不能证明 causal discovery 有效。历史 Pearson/Fisher 显著性计算没有自动校正重叠 spatial embeddings 的依赖。PASS 表示本任务的执行、协议遵守和复现通过，不证明 GCCM 改善区间。

## 11. 精确下一任务

在后续单独授权的阶段，将这四个 frozen weights 按特征名称接入已准备的共同 KNN scale estimator；先验证 gamma=6w、等权退化为 standardized EUC、两分支共享 preprocessing/target/floor/k grid/搜索预算。随后同时执行 Raw CP、Normalized-Global、Normalized-EUC、Normalized-GCCM 的 tuning-calibration/validation 流程，保持已预注册的 coverage tolerance 和选择规则。不得重新拟合 predictor，不因本轮权重看起来异常而临时重调 GCCM。Final-calibration/test 继续关闭。
