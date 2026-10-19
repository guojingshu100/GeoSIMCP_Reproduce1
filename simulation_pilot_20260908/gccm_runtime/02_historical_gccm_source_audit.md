# 1. 审查范围与结果

恢复后的状态：历史 GCCM4Lattice.r 中 graph/simplex/core functions 已成功 source，并在临时 synthetic smoke tests 调用。真实 scale-fit preflight 已验证 R embedding 与独立 Python BFS 一致。规则 raster interface 仅静态审查，没有对 irregular points 调用；历史 notebook 未执行。数据只来自已有 scale_outputs 的 200 行准备文件，没有读取 observed/truth CSV 或受限 labels。

## 2. Source-to-function 映射

以下路径相对于 `F:/GeoSIMCP_Reproduce`。Notebook 的 cell index 从 0 开始，行号指该 cell 拼接后的 source 行，不是 notebook JSON 行。

| 操作 | 具体来源及位置 | 实际含义 |
|---|---|---|
| 规则 raster GCCM | `geoconformal-main/GCCP_Seattle/02_gccm/GCCM_updated 2/R/GCCM/GCCM.r:77`, `GCCM` | 接收二维 xMatrix/yMatrix；`GCCMSingle`、`GCCMSingleInner` 枚举空间窗口 |
| Raster 空间 lag | 同目录 `basic.r:115`, `laggedVariableAs2Dim` | 规则栅格 lag 构造；不直接适用于本模拟 irregular points |
| Delaunay graph | `geoconformal-main/GCCP_PM2.5/03_gccm/pm25_gccm_causal.ipynb`, cell 0 source 175–212 | scipy.spatial.Delaunay，收集 triangle edges 并排序 |
| Graph adjacency | `geoconformal-main/GCCP_PM2.5/03_gccm/run_pm25_gccm_train_only_v1.R:70` | edge list 转换为对称、排序、去重的 1-based neighborMatrix |
| Graph-ring embedding | `geoconformal-main/GCCP_Seattle/02_gccm/GCCM_updated 2/R/GCCM/GCCM4Lattice.r:43,102`, `laggedVar4Lattic`, `generateEmbedings` | 对第 1..E 层邻域 field 求均值，n×E 矩阵；没有本点 lag-0 field |
| Library 与 prediction 范围 | 同文件 `GCCMLattice:128`，source 148–171 | 即使输入已经是 graph embedding，仍从 lib/pred 起点剔除 `(E-1)*tau` 行 |
| Library-size 循环 | 同文件 `GCCMLattice:176` 和 `GCCMSingle4Lattice:1` | 小于最大 L 时枚举所有循环起点；最大 L 只运行一次；不是随机抽样 |
| Simplex | 同文件 `simplex_projection:256` | 排除本点；embedding RMS distance；选 b 邻居；exp(-d/d_min)，下限 1e-6；零距离另作处理 |
| Rho | 同文件 `compute_stats:308` | Pearson corr(observed reconstruction target, reconstructed target)，同时返回 N、MAE、RMSE |
| Parallel | 同文件 source 186–198 | makeCluster/registerDoParallel/foreach `%dopar%`；cores=NULL 时走顺序循环 |
| 历史 R packages | PM2.5 runner:5–7；Seattle `outputs/gccm_author_aligned/run_seattle_logprice_gccm_v1_fix1.R:2–4` | library(parallel), library(foreach), library(doParallel) |
| 原始 rho 汇总 | PM2.5 runner 的 `mean_curve` | 同一 L 的 mean(rho, na.rm=TRUE)，然后 max(0, mean) |
| Convergence scoring | PM2.5 notebook cell 2 source 87–105、268–336 | gain>0、Spearman(L,rho)>=0.60、terminal Pearson 双侧 p<0.05、Fisher-z 95% CI lower>0 |
| Strength 和 normalization | 同 cell 2 source 333–444 | 全部条件通过才取 terminal rho，否则 strength=0；sum>0 后归一化；gamma=p*w；全部不通过时抛错，不能自动给等权 |
| Python-to-R | PM2.5 notebook cell 1 source 64–65、680–690 | 明确 Rscript 绝对路径，subprocess.run([Rscript,runner]) 捕获 stdout/stderr，无 shell 命令拼接 |

Seattle author-aligned runner 使用同一 graph core，且明确 `log_price_resid` 为 effect embedding，feature residual 为 reconstruction target。历史 wrapper 写死 E: 路径和 n=990/2088，不能直接运行或修改来适配新实验。

## 3. 因果方向冻结

本任务只允许六个 standardized observed fields：`s_x,s_y,x1,x2,g_proxy,x_noise`。Cause 是单个 `X_j`，effect 是 `abs_residual`。历史反向 cross mapping 的调用应为：

```r
effect_embeddings <- generateEmbedings(neighborMatrix, abs_residual, E)
result <- GCCMLattice(effect_embeddings, standardized_X_j,
                      lib_sizes, c(1,n), c(1,n), E,
                      tau=1, b=4, cores=NULL)
```

该方向已由新 wrapper 和 smoke tests 验证。方向名为 `effect_xmap_cause`，没有反转为 cause manifold 预测 abs_residual。

## 4. 兼容性与科学解释边界

本生成器是横截面 irregular points。历史 point-graph 接口能够接收一个位置一个标量值，算法不要求每个位置有多期记录。这是输入接口兼容性证据，不是本模拟已满足因果识别假设的证据。

依据 generator 配置，x1/x2 是跨位置独立的随机变量；历史 embedding 又不包含本点 field。邻域残差与中心点 feature 的重构关系可能很弱，S1/S2 不应仅因 mean/variance 驱动关系存在而被预期必然收敛。S3 有确定性空间函数及带噪 proxy，但空间共变也不直接证明因果发现有效。先前“概念上可适配”只能理解为数值接口层面。

## 5. 实施时必须保留或显式披露的约定

- 固定 sample_id 顺序；循环 library 起点和前两行排除都依赖顺序。
- E=3、tau=1 时末端 library 为 n-2；tau 在此图接口主要作用于索引裁剪，不能声称它重新定义了 generateEmbedings 的 ring step。
- 不静默丢 NA：未来 wrapper 应要求全部 200 个 embeddings finite，遇空 ring/failed simplex 停止，避免改变图与 split。
- R preflight 已逐场景实测 valid embedding count=200，并与独立 BFS 核验。
- historical Pearson p/Fisher CI 是项目既有评分门槛；空间依赖与重叠 libraries 下不能直接解释为已校准的因果显著性证据。
- sequential core 不调用 foreach/doParallel 并行分支；两个包已经安装并成功 library 加载，正式执行仍固定顺序 backend。
