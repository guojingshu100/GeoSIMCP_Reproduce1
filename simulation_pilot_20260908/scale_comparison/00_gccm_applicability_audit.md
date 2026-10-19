# GCCM 对 S1--S4 的适用性审查

## 1. 审查结论

历史 GCCM 在概念上可以通过项目既有的 point-graph adaptation 用于当前 irregular spatial samples，但本机当前不能原样、可复现地运行该分支。因此，本阶段状态为 `BLOCKED`，不得生成 learned GCCM weights，不得用 correlation、feature importance、mutual information 或 regression weights 代替 GCCM。

直接阻点有两个：当前 PATH 中找不到 `Rscript`，无法加载和执行历史 R core；当前 `scale_fit` 只有 200 个点，而历史 PM2.5/Seattle 的 library-size grids 针对约 990/2088 个点，项目尚未为 `n=200` 预注册可替代的 `lib_sizes`。在解决这两点前，Normalized-GCCM 的结果不具备可复现性和预注册依据。

## 2. 当前模拟的空间结构

`gccp_sim_s1_s4_v1` 的 `s_x,s_y` 独立服从 `Uniform(0,1)`，所以 2000 个样本及其中 200 个 `scale_fit` 点都是 irregular spatial points，不是规则 raster/lattice。每个 sample ID 对应一个空间位置和一次横截面观测，没有时间序列重复测量。四个场景共享坐标、features、split 和 local-calibration role。

## 3. 历史 GCCM 的两种接口

`GCCM.r` 的 `GCCM(xMatrix,yMatrix,...)` 直接接收二维 matrix，并按规则栅格的空间 rings 构造 embedding；当前数据不满足这个 raster 接口。

`GCCM4Lattice.r` 还提供 `generateEmbedings(neighborMatrix,x,E)` 与 `GCCMLattice(x_vectors,y,...)`。历史 PM2.5 和 Seattle wrapper 已对不规则站点/房屋点先构造 Delaunay graph，再把 adjacency list 传给 `generateEmbedings`。因此，在该项目既有实现语义下，一个位置一个标量观测足以形成 graph field；空间 ring 是图最短路径的第 1、2、3 层，不要求每个位置有时间重复观测。

## 4. Cause field 与 effect field

本研究假设为每个 observed feature `X_j` 是否影响 residual scale target `A=abs_residual`。按历史 notebook 使用的 CCM reverse-mapping 方向，检验 `X_j -> A` 时应使用：

- cause field：scale-fit 上的标准化 observed feature `X_j`；
- effect field：同一批 scale-fit 点上的 `A=abs_residual`；
- embedding/manifold：由 effect field `A` 及空间邻域 rings 构成；
- reconstruction target：cause field `X_j`；
- direction label：`effect_xmap_cause`，即 `abs_residual xmap feature`。

历史 PM2.5 对 `X -> PM2.5` 也是用 PM2.5 manifold reconstruct X；Seattle author-aligned wrapper 用 `log_price_resid` manifold reconstruct feature residual。当前 target 不应换成 response `y`、predictor feature importance 或与 `abs_residual` 的普通相关。

## 5. 已定义和未定义的 GCCM 参数

历史 point-graph workflow 明确定义了 Delaunay adjacency、`E=3`、`tau=1`、`b=4`、两个方向的命名，以及以下 convergence rule：terminal gain 大于 0；Spearman corr(`L`,`rho`) 至少 0.60；terminal rho 的双侧 Pearson p-value 小于 0.05；Fisher-z 95% CI 下界大于 0。通过时以 terminal rho 为 causal strength，否则为 0，再把正 strength 归一化为 feature weights，并用 `gamma_j=p*w_j` 形成 distance coefficients。

仍未定义的是适合 200 个 scale-fit 点的 `lib_sizes`。历史 PM2.5 grid 为 `[10,50,100,200,400,988]`，Seattle grid 的最大值为 2086；在 `E=3,tau=1,n=200` 时历史 R 函数的 terminal maximum 为 198。把历史 grid 截断、按比例缩放或另选中间点会改变 convergence evidence，不能在看到 validation 结果后决定。

## 6. 当前计算环境

当前可用 Python 为 `D:\conda\python.exe` 3.13.5。`Rscript` 不在 PATH，因而无法确认 `foreach`、`doParallel` 等 R packages 是否可用，也无法执行 `GCCM4Lattice.r` 的 smoke test。本阶段不安装或修改 R、Conda、PATH 或全局环境。

## 7. 最小科学修复与两个备选方案

最小科学修复是先让一个版本化、可记录的 R runtime 能调用历史 `GCCM4Lattice.r`，确认 `foreach/doParallel` 可用；然后在不查看 validation 指标的前提下，预注册 200 点的 `lib_sizes`，编写只接收 scale-fit Delaunay graph、`abs_residual` effect field 和六个 standardized cause fields 的 R wrapper。wrapper 必须保存原始 rho curves、terminal statistics、convergence decisions 和 weights。

可接受方案不超过两个：

1. 推荐方案：恢复/定位现有 R runtime，沿用历史 irregular-point Delaunay adaptation，并在运行前冻结 200 点 library grid。
2. 另立新版本的规则 lattice simulation，使数据直接符合 `GCCM.r` raster interface；这会成为与 `gccp_sim_s1_s4_v1` 不同的新实验，不能回写或替换当前生成器结果。

在任一方案完成前，可以准备 shared preprocessing、KNN estimator、Raw CP、Normalized-Global 和 Normalized-EUC 的代码与 scale-fit-only artifacts，但不能运行或报告 primary EUC-versus-GCCM validation comparison。
