# 1. 正式冻结结果

S1–S4 真实 scale-fit R preflight：每场景 200 个 embeddings 全部 finite，Ring1–3 非空，node/sample_id 对齐，R embedding 与独立 Python BFS 一致。证据在 scale_fit_preflight_v1/S1 至 S4。没有读取 tuning、validation、final、test 或 oracle 数据。

## 2. 端点与公式

n=200、E=3、tau=1、b=4。历史 GCCMLattice 从 1+(E-1)*tau=3 起构造 library/prediction 范围，M=198。Simplex 排除自身，L 至少 b+1=5；采用历史起点 10。

六级网格保持预先记录的公式：`L_i=floor(10*(198/10)^(i/5)+0.5)`，i=0..5。六个 levels 与起点 10 来自历史 convention，终点 198 来自源码索引，中间点等对数间距。

正式 grid：**[10,18,33,60,109,198]**。没有使用任何 rho 或 validation 指标，没有替换终点。Rounding 为 floor(x+0.5)；重复处理沿用原预注册的 fail-on-duplicate，本 grid 没有重复，不删除或补点。

## 3. Library 次数与冻结顺序

随机 library draws=0；seed=42；顺序后端 cores=NULL。L<198 各枚举 198 个固定 sample_id 行序的循环起点，L=198 运行一次。每 feature raw rows=991，每场景共 5946 行。

R preflight 通过后、首次 estimation 前冻结完整 JSON，并复制到每个 run，复现比较包括 execution-config hash。没有为匹配 hashes 追加舍入。

## 4. Convergence 与失败

按 PM2.5 notebook cell 2：gain>0，Spearman(L,rho)>=0.60，terminal Pearson 双侧 p<0.05，Fisher-z 95% lower>0。每 L mean curve 为 max(0,mean(raw rho))。全部通过时 strength=terminal rho，否则 0；w=strength/sum(strength)，gamma=6*w。

min_valid_embeddings=200。空 ring、NA/Inf、zero-variance effect/cause、failed simplex、非 finite rho 均失败，不丢点或插补。常数 curve 的 Spearman 留 NA，判不收敛。全零 strength 写 NO_CONVERGED_FEATURES、weights 留空、整体 BLOCKED_PROTOCOL，不用 uniform fallback。

历史 p/CI 是该项目的评分门槛；重叠 libraries 和空间依赖下不能直接作为严格校准的因果显著性证据。
