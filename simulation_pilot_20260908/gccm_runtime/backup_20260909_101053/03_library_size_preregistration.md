# 1. 状态

`DERIVED_CONDITIONAL_NOT_EXECUTION_FROZEN`。环境停止条件已经触发；没有读取 scale-fit field，没有实际构图或 R embedding，因此完整执行协议尚未冻结。以下是由输入规模及历史约定确定的候选推导，不依赖任何方法或 validation 结果。

## 2. 最大与最小 library

固定 n=200、E=3、tau=1、b=4。未来必须先验证全部 200 行 graph-ring embeddings finite；若不满足则停止，不通过删除点或重建图凑足有效数量。

在此条件下，历史 GCCMLattice 剔除前 `(E-1)*tau=2` 行，所以可用 library/prediction 数为 `M=198`。Simplex 排除自身后仍需 b=4 个邻居，因此最小 L 至少 b+1=5。采用历史项目的 L_min=10，同时满足数值可行性。

## 3. 确定性候选网格

保留历史的六个 library 规模，按对数空间等间距覆盖 [10,198]。定义 `L_i=floor(10*(198/10)^(i/5)+0.5)`，i=0..5，并以 10 与 198 固定两端；四舍五入后若有重复则明确报错，不依据 rho 添加或删除点。

结果为 **[10,18,33,60,109,198]**。六个点、起点 10 来自历史 convention；终点 198 来自源码索引；中间点统一采用上式，因此不逐点人工选择。若另有线性/历史截断候选，事先采用上述六点 log-spacing 作为确定性 tie rule，不运行候选网格择优。

该推导允许后续独立版本采用，但当前 config 标为 conditional：只有 R runtime、restricted input、200 个 valid embeddings 及 smoke tests 成功后，才可写入真正供 execution 使用的新冻结配置。不得修改现有 scale_comparison config。

## 4. Library 重复次数、种子与 backend

历史 core 没有随机 draws，随机 draw count=0。每个 L<198 枚举全部 198 个循环 library 起点，L=198 运行 1 次；每 feature 的 raw rho 行数应为 `5*198+1=991`，每场景六 features 共 5946 行。这些是条件期望行数，不是已生成数据。

保留 `set.seed(42)` 作为显式配置，推荐 `cores=NULL` 顺序后端。循环窗口非随机抽样，不能把 seed 当作空间 randomization 或 library 独立性的证据。

## 5. 已恢复的评分与失败行为

历史规则来源见源码审查。Mean curve 使用 max(0, mean rho)。通过 gain>0、Spearman>=0.60、terminal Pearson p<0.05、Fisher-z 95% lower>0 后才取 terminal rho 为 strength；否则 strength=0。六个 strength 和为正时 w=strength/sum，gamma=6*w。若和为 0：记录 `NO_CONVERGED_FEATURES`，normalized weights 不存在，不给 uniform fallback 或声称成功。

拟定实现约束：min_valid_embeddings=200；任何 NA/Inf、空 ring、effect/cause 零方差、无法得到 finite simplex fits 时明确失败，不静默插补；non-convergent 是合法评分结果，但全零时无 learned distance。以上失败约束需在后续 wrapper 测试中验证，本轮没有执行。
