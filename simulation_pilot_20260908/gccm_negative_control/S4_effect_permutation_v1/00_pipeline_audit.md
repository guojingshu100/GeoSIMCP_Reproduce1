# 1. 冻结流水线审计

effect 来自已有 gccm_outputs/S4/restricted_input.csv 的 abs_residual，已与 S4_scale_fit_preparation.csv 的 200 个 scale_fit IDs、六字段和残差逐值核对。不读取混合 split 的数据或 residual 文件。六 cause 字段为 z_s_x、z_s_y、z_x1、z_x2、z_g_proxy、z_x_noise。

# 2. 图和算法

直接复制已有 edges.csv 和坐标，不重建 Delaunay、不重新标准化。generateEmbedings 对 Ring1–3 求 effect 均值；GCCMLattice(effect_embeddings, standardized_X_j, ...) 实现 effect_xmap_cause。E=3、tau=1、b=4；200 个有效 embeddings；历史索引裁剪后最大库 198；grid=[10,18,33,60,109,198]。小于最大库时枚举 198 个循环窗口，最大库一次；随机 library draws=0、R seed=42、cores=NULL。

# 3. 冻结评分

mean curve=max(0,mean(raw rho))；gain>0、Spearman>=0.6、Pearson p<0.05、Fisher-z 95% lower>0 全部满足才通过。通过时 strength=terminal rho，否则零；weight=strength/sum(strength)，gamma=6*weight。全零时保留空权重，不等权替代。

# 4. 运行前预注册

99 次 effect-only 置换，seed=2026090901..2026090999，default_rng(seed).permutation(200)。原始 S4 为 observed statistic；先原样复现。第 1 次置换预先指定做重复核验。最多同时 4 个独立 R 进程，每个内部顺序；完整复制历史 wrapper 和 core，字节 hash 固定。99 次结果与所有检查完成前不进行统计解释。
