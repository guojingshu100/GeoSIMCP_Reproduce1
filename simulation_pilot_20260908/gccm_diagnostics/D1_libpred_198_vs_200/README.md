# D1: 198 vs 200 lib/pred diagnostic

仅比较既有 S4 scale-fit 图环 embedding 的 lib/pred 成员数。主配对网格为 `[10,18,33,60,109,198]`；full-200 的 `L=200` 为补充端点，不进入主配对曲线或收敛评分。条件 A 通过原样 `GCCMLattice()` 复现冻结结果；条件 B 仅对同一历史 `GCCMSingle4Lattice()` 使用完整 1:200 lib/pred。未使用任何盲态标签或 oracle 文件。
