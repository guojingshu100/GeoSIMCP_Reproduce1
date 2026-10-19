# 1. 实验范围

仅运行 S4 `scale_fit=200` 的 effect-only permutation negative control。输出与原 `gccm_outputs/S4` 分离。未读取 generator 数据、混合 split 的 residual、其他场景、oracle 文件或 tuning/validation/final/test 标签。

# 2. 冻结与文件

- `config_v1.json`：首次 R 调用前冻结的参数、种子、统计量、尾概率规则、软件版本及输入 hashes。
- `permutation_seeds.csv`：预先指定的 99 个 seed，2026090901 至 2026090999。
- `frozen_sources/`：原 S4 restricted input、图、参数、历史 R core、原 wrapper 及原统计结果的字节副本。
- `observed_reproduction/`：未置换 S4 复现结果，不重新选择 observed statistic。
- `permutations/perm_001` 至 `perm_099`：每次置换的 restricted input、位置映射、图、参数、raw/mean curves、convergence、weights、embedding、R session、stdout/stderr、运行元数据及实施检查。
- `repeat_perm_001/`：运行前指定的第 1 次置换重复结果。
- `permutation_diagnostics.csv`：99×6 组完整收敛与权重诊断。
- `permutation_library_statistics.csv`：99×6×6 组逐 library 统计及对应收敛链。
- `permutation_summary.csv`：各次最大权重、最大 terminal rho、通过变量数量等。
- `variable_distribution_summary.csv`、`spatial_negative_control_tails.csv`、`summary.json`：分布与负对照尾概率。
- `implementation_check.json`：必需实施检查结果，PASS 不代表因果识别有效。
- `artifact_manifest.json`：完成时文件 hashes；不将运行耗时等元数据用于确定性字节比较。
- `created_files.txt`：完成时详细文件清单。
- `S4_GCCM_PERMUTATION_NEGATIVE_CONTROL_REPORT.md`：中文研究审计报告。

# 3. 复现方式

`run_negative_control.py` 使用现有 `D:/conda/python.exe` 和绝对路径 Rscript。每次 R 调用均使用完整复制的历史 wrapper/core；最多四个独立进程，每个内部仍为 `cores=NULL`。不安装依赖，不重新拟合 scaler 或 predictor。

该 runner 拒绝覆盖已经存在的 `config_v1.json`，因此不要在正式目录直接重跑整套实验。各子目录 `execution_metadata.json` 保存精确 R 调用；`permutation_mapping.csv` 保存每个节点 effect 来自哪个原节点。额外重跑应使用新的独立目录，并按既有参数与输入逐文件复现。

验收测试只读已完成结果，不启动 R 或新实验：

```powershell
& 'D:\conda\python.exe' -B -X utf8 .\test_negative_control.py
```

# 4. 科学解释

primary statistic 是 s_x、s_y 的 terminal rho，上尾概率为 `(1 + count(null >= observed))/100`。gain、strength、normalized weight 是预注册的补充诊断。完整置换允许固定点，不强制每个元素都移动。只改变 effect 对应关系，保留其边际数值多重集。

全零 strength 是有效负对照结果；normalized weights 保持不可用，不用等权替代。若存在不可用权重，权重的分布与尾概率须标明可用分母，不能偷偷赋零。

负对照尾概率不是已证明有效的因果 p 值。实验结束后停止，不自动执行任何尺度比较、oracle 对照或最终校准/测试。
