# 1. 当前状态

本目录记录 `gccp_gccm_runtime_v1` 的环境停止审查，状态 `BLOCKED_ENVIRONMENT`。尚未创建可运行的 GCCM wrapper、Python runner、restricted inputs 或 S1--S4 weights。

## 2. 文件入口

- `01_r_environment_audit.md`：搜索范围、缺失解释器、人工恢复命令。
- `02_historical_gccm_source_audit.md`：精确 source/function 映射、反向重构方向与科学边界。
- `03_library_size_preregistration.md`：条件网格推导，非已验证执行协议。
- `gccm_runtime_config_v1.json`：结构化环境状态、候选协议、历史代码 hashes。
- 上级 `gccm_runtime_check_v1.json` 与 `05_gccm_runtime_and_weights_report.md`：本轮停止记录。

## 3. 恢复顺序

先找到已有 Rscript，或由用户批准安装 R；读取 sessionInfo 和包版本并运行 temporary smoke scripts。只有这些通过后才继续独立新版本配置、restricted scale-fit input、point-graph wrapper 与双次 weight estimation。现有 scale-comparison/predictor/generator 文件不回写。
