# 1. GCCM scale-fit runtime

绝对路径 Rscript 4.6.1，历史 GCCM4Lattice.r，顺序 backend；唯一方向 effect_xmap_cause：abs_residual graph manifold 重构六个 standardized observed features。

## 2. 入口

`run_gccm_scale_fit_v1.py --preflight` 验证真实 embeddings 并冻结 config；不带参数执行 S1–S4，并在两个独立 temporary output directories 重复完整估计。已有目录存在时拒绝覆盖，不能直接对现有输出重跑。

R wrapper 只读取 Python 指定的 restricted input、edges、parameters 和复制的 historical core。Python 只读取 scale_outputs 的 200 行 preparation CSV 和 frozen scaler，坐标由 inverse_transform 恢复，不打开包含受限标签的 observed CSV。

测试入口：

```powershell
python -B -X utf8 simulation_pilot_20260908/gccm_runtime/test_gccm_runtime_stage.py
```

## 3. 结果与证据

环境见 01_r_environment_audit.md、package_install_v1.log、r_environment_v1.log；临时 smoke tests 见 smoke_check_v1.json；真实 preflight 在 scale_fit_preflight_v1。原始曲线、convergence、weights、sessionInfo、warnings、returncode、runtime 和 hashes 在 ../gccm_outputs/S1 至 S4。

总体状态以 ../gccm_runtime_check_v1.json 和 ../05_gccm_runtime_and_weights_report.md 为准。执行 config 冻结后保持不变；本轮没有运行 primary validation comparison。
