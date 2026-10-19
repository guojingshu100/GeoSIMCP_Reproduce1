# REAL_DATA_END_TO_END_AUDIT

日期：2026-10-12。审计对象：已封存 clean benchmark 与四方法数学入口。最终判定：**BLOCKED；真实数据实验继续暂停。**

角色文件、原始数据到角色的路由和接口拒绝规则已复核，但**尚不能证明一个可工作的端到端训练/画像/校准链不会泄漏 test**。当前没有完整学习流水线及状态来源认证；标签接口全部关闭，四方法调度全部暂停。直接文件打开测试也证实，现有角色 API 不能构成操作系统隔离。

本轮未训练点预测器或画像模型，未计算预测、局部分位数、外层校正、区间或评价指标，未比较方法。新增审计材料独立存于 `experiments/real_data_end_to_end_dataflow_audit_20261012_v1`；既有规范、clean benchmark、历史源码和结果均保留。

## 1. 审计证据和边界

审计驱动：[audit_dataflow.py](experiments/real_data_end_to_end_dataflow_audit_20261012_v1/audit_dataflow.py)。完整核验记录：[AUDIT_CHECKS.json](experiments/real_data_end_to_end_dataflow_audit_20261012_v1/AUDIT_CHECKS.json)。本轮 1207 项数据/合同/拒绝检查满足各自断言，134 个既有文件在审计前后均匹配封存 SHA256。这些断言通过不等于总体实验就绪。

审计通过条件区分为：**已证明**的文件和路由属性、**仅接口层成立**的访问规则、**未证明**的实际学习状态来源及运行隔离。文件夹名 `sealed_labels` 不等于标签已经加密或受权限保护。

依据 [GCCP_FINAL_METHOD_SPEC.md](GCCP_FINAL_METHOD_SPEC.md) 与其绑定的 [REAL_DATA_PROTOCOL_DESIGN.md](REAL_DATA_PROTOCOL_DESIGN.md)，未更改 ΦR-SLCP、参数、角色、特征名单或方法定义。已有 [clean benchmark 报告](REAL_DATA_CLEAN_BENCHMARK_REPORT.md) 的暂停状态继续有效。

## 2. 原始读取到四类角色的实际数据流

```text
原始 CSV（含全部目标字符串）
  → data custodian：验证精确哈希/字段；读取 ID、属性、坐标、日期；路由不透明目标
  → target-free metadata：ID、单位组、空间块、日期、目标存在标记
  → 固定 seed 的组件拆分；不传入目标值；不重抽
  → train / D_G / D_Q / calibration / test 行级 manifest
  → 各角色 features.csv（无目标） + sealed_labels.csv（ID + target_token）
  → 当前 RoleStore：允许指定角色的特征；全部实际标签读取拒绝
  → 当前 dispatch_method：四方法均在数学模块导入前拒绝
  → predictor fit / profile fit / q0 learning / calibration / evaluation：均未执行
```

第一段是真实数据保管流；最后的学习与校准阶段是未执行的合同，不能把上述拒绝当作这些阶段已跑通。保管方读取全部目标字符串/文件字节仅用于哈希与逐行路由核对，不解析数值目标或计算目标统计。学习角色没有接收这些保管方缓冲区；同一审计进程的变量删除不构成进程隔离证明。

| 环节 | 实际复核 | 判定与限制 |
|---|---|---|
| 原始版本 | Seattle/PM2.5 精确 SHA、编码、字段；旧输入哈希前后核对 | 本地版本可追溯；上游来源缺口仍在 |
| 纳入规则 | 只依赖 ID、合法坐标、目标存在标记；当前零排除 | 未用目标大小选择 cohort |
| 分组拆分 | 以原始元数据重放固定 PCG64 seed；四份 manifest 每个单元格精确一致；额外 target canary 字段被拒绝 | 拆分接口不接收目标值；目标存在标记是例外且已声明 |
| 单位约束 | unit_key、group_id 不跨角色；压力拆分的空间块不跨角色 | Seattle 仅有属性/坐标重复代理，无独立 parcel ID，无法证明全部重复房屋已识别 |
| 特征路由 | 四任务 × 五物理角色；字段白名单、ID、全部特征值与原始转换逐项一致 | 文件级无目标列；不能证明外部协变量的上游生成未用留出目标 |
| 标签路由 | 保管方核对角色 ID 及不透明字符串与原始目标一致 | 不把保管方权限交给 learner；当前没有 OS 隔离 |
| profile-learning | D_G、D_Q 互斥，联合角色文件精确匹配两者并集 | 学习器未执行；状态来源尚未证明 |

| 数据任务 | train | D_G | D_Q | calibration | test |
|---|---:|---:|---:|---:|---:|
| Seattle random | 1350 | 450 | 300 | 450 | 450 |
| Seattle spatial_stress | 1400 | 402 | 306 | 493 | 399 |
| PM2.5 random | 636 | 212 | 141 | 212 | 214 |
| PM2.5 spatial_stress | 634 | 217 | 137 | 199 | 228 |

同一数据集的 random 与 spatial_stress 是不同任务，同一行在两任务可能分别是 test 和 train。实测交叉角色计数见 `AUDIT_CHECKS.json → cross_split_overlap`。未来执行不得混用另一任务的标签、预处理、predictor、画像或缓存；两份拆分不能称为独立重复实验。

## 3. 文件访问记录

逐事件记录：[FILE_ACCESS.jsonl](experiments/real_data_end_to_end_dataflow_audit_20261012_v1/FILE_ACCESS.jsonl)。语义请求：[ROLE_ACCESS_REQUESTS.json](experiments/real_data_end_to_end_dataflow_audit_20261012_v1/ROLE_ACCESS_REQUESTS.json)。直接打开及身份探针：[FILESYSTEM_CAPABILITY_PROBES.json](experiments/real_data_end_to_end_dataflow_audit_20261012_v1/FILESYSTEM_CAPABILITY_PROBES.json)。输入保全：[INPUT_INTEGRITY.json](experiments/real_data_end_to_end_dataflow_audit_20261012_v1/INPUT_INTEGRITY.json)。新审计文件与报告哈希见同目录 `AUDIT_ARTIFACT_HASHES.json`。

记录范围为**本次 audit worker 安装 hook 后的 Python 文件打开/目录访问，以及包装的 Python os.stat/lstat 调用**；每条带阶段、角色、数据集、split、路径、模式/flags 和放行/拒绝信息。API 请求另记实际是否返回及拒绝前是否出现 open。`PROCEED` 只表示审计 hook 放行该 OS 调用，不自动表示打开成功或读取了内容；能力探针另记真实打开结果和 0 字节内容读取。

本轮关键证据文件在记录开启后重新读取，依赖导入涉及的文件打开也纳入记录。目标文件打开只出现在保管/完整性核验或明确的零内容能力探针阶段，没有出现在角色 API 的特征阶段或固定超参数读取阶段。日志不保存任何目标值。

**不声称全操作系统访问审计**：Python 启动和 logger 之前的标准库 bootstrap、原生扩展/DLL 内部 I/O、其他进程、准备阶段的 PowerShell 只读检查与补丁写入不在 hook 范围内。Python audit hook 和包装 stat 不能替代 OS tracing，也没有覆盖尚未存在/未执行的未来模型进程。因此“所有文件访问已证明安全”这一无限范围命题未成立。

## 4. 三个关键隔离要求

| 用户要求 | 观察到的边界 | 最终结论 |
|---|---|---|
| profile learning 不访问 calibration/test 标签 | profile_fit/profile_quantile 的越权 API 请求在文件打开前拒绝；各自合法标签也因暂停门被拒绝 | **接口拒绝成立；功能性学习链未证明** |
| calibration 不访问 test | calibration actor 的 test 特征/标签请求被拒绝；独立 query_inference 可读无标签查询特征；cal 标签当前也关闭 | **接口拒绝成立；实际校准进程未证明** |
| 超参数选择不访问 test | 参数来自封存 JSON/规范；RF 完整参数 AST 对齐；本轮未做选择或评分；不存在 hyperparameter_selection actor | **本轮无选择/无 test 标签使用；未来选择流程未实现，不能认定已隔离** |

共 388 次语义 API 请求，其中 328 次按既定规则拒绝；被拒请求均没有触发文件打开。角色读取发生在 data-only 审计探针中，未运行 learner。当前“未访问 calibration/test 标签”的结论不能靠禁用全部学习来替代未来可用的权限设计。

另有两项实质边界缺口：

1. **身份由调用方自报且可修改。** 调用方可构造 `actor="evaluation"` 或修改 `store.actor`，使 test 标签路径解析通过；当前 `read_labels` 仍被无条件暂停门拒绝，故没有通过该接口读取标签。将来若只解除总门而不限制身份与能力，当前 API 规则不足以防止跨角色访问。
2. **API 外的普通文件打开可达。** 对原始 CSV、当前任务 calibration/test 标签及另一个 split 的 train 标签做 16 个 `open("rb", buffering=0)` 后立即关闭的探针，16 个打开成功，均未 read/迭代/解析，内容读取为 0 字节。这证明当前 OS 用户有绕过角色 API 的读取能力，不证明某个模型已经泄漏。

未通过后继续暂停；未解除任何原有门，也未增加可运行实验入口。

## 5. 四方法入口与最终规范

| 方法 | 入口 | 静态合同核验 |
|---|---|---|
| CP | `method_entries.CP` | 仅 calibration scores + query 数量；精确有理数 .9 秩；支持不足保留 infinity |
| GeoCP | `method_entries.GeoCP` | 冻结 train d_ref、km 地理距离、h=.4、原始 kernel + self atom=1、.9 step 与 1e−9 容差 |
| GeoSIMCP | `method_entries.GeoSIMCP` | 冻结 train 标准化属性、RMS 属性距离、λ=.5、h=.4、epsilon=1e−8、同一 self atom 与 step |
| ΦR-SLCP | `residual_profile_cdf` → `select_local_weights` → `learn_local_quantile` → `Phi_R_SLCP` | folded-normal CDF；[0,12] 的 2401 点网格；冻结 h/ESS/pmax/B_R；D_Q 分位数；独立 D_cal 外层；H=0；允许负校正；最终非负半径 |

纯数学模块无文件 I/O、模型 fit/predict 或 test 标签参数。localizer/SLCP 核心与原封存文件逐字节一致，参数合同按 AST/源码检查复核。上述数学函数**未导入执行**，未以真实数据计算任何分位数或区间。两个数据集 × 四方法的公开 dispatch 全部在导入数学模块前被暂停门拒绝。

**入口合同匹配不等于状态来源匹配。** 数值数组及 `beta/gamma`、标准化属性、`train_distance_reference`、`frozen_calibration_q0`、`frozen_query_q0` 由 caller 提供，没有携带可信的角色 lineage。现有纯函数无法证明：f/c/预处理/d_ref 只从 train 学习；β/γ 只来自 D_G；权重在 D_Q scores 前冻结；q0 未用 D_cal/test 标签。因此只能判“数学入口静态合同符合”，不能判“实际端到端实现已符合”。

未来的数据编排还需记录以下有向依赖，不修改冻结统计方法：

```text
train features + train labels → preprocessing, f, c, d_ref（冻结）
D_G features + D_G labels + frozen train state → beta/gamma/G（冻结）
D_Q features + query features + frozen G → d_R/weights/bandwidth（冻结规则）
上述冻结权重 + D_Q scores → q0 mapping（冻结）
calibration labels + frozen q0/f/c → outer correction / native baseline thresholds
test features + frozen state → all intervals（冻结并记录哈希）
独立 evaluator 最后解封 test labels → 三个评价指标
```

这里的 query 可以是 calibration/test **协变量**；不得借推理权限传入其标签。test 特征按固定规则计算自身画像和邻域，与用 test 标签训练/调参不同。补充预算对照仍按冻结协议必须报告，不能在未来把额外 profile-learning 标签预算的收益全部归于画像几何。

## 6. PM2.5 Data Card

完整卡片：[PM25_DATA_CARD.md](experiments/real_data_end_to_end_dataflow_audit_20261012_v1/PM25_DATA_CARD.md)。已整理精确文件、24 个字段及用途、目标/单位证据状态、时间/空间属性、两份角色清单和来源链。

可确认：GB18030、470736 bytes、SHA256 `eeae7b3801bf11d4a243ac40ce70f16a5ddfa07ce9c343a1864f4d54b77221c9`；1415 个站点；唯一 date 标记 `20180531`；七个固定协变量 `dem, sp, tp, blh, r, aod_sat, ndvi`；经纬度按冻结球面转换，半径 6371.0088 km，弦长距离，2° 压力块。

未证明：绑定当前 CSV 的监测提供方及精确版本链、PM2.5 的 µg/m³ 单位、实际日/月/年聚合、协变量产品/单位/取得时间与浓度独立性、原始 CRS、数据许可。date 标记不能证明日均；论文的 annual mean 描述与本地文件尚未闭合。`tp` 不按缩写猜测物理意义，软件许可不代替数据许可。**Data Card 整理完成不等于来源认证完成。**

## 7. 阻断与后续解除条件

保持现有两个 `STATUS.json` 的 `model_execution_allowed=false`、`comparison_allowed=false`。本报告不批准真实数据执行。

| 阻断 | 解除所需证据 |
|---|---|
| 无功能性端到端学习状态 lineage | 独立于 test 标签的编排及不可变状态来源记录；在授权执行前先用无真实标签的路由/canary 验证，可不训练模型 |
| 角色身份可自报、原始/标签可直接打开 | 绑定真实阶段身份的最小文件能力或进程/账户隔离；learner 不挂载原始全标签表和其他任务标签；保管/评价方权限独立；拒绝证据不能只依靠 caller 传字符串 |
| 其他 split 标签可构成间接泄漏 | 每任务隔离的数据及状态存储，禁止跨任务标签/缓存/模型复用，记录完整输入哈希 |
| 历史 test 已暴露 | 如实保持 retrospective/exploratory 解释；前瞻盲测需要另取未暴露留出数据并在评价前封存，不能靠重划分洗去历史 |
| Seattle 来源/单位身份缺口 | 精确抽样/处理版本链及独立房屋识别证据；目标转换只能用已核实字典 |
| PM2.5 来源与协变量生成缺口 | 绑定字节版本的数据字典、单位/聚合/来源链和协变量不使用留出浓度的证明 |

**结论：数据保管流可复核，数学入口静态合同符合，当前暂停门生效；可用学习流水线的无泄漏证明未完成，继续暂停 Seattle/PM2.5 实验。** 研究定位维持 *Residual distribution driven spatial conformal calibration*，不宣称 causal semantic gain。
