# REAL_DATA_CLEAN_BENCHMARK_RECONSTRUCTION report

日期：2026-10-12。阶段：**data foundation only**。最终方法：ΦR-SLCP。

**Seattle 和 PM2.5 的新 benchmark 基础已建立；实验继续暂停。** 本轮生成真实行级角色、隔离数据文件、来源说明、版本哈希、四方法校准入口及静态检查证据。未训练/运行模型，未计算残差、画像、预测、区间或评价指标，未比较方法结果。

clean 表示新的可追溯角色/文件/API 基础，**不表示原数据变成 never-accessed holdout**。两套数据仍有历史 test-label 暴露；来源与操作系统级隔离未完全证明，按用户要求保持暂停。

## A. 独立目录与交付内容

根目录：[real_data_clean_benchmark_20261012_v1](experiments/real_data_clean_benchmark_20261012_v1/README.md)。两个数据 benchmark 独立保存：

- [Seattle](experiments/real_data_clean_benchmark_20261012_v1/seattle/DATA_SOURCE.md)
- [PM2.5](experiments/real_data_clean_benchmark_20261012_v1/pm25/DATA_SOURCE.md)

每个数据目录含 `DATA_SOURCE.md`、`DATA_VERSION.json`、`SCHEMA.json`、`STATUS.json`、纳入审计，以及随机/空间压力各一份 `split_manifest.csv`。每个 split 保存四类角色文件；profile-learning 另保留互斥的 profile_fit(D_G) 和 profile_quantile(D_Q) 清单。features 与 sealed_labels 分目录存放，roles/manifest/features 均无目标值。

共享目录保存冻结规范快照、完整配置、依赖锁、原输入保护哈希、首次读数前源码封存、四方法入口、访问策略与 [FOUNDATION_VALIDATION.json](experiments/real_data_clean_benchmark_20261012_v1/FOUNDATION_VALIDATION.json)。历史 smoke、formal、preflight 和最终方法规范不修改。

## B. 数据版本和行级划分

| 数据 | 字节数 | 固定输入 SHA256 | 本轮行数/唯一原始 ID | 纳入前排除 |
|---|---:|---|---:|---:|
| Seattle | 251818 | `33fe6a413827718a5a209312e24a804ab6c8402ba86cd0ad66bc0c36e53d8c8d` | 3000 / 3000 | 0 |
| PM2.5 | 470736 | `eeae7b3801bf11d4a243ac40ce70f16a5ddfa07ce9c343a1864f4d54b77221c9` | 1415 / 1415 | 0 |

source hash 和既有版本一致。本轮解析 ID/日期/协变量与坐标，复核标签可用性布尔标志；数据保管程序将标签作为原始字符串路由到各 sealed_labels，不转换成数值、不计算目标统计，不用其值选样、分组或挑 seed。这个保管过程会接触标签字符串，不能宣称其从未可见。

### 已实际生成的五物理子集

| 数据/拆分 | train | D_G 画像拟合 | D_Q 局部分位数 | calibration | test | profile-learning 合计 |
|---|---:|---:|---:|---:|---:|---:|
| Seattle random | 1350 | 450 | 300 | 450 | 450 | 750 |
| Seattle spatial_stress | 1400 | 402 | 306 | 493 | 399 | 708 |
| PM2.5 random | 636 | 212 | 141 | 212 | 214 | 353 |
| PM2.5 spatial_stress | 634 | 217 | 137 | 199 | 228 | 354 |

四份精确 manifest：

- [Seattle random](experiments/real_data_clean_benchmark_20261012_v1/seattle/random/split_manifest.csv)
- [Seattle spatial stress](experiments/real_data_clean_benchmark_20261012_v1/seattle/spatial_stress/split_manifest.csv)
- [PM2.5 random](experiments/real_data_clean_benchmark_20261012_v1/pm25/random/split_manifest.csv)
- [PM2.5 spatial stress](experiments/real_data_clean_benchmark_20261012_v1/pm25/spatial_stress/split_manifest.csv)

随机 seed=2026101501；空间压力 seed=2026101502，沿用冻结规范。组 ID 事先定义为成员稳定 ID 的 SHA256；排序后用 numpy PCG64 的固定 seed 排列；按冻结比例的累计目标选择最近组边界，平局取较小累计数，test 取余。实现细则在首次读 CSV 前封存，未重新抽 seed 或根据表现调整。

Seattle 精确八属性/坐标组合形成 2995 个随机分配组件；完全相同组合不跨角色。空间压力按固定 10 km 块与该约束取连通组件，共 21 组件，不拆块追求比例。原始 ID 非 parcel ID，未知真实重复房屋不能由此排除。

PM2.5 同站点不可跨角色；2°固定块压力任务遵守同一角色边界规则。两种拆分各自独立定义实验任务，**不能合并使用**：同一行可能在 random 是 test、spatial_stress 是 train。跨任务复用及历史暴露继续限制独立验证声明，不能把两任务当作独立 replicate 或真正未暴露 holdout。

四份划分均满足文本最低数值支持；这仅表明样本数量足以保留冻结程序，不证明实际 ESS、覆盖或效率。

## C. Profile-learning 与 test 的访问边界

[access_control.py](experiments/real_data_clean_benchmark_20261012_v1/access_control.py) 的 actor 权限明确分开：

| actor | 可申请的学习角色 | test 访问 |
|---|---|---|
| train | train | features/labels 均拒绝 |
| profile_fit | 仅 D_G | features/labels 均拒绝 |
| profile_quantile | 仅 D_Q | features/labels 均拒绝 |
| query_inference | 冻结状态的逐查询 features | 允许查询 features；不允许任何 labels |
| calibration | D_cal | features/labels 均拒绝 |
| evaluation | test | 仅未来最终解封；当前 label reads 仍拒绝 |

profile-learning 不能获取原始 CSV、完整混合 DataFrame、test features 或 test labels。模型冻结后的逐查询应用由独立 query_inference actor 处理，不把 test 协变量用于画像拟合、预处理或参数学习。

当前所有 actor 的 `read_labels` 都无条件拒绝；不能用可修改的 `allow_test=true` 标志绕过。路径穿越、跨角色申请、profile actor 读取 test、两数据×四方法的实验调用均在本轮拒绝检查中通过。所有模型入口在导入校准数学模块之前停止。

**证明范围必须明确：** 这证明了提供的应用 API 的访问拒绝和文件角色边界，不证明 Windows ACL/账户/容器级隔离。原始 CSV 和 sealed_labels 仍在同一用户可读的文件系统内；数据保管程序接触原始标签字符串。完整模型训练/画像 lineage 和运行时 canary 尚未执行，因此不能将此基础检查称为端到端防泄漏证明。隔离条件未完全证明，继续暂停实验。

## D. 四方法代码入口与冻结规范

[ENTRY_CONTRACT_AUDIT.md](experiments/real_data_clean_benchmark_20261012_v1/ENTRY_CONTRACT_AUDIT.md) 给出逐方法映射；[method_entries.py](experiments/real_data_clean_benchmark_20261012_v1/method_entries.py) 提供四个纯校准数学入口：

| 方法 | 核对到的冻结程序 |
|---|---|
| CP | D_cal 原始 score，精确 .9 conformal rank |
| GeoCP | train d_ref、km 距离、h=.4、原始 kernel+self=1 infinity atom、固定 step |
| GeoSIMCP | train-only 标准化属性、RMS d_a、λ=.5/h=.4/1e−8、相同 atom/step |
| ΦR-SLCP | 固定 CDF 网格、原 localizer h/ESS 规则、D_Q q₀、独立 D_cal 校正、单活动层、允许负 Δ |

未改变历史代码：localizer/slcp 以哈希相同的字节复制为新目录文件；新适配依照已冻结公式使用 RMS 属性距离，不调用旧工厂的 calibration-fitted StandardScaler，不复用模拟 S/2、人工属性或 treatment 机制。

完整 RF 参数经 AST 从冻结协议读取后逐项比对；没有构造或拟合 RF。基/模型族/优化器以及真实特征名单封存在配置中。所有主入口没有 test-label 参数，localizer 没有 score 参数，数值模块没有文件 I/O、fit、predict 或优化调用。

**静态合同检查通过，数值/端到端方法验证未运行。** 输入数组是否确实来自 train/D_G/D_Q/D_cal 的受控 lineage，需要未来获准执行后核实；本阶段不虚构已经拟合的 RF/Gaussian 模型，也不虚构已有 profile/weights/q₀ 冻结文件。

四主方法名称和额外预算控制均沿用最终规范；不把 Residual-profile SLCP 别名作为独立第五方法，不恢复 causal semantic gain 声明。

## E. PM2.5 补充审计

详见 [PM2.5 DATA_SOURCE.md](experiments/real_data_clean_benchmark_20261012_v1/pm25/DATA_SOURCE.md)。当前可证明与不能证明的项目为：

| 项目 | 本轮结论 |
|---|---|
| 目标字段 | 精确 PM2.5 列已绑定；数值未用于选择/统计/评价 |
| 单位 | 冻结协议预期 µg/m³，但本地原始元数据证据缺失，仍 UNVERIFIED |
| 协变量 | dem、sp、tp、blh、r、aod_sat、ndvi 固定；列存在且无缺失标记；产品/单位/时间/处理链未核实 |
| 时间 | date 唯一值 20180531；目标聚合窗口不能由 date 单独证明 |
| 空间 | 1415 唯一站点编码；经纬度与固定球面/2°块适配已封存；原 datum 文档未核实 |
| 来源链 | 本地 CSV→本轮文件可追溯；上游监测产品→本地标签、外部产品→协变量尚未闭合 |

本轮查阅作者论文、作者仓库与 GeoDa 官方目录，没有下载替代数据。作者论文将 PM2.5 描述为 annual mean，和本地单日期标记文件的聚合链仍需核对；不能强行把它认定为日均浓度。[作者论文](https://www.researchgate.net/publication/403893444_Quantifying_Uncertainty_in_Spatial_Prediction_for_Nonstationary_Spatial_Processes)

外部页面没有提供可绑定当前 PM CSV 哈希的单位/采样/聚合/遥感填补链。**未找到证据不是证明不存在，常见单位或缩写含义也不是该文件的来源证据。** PM2.5 来源链阻断保持。

## F. 检查证据与最终放行状态

[FOUNDATION_VALIDATION.json](experiments/real_data_clean_benchmark_20261012_v1/FOUNDATION_VALIDATION.json) 记录 **605 项检查通过**：固定源哈希、角色全集/互斥、分组/空间块不跨角色、features 无目标列、标签 ID 对齐、API 拒绝、参数/入口静态合同及参考核心字节一致等。检查只处理数据基础和拒绝路径，不执行真实数据方法，也不生成任何方法性能。

[LINEAGE_CHECKS.json](experiments/real_data_clean_benchmark_20261012_v1/LINEAGE_CHECKS.json) 另有 **28 项保管检查通过**：四份划分可确定性重放，splitter 拒绝携带响应列的输入，各角色标签字符串与精确源行逐一相同。这里的字符串完整性核对不作数值分析，也不是模型运行。

| 原 preflight 阻断 | 本轮进展 | 当前状态 |
|---|---|---|
| 历史 test 标签暴露 | 原样记录，未包装为 fresh holdout | **仍限制验证等级** |
| split 角色无证据 | 四份行级 manifest 与角色文件已生成且交集核验通过 | **基础文件检查通过** |
| profile-learning 隔离未证明 | D_G/D_Q/cal/test 互斥及应用级拒绝检查通过 | **端到端/OS 隔离仍未证明** |
| 四方法实现未核验 | 新校准入口按冻结规范静态核对，保持运行拒绝 | **数值/完整模型流水线尚未运行** |
| PM 数据链不完整 | 明确目标、预期单位、协变量、单日期结构和具体缺口 | **仍阻断** |

本轮冻结规范 SHA256 仍为 `32e6e1fb5cd410f1ada8686617a70924806cf2c86ddaa9e880a1701dd89cddc4`；绑定协议 SHA256 仍为 `1469ce66ed5574df9ad3e923da44e3529e8533b6be7541faf5fd71e6ba73d9a4`。

两数据目录各有 `FILE_HASHES.json`，封存角色清单、feature-only 文件、sealed_labels 及来源/版本/状态文件；根目录 `ARTIFACT_HASHES.json` 汇总新交付文件和本报告哈希。`PRESERVATION_CHECK.json` 记录本轮保护的 27 个既有文件全部未变，包括历史 formal 的 22 个冻结文件、三个根规范/审计及两份原始 CSV。保护检查范围明确为这 27 项，不冒充整个工作区逐文件扫描。

最终状态：**FOUNDATION ESTABLISHED / EXPERIMENTS PAUSED**。来源或隔离条件未完全证明，不进入 Seattle/PM2.5 模型执行，不比较结果，不修改 ΦR-SLCP 冻结定义。
