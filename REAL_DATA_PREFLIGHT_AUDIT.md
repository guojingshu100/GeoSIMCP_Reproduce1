# GCCP real-data preflight audit

日期：2026-10-12。审计版本：preflight v1。最终方法：**ΦR-SLCP**。

**总判定：BLOCKED / NOT READY。Seattle 与 PM2.5 均不放行真实数据实验。**

原因：两套数据已有历史 evaluation-label 暴露；最终真实数据适配器、行级角色清单和标签隔离证据尚不存在；数据来源/单位/特征生成链尚有缺口。统计方法与主要超参数已经文本冻结，但不能据此证明实际运行符合冻结规范。本轮在审计层停止，不训练、不计算真实数据区间或性能，不修改方法、代码或历史结果。

## 1. 审计依据、边界与判定口径

封存依据：

| 文件 | 本轮复核 SHA256 | 状态 |
|---|---|---|
| [GCCP_FINAL_METHOD_SPEC.md](GCCP_FINAL_METHOD_SPEC.md) | `32e6e1fb5cd410f1ada8686617a70924806cf2c86ddaa9e880a1701dd89cddc4` | 与交付封存值一致 |
| [REAL_DATA_PROTOCOL_DESIGN.md](REAL_DATA_PROTOCOL_DESIGN.md) | `1469ce66ed5574df9ad3e923da44e3529e8533b6be7541faf5fd71e6ba73d9a4` | 与最终规范绑定值一致 |

本轮检查了本地来源说明、历史暴露记录、两个 CSV 的文件哈希与首行字段、冻结文档、既有模拟核心源代码和依赖版本。notebook 仅检索 source 内容，不分析其执行输出；没有提取 CSV 的目标数值、读取历史逐查询测试结果、生成新的角色分配、拟合模型或运行性能评价。文件哈希会读取字节，但不把响应列解析为模型或评价输入。

CSV 文本换行数与历史样本量相容；本轮没有通过完整 CSV 数据行解析重新验证行数、唯一 ID、日期、重复记录、标签缺失或坐标范围。下文明确区分直接复核事实与历史记录，不将历史数字冒充本轮行级审计结果。

状态含义：**PASS** 表示所述有限检查有直接证据；**NOT VERIFIED** 表示缺少执行证据；**FAIL** 表示已有记录明确否定要求；**BLOCKED** 表示对应任务不能放行。静态源代码检查不是完整运行时防泄漏证明。

## 2. 六项检查结论

| 用户要求 | 文档层检查 | 执行层证据 | 最终判定 |
|---|---|---|---|
| 1. 数据来源和版本 | 精确文件、编码和预定任务已规定 | 本地字节版本可确认；上游来源链、若干单位/特征来源缺失 | **PARTIAL / BLOCKED** |
| 2. 四类角色划分 | 四角色及内部五子集、比例、seed、分组规则一致 | 新的行级角色 manifest 不存在 | **NOT VERIFIED / BLOCKED** |
| 3. profile-learning 严格独立 | train、D_G、D_Q、cal、test 的允许用途明确 | 无真实数据 ID 交集、重复单位及模型 lineage 核验 | **NOT VERIFIED / BLOCKED** |
| 4. test 标签不可见 | 最终评价前禁止进入预测器、画像、选择和校准 | 历史暴露已确认；本轮真实数据 label gate 尚未实现 | **FAIL 历史未暴露要求；BLOCKED 当前执行** |
| 5. 四方法实现公平 | 共同 predictor、cohort、尺度及补充预算对照已规定 | 最终四方法真实数据实现/调用配置不存在 | **NOT VERIFIED / BLOCKED** |
| 6. 所有超参数来自冻结规范 | 主要统计超参数可逐项追溯 | 尚无最终配置、适配源码、环境锁与入口哈希 | **文本 PASS；实际调用 NOT VERIFIED** |

这里将“test 标签不可见”解释为**最终 evaluator 解封前对所有构建、选择及校准阶段不可见**。最终评价必须读取 test 标签；要求其永远不可读将无法计算 Coverage/IS。

## 3. 数据来源和版本审计

### 3.1 本轮直接确认的本地字节版本

| 数据 | 本地文件 | 字节数 | 本轮 SHA256 | 与历史 input manifest |
|---|---|---:|---|---|
| Seattle | [seattle_sample_3k.csv](geoconformal-main/example/seattle_sample_3k.csv) | 251818 | `33fe6a413827718a5a209312e24a804ab6c8402ba86cd0ad66bc0c36e53d8c8d` | 一致 |
| PM2.5 | [china_sites_2018_05_PM2.5_all.csv](geoconformal-main/GCCP_PM2.5/00_data_audit/china_sites_2018_05_PM2.5_all.csv) | 470736 | `eeae7b3801bf11d4a243ac40ce70f16a5ddfa07ce9c343a1864f4d54b77221c9` | 一致 |

历史参照为 [stage14 input_manifest.json](experiments/real_data/20260918_stage14_realdata_feasibility_v1/input_manifest.json)。哈希一致说明文件未变，不证明抽样过程、数据授权、独立性或未曾暴露标签。

Seattle 表头按 UTF-8 解码，包含未命名原始行 ID 列、八个住房属性、UTM_X、UTM_Y、log_price。物理换行数 3001，与历史 3000 样本加表头相容。表头没有 parcel ID 或交易日期，不能由行号证明房屋身份，也不能构造时间 holdout。

PM2.5 表头按 GB18030 解码，包含站点编码/名称、城市、经纬度、date、PM2.5、dem、sp、tp、blh、r、aod_sat、ndvi 等共 24 列。物理换行数 1416，与历史 1415 样本加表头相容。历史 [DATA_PROVENANCE.md](experiments/real_data/20260918_stage14_realdata_feasibility_v1/DATA_PROVENANCE.md) 记录唯一站点数 1415、唯一日期 2018-05-31；本轮未重新读取数据日期列验证。

### 3.2 来源链未完成的项目

依据 [既有来源缺口审计](experiments/geosimcp_reproduction/20260921_geosimcp_paper_protocol_audit_v1/DATA_PROVENANCE_GAPS.md) 和 [既有项目审计](docs/experiment_notes/CAUSAL_ONLY_CP_AUDIT.md)：

- 本地 `geoconformal-main/.git` 不存在，不能报告该副本的上游 commit、checkout 日期或 clean 状态；README 也不是精确的数据版本证明。不能将软件包版本号当作 CSV 数据版本。
- Seattle 3000 行样本的精确字节可固定，但从原始交易全集生成该样本的采样 seed/代码未恢复，原始来源、获取记录和抽样链未闭合。代码中的 EPSG:32610 及 `10**log_price / 10000` 仅证明历史代码如何解释坐标/价格，不代替原始数据字典。
- Seattle 主任务仍固定文件中的 log_price；美元辅助结果要求 log 底及币种核实，未核实则辅助任务不可执行。不得为此更换主目标尺度。
- PM2.5 既有论文描述 annual mean 与本地单日快照存在冲突。当前冻结任务已经明确为单日空间估计，因此不能沿用年均值/时间预报表述；这不是要求修改当前方法。
- PM2.5 的 µg/m³ 单位仍缺可核验元数据；AOD/气象等产品的原始来源、生成时间、预测时可用性及未使用 held-out 站点浓度的证明未闭合。表头和文件哈希不能解除这些缺口。

本轮没有外网下载、替换数据或补写未经证实的来源声明。**PM2.5 单位及特征来源门槛未通过，不能进入当前冻结协议的执行。** 两套数据均不能被标为来源链已完整核实。

## 4. 四类角色与 profile-learning 独立性

角色关系与冻结规范一致：

| 四类角色 | 互斥物理子集 | 比例 | 必须满足的隔离 |
|---|---|---:|---|
| train | D_train=D_fit | 45% | 唯一可拟合 RF、插补、标准化及 score 单位 c 的集合 |
| profile-learning | D_G=D_profile；D_Q=D_local | 15%+10% | D_G 只拟合画像；画像/权重冻结后 D_Q 标签才用于 q₀ |
| calibration | D_cal | 15% | 只计算外层 Δ 或直接 baseline 阈值 |
| test | D_test | 余数约 15% | 最终全部区间冻结后，仅 evaluator 使用标签 |

四类名称不是四次独立抽样，也不允许将 D_G 与 D_Q 合并复用。五个物理子集需要按独立单位互斥，而不只是行号互斥。

主随机拆分 seed=2026101501，空间压力拆分 seed=2026101502；Seattle 重复属性/坐标组合、PM 站点编码及预定空间块按原协议约束角色。文本规则已经冻结。

本轮确认以下最终目录均不存在：

```text
experiments/gccp_v3_realdata_seattle_v1
experiments/gccp_v3_realdata_pm25_v1
```

在排除旧 outputs/inputs、环境和工具目录后，对新 seed/目标目录名称的源码与配置检索仅发现两个冻结 Markdown 文档中的说明，没有发现相应最终运行入口。因此当前没有可审计的新 cohort ID 清单、group/block 清单、角色 manifest、模型训练 ID、画像拟合 ID 或标签读取日志。历史 seed=42 的三/四分割以及历史冻结 predictions 不满足新协议，不能拿来代替新角色 lineage。

以下项目全部 **NOT VERIFIED**：

1. 五物理子集之间的所有 ID 交集为零，且重复房屋/同站点/空间块约束不跨角色。
2. RF、预处理与 c 的来源仅为 D_train；画像使用 D_G 的 out-of-fit signed residual。
3. D_Q 不参与画像拟合，D_G/D_Q 不进入主方法最终秩校正。
4. D_cal/test 标签未进入画像、权重、带宽、q₀ 或全局预处理。
5. 实际 n_G≥50、n_Q≥1、n_cal≥9；块分配下的实际样本量满足支持。

规范中的 1350/450/300/450/450 和 636/212/141/212/214 是不带分组约束的预算示意，不是本轮已生成或已核验的分配。没有 ID 证据不能将“独立 profile-learning”判定为通过。

## 5. Test 标签可见性审计

### 5.1 已确认的历史暴露

[split_exposure_audit.csv](experiments/real_data/20260918_stage14_realdata_feasibility_v1/split_exposure_audit.csv) 明确记录：

| 数据/历史集合 | 暴露分类 | 既有证据 |
|---|---|---|
| Seattle outer test / author-aligned final test | HISTORICALLY_EXPOSED | 已存在包含 y_true 的逐样本区间结果 |
| PM2.5 formal test | HISTORICALLY_EXPOSED | 已存在包含 y_true 与评价结果的 frozen-test 文件 |
| PM2.5 formal validation | HISTORICALLY_EXPOSED_DEVELOPMENT | 已有 validation predictions 与历史选择输出 |

这些结论也见 [stage14 README](experiments/real_data/20260918_stage14_realdata_feasibility_v1/README.md)。本轮读取暴露记录，不读取上述逐样本测试结果内容。既有 [来源缺口审计](experiments/geosimcp_reproduction/20260921_geosimcp_paper_protocol_audit_v1/DATA_PROVENANCE_GAPS.md) 另记录历史 b/λ selection 曾使用 test labels；这属于历史流程，不能据此断言尚未实现的新 ΦR-SLCP 已发生同样泄漏。

结论：**“数据历史上从未暴露 test 标签”要求不成立。** 冻结协议已经将复用定位为 retrospective/exploratory；新 seed、重命名角色、删除旧结果或补加 label gate 均不能撤销历史暴露。本轮没有检查新 test 与旧 test 的逐 ID 重叠，不能声称“新 test 全部曾暴露”，也不能声称“新 test 全部未暴露”。

### 5.2 当前运行时隔离尚无证据

原始 CSV 同时包含输入和响应列，物理上可被直接读取。真实数据 feature-only loader、按角色 label store、受控 evaluator 和防提前访问日志尚不存在。冻结文档中的要求不等同于已经部署访问隔离。

只读 AST 检查确认模拟 [localizer.py](experiments/gccp_v3_formal_validation_20261012_v1/localizer.py:6) 的接口仅为 query/local CDF、层标签和数值设置，导入仅 numpy/scipy，未发现文件 I/O 或标签参数；这是**该组件接口检查 PASS**。它不能证明上游 CDF 没有提前使用 calibration/test 标签，也不能替代真实数据端到端 lineage。

新流程未来至少需要实际证据证明：替换 Y_cal 不改变 G、weights/h/q₀，允许改变 Δ；替换 Y_test 不改变任何模型或区间；所有区间及中间状态哈希在 test 解封前已保存。这些检查本轮未运行，也没有将历史模拟检查冒充真实数据检查。

因此：未发现本轮新真实数据运行造成的标签泄漏，因为本轮没有启动该运行；但当前**无法证明评价前 test 标签不可见**，不能放行。

## 6. Baseline 实现公平性审计

### 6.1 设计层已经固定的公平条件

CP、GeoCP、GeoSIMCP、ΦR-SLCP 使用同一 RF、训练角色、预测目标单位、c、拆分与 test cohort。CP/GeoCP/GeoSIMCP 使用 D_cal 原始 score；ΦR-SLCP 使用独立 D_G/D_Q 学习局部阈值，再由同一 D_cal 外层校正。不得用不同 point predictor、删掉困难查询或复用历史最佳 b/λ。

主方法额外消耗 profile-learning 标签，直接 baseline 没有使用相同学习预算。冻结协议已要求 GeoCP-SLCP、GeoSIMCP-SLCP 及 CP-all-residual-budget 补充对照全部报告；不得将四方法主表的差异全部解释为 residual geometry 收益。Residual-profile SLCP 与 ΦR-SLCP 是别名，不增加独立 baseline，也不制造因果增量。

### 6.2 可复用组件的静态检查及适配阻断

以下只描述历史模拟组件，不宣称已有最终真实数据实现：

| 组件 | 静态证据 | 实数据放行所缺证据 |
|---|---|---|
| CP | [slcp.py](experiments/gccp_v3_formal_validation_20261012_v1/slcp.py:18) 用 Fraction 计算精确秩；样本不足保留 infinity | 实际输入仅为新的 D_cal 原始 R，且使用共同 c |
| GeoCP | [strategies.py](experiments/gccp_v3_formal_validation_20261012_v1/strategies.py:59) 给 native quantile 显式传 self_weights=1 | 真实距离为预定 km/d_ref，d_ref 仅来自 train，原始 kernel 尺度保留 |
| GeoSIMCP | [baseline_weights.py](experiments/gccp_v3_formal_validation_20261012_v1/baseline_weights.py:274) 可组合地理/属性 kernel | train-only 预处理、RMS 属性距离、固定 λ/h 的调用证明 |
| ΦR-SLCP | localizer 的候选 h/ESS 决策，以及 [slcp.py](experiments/gccp_v3_formal_validation_20261012_v1/slcp.py:6) 的局部分位数/负 Δ/外层精确秩与规范相符 | 单活动层、signed-residual 估计器、真实特征基、完整四角色隔离适配 |

**不能直接运行旧代码的具体原因：**

- 模拟 `strategies.kernel` 使用 S/2 和人工 attributes，不能直接代表 Seattle UTM km 或 PM 球面弦长及真实属性。
- GeoSIMCP 工厂默认 `standardize=True`，会在 calibration 属性上拟合均值/SD；冻结规范要求 train-only 中位数/IQR。这个默认行为不是已证明的 test-Y 泄漏，但若直接调用即偏离冻结预处理。
- 同一工厂的 Euclidean feature distance 为 \(\|z_A-z'_A\|_2\)，冻结规范为 \(d_a=\|z_A-z'_A\|_2/\sqrt p\)。即使关掉工厂标准化，未经适配也仍不相同。最终调用必须忠实实现已冻结 RMS 定义，本审计没有改代码或新增距离。
- [scientific.fit_distribution](experiments/gccp_v3_formal_validation_20261012_v1/scientific.py:87) 是模拟的双臂 outcome 估计器，要求两个臂均达到支持；真实数据规范是单活动层、D_G signed-residual 模型和新的固定条件基，不能伪造 treatment 后直接复用旧 runner。
- `outer_correct` 的旧接口保留两臂数组；真实数据若采用全零层标签，未被查询使用的臂 1 空组 infinity 不应计为实际 infinity。最终包装尚未核验。
- native weighted quantile 的 self-weight 与原始 K 相对尺度、step 模式及 1e−9 容差必须保留；不得切到插值、Bayesian 路径、去掉 atom 或重新缩放权重来改善结果。

这些是适配检查点，**不是已观察到某次真实数据结果不公平**。当前缺少最终真实数据包装代码、配置和实际共同 query-ID 证据，四方法实现公平性不能判为 PASS。

## 7. 超参数来源与封存一致性

本轮逐项核对最终规范及其绑定协议，未发现必须通过 test 选择的统计参数。以下数值均来自冻结文本，没有在审计中重新选择：

| 项目 | 冻结值/规则 | 来源 |
|---|---|---|
| 置信水平 | α=.10 | 最终规范 §6–7 |
| 数据比例 | 45/15/10/15/余数；四类角色内 D_G/D_Q 互斥 | 最终规范 §2；协议 §3 |
| split / RF seeds | 主 split 与 RF 2026101501；空间压力 2026101502 | 最终规范 §10；协议 §3–4 |
| 空间块 | Seattle 10 km、原点 (0,0)；PM 2°、原点 (−180°,−90°) | 协议 §3.3 |
| RF 完整参数 | 按协议 §4.1 的完整构造参数，300 trees、leaf=5、max_features=1.0、n_jobs=1；其余参数同封存原文 | 协议 §4.1 |
| score | c=max(train IQR,1e−6)；IQR 线性插值；R=abs(Z−f)/c | 最终规范 §3 |
| 画像 | Gaussian signed-residual location-scale；固定 B；OLS/log-RMS 单初始化 | 最终规范 §3.1 |
| 优化 | L-BFGS-B：maxiter=2000、maxls=50、ftol=1e−12、gtol=1e−6；梯度门槛 1e−4 | 最终规范 §3.1 |
| CDF / distance | [0,12] 2401 点；网格 Chebyshev | 最终规范 §4 |
| localizer | h=(.025,.05,.1,.2,.4)；ESS≥50、pmax≤.05、B_R≤.10；容差 1e−12；不可行取最小 B_R、首个平局 | 最终规范 §5 |
| 主方法空间项/层 | gamma_spatial=0；H≡0，不作 Mondrian 合并 | 最终规范 §5 |
| 局部分位数 | .9 stable step；不加 infinity self atom | 最终规范 §6 |
| 外层校正 | 精确 ceil[.9(n_cal+1)]；允许负 Δ；最终半径≥0，无上截断 | 最终规范 §7 |
| 最低数值支持 | n_G≥50、n_Q≥1、n_cal≥9；不改变 ESS 门槛 | 最终规范 §6 |
| GeoCP / GeoSIMCP | h=.4、λ=.5；train d_ref；RMS d_a；GeoSIMCP 1e−8 常数 | 最终规范 §8 |
| baseline quantile | 原始 kernel + self-weight=1；step .9；累计边界 1e−9 | 最终规范 §8 |
| PM 坐标 | 球半径 6371.0088 km；Euclidean chord distance | 最终规范 §8；协议 §5.2 |
| 主要评价 | Coverage、mean full Width、mean IS_.10；全部查询；联合改善规则固定 | 最终规范 §9 |

现有只读依赖环境元数据与协议一致：Python 3.11.16、numpy 1.26.4、scipy 1.15.3、scikit-learn 1.6.1。本轮仅查询版本，不实例化或拟合模型。未来执行环境锁尚未在新真实数据目录生成，不能据此认定未来入口自动使用这些版本。

另外复核历史 formal [FREEZE.json](experiments/gccp_v3_formal_validation_20261012_v1/FREEZE.json) 所列 **22 个文件哈希全部一致**；这证明参考实现未变，不表示其模拟 protocol.json 就是真实数据配置。其 arm-specific 模型、模拟 seed、SCM、尺寸等不能覆盖最终实数据规范。

**结论：主要统计超参数的文本封存检查 PASS；“所有实际运行参数均来自封存规范”尚不能核验。** 未封存的是最终执行配置、输入/角色清单、适配实现和运行环境记录，不应谎称已发现新的最佳 h/λ，也不能借审计补写或优化新超参数。seed 不代替角色 manifest；源码存在不代替实际调用证明。

## 8. 阻断项与解除所需证据

| ID | 阻断项 | 必须提供的证据；不构成本轮实施授权 |
|---|---|---|
| B1 | 历史 evaluation-label 暴露，无法满足 never-accessed test 声明 | 当前数据继续明确 retrospective/exploratory；若目标为真正未暴露测试，须另有可核验来源与暴露史的数据批次，不能靠重 seed 修复 |
| B2 | 数据来源链/数据字典不完整；PM 单位和外部特征生成链未核实 | 精确来源/版本/采样或处理记录、适用单位和预测时可用性；无法核实时按冻结协议停止对应任务 |
| B3 | 新 cohort、group/block 与五物理子集 manifest 缺失 | 不使用响应值选样的清单、所有 ID/单位交集检查及实际最低支持结果 |
| B4 | 真实数据 label gate 和阶段 lineage 缺失 | feature-only 输入、分角色标签读取日志、上游画像 lineage、cal/test canary 与解封前状态/区间哈希 |
| B5 | 四方法真实数据适配及补充预算对照未实现/核验 | 相同 predictor/c/cohort；冻结距离、原始权重尺度、atom、q₀/Δ 的独立核验；全部预定预算控制 |
| B6 | 最终执行配置/源码/环境未封存 | 每一参数到规范的映射、实现与环境锁哈希、禁止旧模拟/历史选择参数隐式进入的检查 |

解除项限于补齐来源证据及忠实实现/审计已冻结方案，不允许调参数、替换主方法、删除困难查询或根据结果选择实验版本。本轮不创建执行目录、适配代码或真实数据角色 manifest；不自动下载新数据，也不运行准备性训练来“验证可行”。

## 9. 最终决定

**Seattle：BLOCKED。PM2.5：BLOCKED。不启动真实数据实验。**

已完成的是冻结文档/本地文件版本/有限源码及环境检查；未完成的是来源门槛、实际数据角色、端到端标签隔离与四方法实现的执行审计。没有真实数据 Coverage、Width 或 IS 结果，也没有证据支持新的因果语义收益。

本轮仅新增本审计报告；方法定位继续固定为 **Residual distribution driven spatial conformal calibration**，保留 ΦR-SLCP 名称与既有全部冻结规则。
