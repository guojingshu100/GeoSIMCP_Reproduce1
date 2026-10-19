# ΦR-SLCP 真实数据具体阻断修复报告

日期：2026-10-12。状态：**Seattle 有限探索性准备通过；PM2.5 继续暂停；当前两批数据均无独立确认性资格。**

最终方法维持 **ΦR-SLCP / Residual distribution driven spatial conformal calibration**。未改变连续 CDF 距离、局部分位数学习、独立外层校正、数据角色、固定参数或 baseline；不回到其他研究路线，不宣称 causal semantic gain。

本轮按用户授权重建了 Seattle 两个预定拆分的 train/profile 状态。**未读取真实 test 标签、未运行真实局部分位数/最终校准/区间生成/测试评分、未比较效果、未根据结果调参。** PM2.5 未读取任何角色目标用于学习。旧报告、规范和历史实验保留原状。

## 1. 固定缺口与修复范围

修复前登记为 [GAP_REGISTER_FROZEN.json](experiments/real_data_blocker_remediation_20261012_v1/GAP_REGISTER_FROZEN.json)，可读表及逐项验收为 [REAL_DATA_FIXED_GAP_TABLE.md](REAL_DATA_FIXED_GAP_TABLE.md)。三类缺口分别处理：

1. **实际流程风险**：角色自报、直接 Python 文件访问、未知学习状态复用、提前评价、跨 split 状态混用。实施统一入口、阶段能力及来源收据。
2. **数据来源/语义缺失**：Seattle 与 PM2.5 分别验收；没有用工程修复代替数据字典，也没有猜测来源/单位。
3. **证据与隔离声明不匹配**：改为可信受控 Python 流程范围内声明。未把缺少全 OS 证据自动变成新的全 OS 隔离任务或批准条件。

本轮实现独立存于 `experiments/real_data_blocker_remediation_20261012_v1`。第一次拟合前 `CODE_FREEZE.json` 封存 14 个实现/配置/方法规范文件；合成验收、收据检查和报告后续追加，不改变先前封存内容。

## 2. 所有影响区间的状态及来源

完整清单为 [STATE_CATALOG.json](experiments/real_data_blocker_remediation_20261012_v1/STATE_CATALOG.json)，包含 20 类状态/规则。主要依赖固定为：

| 状态 | 唯一允许来源 | 当前状态 |
|---|---|---|
| cohort、五物理角色、固定 feature/coordinate 映射 | 原冻结 manifest、元数据和既定转换 | 复用有来源的文件；核对 manifest 哈希、ID/支持；未读原始全标签表 |
| 属性插补/缺失指示/中心/IQR、地理中心/IQR | train 特征 | Seattle 两拆分已分别重建 |
| RF predictor、score 尺度 c、d_ref | train；c 只用 train 目标，d_ref 只用 train 坐标 | 两拆分已分别重建；不混用状态 |
| D_G out-of-fit signed residual、β/γ、拟合诊断 | 已冻结 train 状态 + D_G 特征/标签 | 两拆分各一次拟合并通过原冻结收敛门槛 |
| α、网格、模型族/基、RF/优化参数、h 候选及门槛、seed、baseline 参数 | 冻结规范及绑定配置 | 忠实复用；没有超参数搜索阶段或结果驱动选择 |
| 查询 CDF、d_R、weights/h/ESS/pmax/B_R/flags | 冻结 train/profile + D_Q/cal/test 协变量 | 已实现、合成验收；真实数据未执行 |
| D_Q scores、q₀(cal/test) | 先冻结 weights，然后 D_Q 标签 | 已实现、合成验收；真实状态尚未生成 |
| D_cal scores、秩/Δ、CP/GeoCP/GeoSIMCP 阈值 | D_cal 标签 + 已认证上游状态 | 已实现、合成验收；真实状态尚未生成 |
| 必需预算控制 | 冻结 D_G/D_Q/D_cal 预算及各自规定程序 | 已实现独立辅助状态；未反向影响主方法；仅合成运行 |
| 最终区间、query IDs、interval seal | 已认证校准/预算状态及共同预测器/尺度 | 合成链验证；未生成真实区间 |
| USD 转换 | 需核实 log 底/币种的字段字典 | 不可用，不猜转换 |
| 历史未知模型、scaler、画像、权重、q₀、校准缓存 | 无合格 lineage | 不读取、不反序列化、不复用，不按旧表现筛选 |

每个实际状态的 `RECEIPT.json` 记录：有序输入样本 ID、物理角色、manifest 哈希、完整配置哈希、代码/规范 SHA256、固定环境版本、每个输入及输出哈希、上游收据路径/哈希、数据集/split/run 身份。消费时递归核验，未知、篡改或异任务状态直接停止，不自动回退到历史缓存。

固定配置中的超参数是预先确定的规则，不是待优化变量。逐查询带宽选择只用 CDF/协变量按原 ESS/pmax/B_R 规则；候选、门槛和低信息回退均未更改。独立校准 Δ 是规定的估计状态，不是按 test coverage 调出的参数。

## 3. Seattle 实际重建记录

| 冻结任务 | train 拟合行数 | D_G 画像拟合行数 | 状态路径 |
|---|---:|---:|---|
| random | 1350 | 450 | `runs/seattle/random/learning_rebuild_v1/{train,profile}` |
| spatial_stress | 1400 | 402 | `runs/seattle/spatial_stress/learning_rebuild_v1/{train,profile}` |

路径相对本修复目录。两任务各有一个 RF 和一个 Gaussian signed-residual 画像；使用完整冻结 RF 参数、单次规定初始化和优化配置。没有重抽 seed、多起点、调参或根据测试表现重跑。优化成功及最大梯度符合原门槛仅作为数值验收，未报告预测效果。

两任务模型/预处理/画像及 lineage **分别生成**；未将另一个 split 的训练状态带入当前任务。一个任务的 train 中可能包含另一个任务的 test 行，故禁止跨任务标签和状态复用；历史暴露分类始终保留。

状态身份和输出哈希汇总：[REMEDIATION_VALIDATION.json](experiments/real_data_blocker_remediation_20261012_v1/REMEDIATION_VALIDATION.json)。详细输入 ID/角色/哈希在四份原始收据内，例如 [random train 收据](experiments/real_data_blocker_remediation_20261012_v1/runs/seattle/random/learning_rebuild_v1/train/RECEIPT.json) 和 [random profile 收据](experiments/real_data_blocker_remediation_20261012_v1/runs/seattle/random/learning_rebuild_v1/profile/RECEIPT.json)。未生成的下游状态不能声称已在真实数据验证。

## 4. 角色与访问边界

统一入口为 [controlled_entry.py](experiments/real_data_blocker_remediation_20261012_v1/controlled_entry.py)。只接受固定 dataset/split/run ID 与预定命令，不接受 caller actor、任意文件路径或历史 state 入口。Controller 依据已封存 manifest 确定样本角色和固定 task DAG，构造阶段文件能力。Worker 不接受角色 CLI 参数，内部启动标记用于排除误调用；它不是针对恶意同用户的密码学身份机制。

| 阶段 | 可直接读取的标签 | 上游状态/特征范围 |
|---|---|---|
| train | train | train 特征、冻结配置/manifest |
| profile | D_G | train 状态、D_G 特征 |
| weights | 无 | train/profile 状态；D_Q、cal/test 的 feature-only 查询 |
| local_quantile | D_Q | 已冻结 weights 与 train 状态 |
| calibration | D_cal | train、weights、q₀；主方法及固定 baseline/SLCP 控制 |
| budget_control | D_G | 已冻结 train/local/cal 状态；仅构造规定的辅助 CP 总残差池，不反馈主方法 |
| intervals | 无 | 已冻结校准/预算状态 |
| evaluation | test，最后 | 全部 interval seal 和递归来源核验通过后 |

获准标签的哈希从旧保管 manifest 获取；controller 不为“验证旧文件未改”而打开真实 raw/test 标签。实际 worker 仅对自己的获准标签文件做哈希核验和数值读取。raw 全标签文件、历史敏感文件、另一 split 标签与未授予标签均不在 worker 白名单。

采用独立 Python worker、`-I -B`、精确 source/data/state 读取名单、阶段独占输出目录、无外部网络/子进程执行。依赖路径限于所用解释器/固定环境，源码先核验哈希。最终评估独立入口为 [evaluation_entry.py](experiments/real_data_blocker_remediation_20261012_v1/evaluation_entry.py)：先校验所有区间、查询清单、配置/代码和上游状态，再生成唯一 test 标签读取能力；不返回学习阶段改方法。

**范围限制**：保证针对可信 controller 与固定 worker 的 Python 文件操作。标准库 bootstrap、原生库/系统 DLL 内部 I/O、恶意同用户另起进程不在证明范围。没有宣称文件 ACL/系统级账户隔离或全 OS 防护，也不将它们列为本项目默认阻断。输入/代码/状态哈希用于来源及完整性核验，不声称抵御能够任意改写 controller 与所有审计材料的攻击者。

## 5. 哨兵与状态边界验收

验收只使用合成文件，没有打开真实 test 标签进行探测。

- [SENTINEL_VALIDATION.json](experiments/real_data_blocker_remediation_20261012_v1/SENTINEL_VALIDATION.json)：31 项检查；三个固定配置的合成阶段链，仅用于标签替换验收，不作为 replicate 或科学效果结果。
- [RECEIPT_BOUNDARY_VALIDATION.json](experiments/real_data_blocker_remediation_20261012_v1/RECEIPT_BOUNDARY_VALIDATION.json)：4 项补充检查；即使同步修改合成收据校验记录，异 run 身份仍被拒绝；query IDs、manifest 修改也被拒绝。

验证到的具体性质：获准文件可读；直接打开原始/cal/test/历史/跨 split 哨兵被拒绝；actor CLI 被拒绝；提前评估被拒绝；改 Y_cal 不改变 train/profile/weights/q₀，允许校准改变；改 Y_test 不改变所有模型/权重/校准/最终区间；状态/区间/清单篡改拒绝。只生成评估能力计划验证门槛，**未启动 synthetic 或真实 evaluator，未计算效果指标**。

真实四个学习 worker 合并记录为 [REAL_LEARNING_FILE_ACCESS.jsonl](experiments/real_data_blocker_remediation_20261012_v1/REAL_LEARNING_FILE_ACCESS.jsonl)，共 4924 个 Python open/目录事件。目标读取只来自当前阶段的 train 或 D_G 标签。日志保留科学依赖的 32 次拒绝探测（源码字节码缓存、hostname、NUL），经逐项审查属于依赖回退，未放宽白名单；不是 test/历史文件访问。Controller 自身的源码/配置/manifest/自生成状态读取不冒充 OS 全程日志。

## 6. 分数据集语义验收

详见 [DATA_EVIDENCE_REVIEW.md](experiments/real_data_blocker_remediation_20261012_v1/DATA_EVIDENCE_REVIEW.md)。

**Seattle：可进行有限探索性实验。** 使用文件原生 log_price，不执行未证实的美元还原；来源和 parcel 身份缺口保留，按已冻结重复代理和角色开展回顾性分析。主尺度不依赖猜测 log 底或币种。这个接受范围依据冻结协议既有“主尺度直接用 log_price；无法确认字典则美元辅助不可执行”规定，并非变更方法或删困难样本。[GeoDa 官方目录](https://geodacenter.github.io/data-and-lab/) 只能支持背景数据集存在，不能补成精确 3000 行样本的映射证明。

**PM2.5：仍暂停。** 已逐项登记来源、每行含义、浓度单位、时间/聚合窗口、七个协变量的产品/版本/单位/匹配和生成过程是否使用浓度。现有文件 date 标记不能证明日均，论文年度描述不能直接移植。未查到绑定当前精确文件的语义证据，不认定已经泄漏，也不认定 feature-provenance gate 已通过；不猜 ERA5/MODIS、单位或 `tp` 含义。

下一项唯一缺失证据：**提供方/作者出具、绑定 SHA256 `eeae7b3801bf11d4a243ac40ce70f16a5ddfa07ce9c343a1864f4d54b77221c9` 的数据生成说明及字段字典**，覆盖每行观测意义、目标单位与聚合窗口、七协变量的来源/处理及是否使用站点目标标签。收到这份材料再逐字段验收，不用训练效果替代证据。

## 7. 一次性可执行结论

| 数据/目标 | 当前决策 | 唯一下一项证据或行动 |
|---|---|---|
| Seattle 原生尺度回顾性探索 | **可进行**；当前已完成 train/profile 重建，未评价 | 没有前置证据阻断；下一行动是沿统一入口的固定 DAG 准备剩余状态与冻结区间，本轮未启动 |
| Seattle 独立确认性实验 | **不可用当前历史暴露批次** | 有完整来源、身份及未暴露记录的新批次清单；不以重新 seed 代替 |
| Seattle 美元辅助任务 | **不可执行** | 绑定当前样本的 log 底/币种字段映射；它不阻塞原生尺度探索 |
| PM2.5 探索性或正式任务 | **仍暂停** | 上述绑定当前 SHA 的提供方/作者生成说明及字段字典 |
| PM2.5 独立确认性实验 | **不可用当前历史暴露批次** | 首先解决上述数据语义材料；即使通过，当前批次仍仅回顾性，不自动升级确认性 |

未来 Seattle 探索性区间准备可使用 README 中的统一入口；已有完整来源状态可经 `Controller.verify_state` 递归认证后交给下游固定阶段。公开全流程使用新 run ID，拒绝覆盖旧状态/失败目录。独立 evaluator 本轮保持未运行，历史标签已暴露的事实不受工程修复、重划分或哈希封存影响。

**本轮交付是具体风险修复、合法学习状态重建和分数据集验收；不包含任何真实测试效果结论。**
