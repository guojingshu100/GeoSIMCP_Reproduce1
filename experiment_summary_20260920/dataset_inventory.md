# 数据集清单

> 本文档基于本地现有数据文件和实验代码审计整理。样本量为文件实际记录数，不包含表头。

## Seattle housing

### 基本信息

| 项目 | 内容 |
|---|---|
| 文件路径 | `geoconformal-main\example\seattle_sample_3k.csv` |
| 数据格式 | CSV |
| 本地样本数 | 3,000 |
| 目标变量 | 原始字段 `log_price`；实验中转换为 `price = 10^log_price / 10000`，单位为万美元 |
| 坐标字段 | `UTM_X`, `UTM_Y` |
| 坐标参考 | UTM Zone 10N，EPSG:32610；部分流程转换为经纬度 |
| 模型输入 | 8 个房屋属性 + `UTM_X`, `UTM_Y` |
| 相似度属性 | bathrooms, sqft_living, sqft_lot, grade, condition, waterfront, view, age |

项目材料还提到原始 GeoDa Seattle 房价数据包含 21,613 笔、时间范围为 2014 年 5 月至 2015 年 5 月的交易；本仓库正式实验文件是从中形成的 3,000 条样本版本。

### Train / calibration / test split

| 层级 | 划分 | 样本数 |
|---|---|---:|
| 外层正式实验 | Train | 2,100 |
| 外层正式实验 | Calibration | 450 |
| 外层正式实验 | Test | 450 |
| Train 内层调参 | Fit | 1,470 |
| Train 内层调参 | Inner calibration | 315 |
| Train 内层调参 | Validation | 315 |

### Variables

| Variable | Meaning | Used in model |
|---|---|---|
| unnamed index column | CSV 保存时产生的行索引 | 否 |
| bathrooms | 浴室数量 | 是；预测与属性相似度 |
| sqft_living | 居住面积（平方英尺） | 是；预测与属性相似度 |
| sqft_lot | 地块面积（平方英尺） | 是；预测与属性相似度 |
| grade | 房屋建造/质量等级 | 是；预测与属性相似度 |
| condition | 房屋状况等级 | 是；预测与属性相似度 |
| waterfront | 是否临水 | 是；预测与属性相似度 |
| view | 景观/视野等级 | 是；预测与属性相似度 |
| age | 房龄 | 是；预测与属性相似度 |
| UTM_X | UTM 东向坐标 | 是；预测与空间距离 |
| UTM_Y | UTM 北向坐标 | 是；预测与空间距离 |
| log_price | 房价的对数形式；实验转换为 `10^log_price / 10000` | 是；目标来源 |

## PM2.5

### 基本信息

| 项目 | 内容 |
|---|---|
| 文件路径 | `geoconformal-main\GCCP_PM2.5\00_data_audit\china_sites_2018_05_PM2.5_all.csv` |
| 数据格式 | CSV，GBK 编码 |
| 本地样本数 | 1,415 |
| 字段数 | 24 |
| 目标变量 | `PM2.5` |
| 坐标字段 | `经度`, `纬度`；另有投影坐标 `proj_x`, `proj_y` 与栅格索引 `row_index`, `col_index` |
| 正式模型属性 | dem, sp, tp, blh, r, aod_sat, ndvi |
| 正式预测输入 | 上述 7 个属性 + 经纬度 |
| 时间字段 | 本地文件 `date` 均为 `20180531` |

注意：项目文字材料曾将该数据描述为 2018 年年均数据，但本地正式 CSV 的日期值全部为 2018-05-31。后续论文写作前应统一数据口径。

### Train / calibration / validation / test split

| 划分 | 样本数 |
|---|---:|
| Train | 990 |
| Calibration | 141 |
| Validation | 142 |
| Test | 142 |

### Variables

| Variable | Meaning | Used in model |
|---|---|---|
| 监测点编码 | 空气质量监测站唯一编码 | 否；仅标识 |
| 监测点名称 | 监测站名称 | 否；仅标识 |
| 城市 | 监测站所在城市 | 否；描述信息 |
| 经度 | 站点经度 | 是；预测与空间距离 |
| 纬度 | 站点纬度 | 是；预测与空间距离 |
| date | 观测日期 | 否；当前文件为单日截面 |
| PM2.5 | 细颗粒物浓度 | 是；预测目标 |
| row_index | 栅格行索引 | 否；辅助定位 |
| col_index | 栅格列索引 | 否；辅助定位 |
| proj_x | 投影坐标 X | 否；现有正式流程主要用经纬度 |
| proj_y | 投影坐标 Y | 否；现有正式流程主要用经纬度 |
| dem | 数字高程/海拔 | 是 |
| w10 | 10 m 风速（按字段命名推断） | 否 |
| d10 | 10 m 风向（按字段命名推断） | 否 |
| t2m | 2 m 气温 | 否 |
| sp | 地表气压 | 是 |
| tp | 项目材料称“温度”；ERA5 标准字段通常表示总降水，需核实 | 是 |
| blh | 边界层高度 | 是 |
| e | 变量含义未在本地审计材料中明确记录 | 否 |
| r | 相对湿度 | 是 |
| u10 | 10 m 东西向风分量 | 否 |
| v10 | 10 m 南北向风分量 | 否 |
| aod_sat | 卫星气溶胶光学厚度 AOD | 是 |
| ndvi | 归一化植被指数 | 是 |

## Simulation S1–S4

### 文件和用途

模拟数据位于：

- `simulation_pilot_20260908\data\S1_observed.csv` 与 `S1_truth.csv`
- `simulation_pilot_20260908\data\S2_observed.csv` 与 `S2_truth.csv`
- `simulation_pilot_20260908\data\S3_observed.csv` 与 `S3_truth.csv`
- `simulation_pilot_20260908\data\S4_observed.csv` 与 `S4_truth.csv`

每个情景包含 2,000 个样本。S1–S4 用于检验因果/属性加权在不同均值结构、异方差结构和空间潜变量结构下是否真正改善局部 conformal 预测区间，而不是只在单一真实数据集上产生偶然收益。

### Observed variables

| Variable | Meaning | Availability |
|---|---|---|
| sample_id | 样本编号 | 可观测 |
| scenario | 情景编号 S1–S4 | 可观测 |
| split | train / local calibration / validation / final calibration / test | 可观测 |
| local_calibration_role | local calibration 内的 scale_fit 或 tuning_cal 角色 | 可观测 |
| s_x, s_y | 二维空间坐标 | 可观测 |
| x1, x2 | 模拟属性变量 | 可观测 |
| g_proxy | 空间潜变量 `g_true` 的含噪代理 | 可观测 |
| x_noise | 与目标机制无关的噪声属性 | 可观测 |
| y | 响应变量 | 可观测；预测目标 |

### Truth variables

| Variable | Meaning |
|---|---|
| mu_true | 条件均值真值 |
| sigma_true | 条件尺度真值 |
| epsilon | 生成响应的随机误差 |
| g_true | 空间潜变量真值 |
| eta | 构造 `g_proxy` 的测量噪声 |

公共生成式为：

$$
g_{true}=\sin(2\pi s_x)+\cos(2\pi s_y),\qquad
g_{proxy}=g_{true}+0.35\eta
$$

$$
h(z)=0.6+\frac{1.4}{1+\exp(-1.2z)},\qquad
y=\mu_{true}+\sigma_{true}\epsilon
$$

### True causal structure

| Scenario | Purpose | Mean structure | Scale structure | Key causal interpretation |
|---|---|---|---|---|
| S1 | 均值与尺度由同一属性驱动 | `mu_true = 2*x1` | `sigma_true = h(x1)` | x1 同时影响均值与误差尺度 |
| S2 | 均值与尺度驱动变量分离 | `mu_true = 2*x1` | `sigma_true = h(x2)` | x1 决定均值，x2 决定预测区间所需尺度 |
| S3 | 检验空间潜变量及其代理 | `mu_true = 2*x1 + 1.5*g_true` | `sigma_true = h(g_true)` | 空间潜变量同时影响均值和尺度，g_proxy 提供含噪观测 |
| S4 | 同方差负对照 | `mu_true = 2*x1` | `sigma_true = 1.3` | 没有属性驱动的尺度异质性，局部因果加权不应带来系统收益 |

### Split information

| Split | Samples | Purpose |
|---|---:|---|
| Train | 1,000 | 拟合基础预测模型 |
| Local calibration | 300 | 局部尺度/因果模块开发 |
| └ scale_fit | 200 | 尺度模型或相关模块拟合 |
| └ tuning_cal | 100 | 局部参数调节 |
| Validation | 200 | 超参数和方法选择 |
| Final calibration | 200 | 最终 conformal 校准 |
| Test | 300 | 冻结后的最终评估 |

