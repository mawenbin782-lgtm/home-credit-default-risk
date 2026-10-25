# Home Credit Default Risk 信贷违约风控项目

> 基于 Kaggle「Home Credit Default Risk」竞赛数据（30.75 万申请客户、7 张异构信贷表），
> 从 0 到 1 完成信贷违约预测的完整链路：数据审计 → 特征工程 → 建模 → 评分卡 → 业务权衡。

## 两个子项目

| 子项目 | 方法 | 验证集 AUC | 验证集 KS | 定位 |
|---|---|---|---|---|
| 违约预测（LightGBM） | 7 表融合 + 174 特征 + LightGBM | **0.777** | **0.417** | 更强、但黑盒，适合反欺诈/辅助排序 |
| 申请评分卡 | 16 个申请时点特征 + WOE/IV + 逻辑回归 | **0.733** | **0.351** | 可解释、合规，适合授信主策略 |

## 关键结果

### 1. 违约预测（LightGBM）
- 数据治理：7 张表（申请、外部征信、历史申请、POS、分期、信用卡）主键唯一性、缺失、异常值审计，客户级聚合。
- 特征工程：历史逾期、还款覆盖率、近期信贷活动、历史拒绝率、额度使用率等；**剔除 20 个泄漏特征**（依赖申请时点后的付款结果）。
- 建模：逻辑回归 + LightGBM，分层抽样 + 5 折交叉验证筛选，特征 229 → 174。
- 业务价值：风险十分位分层，最高风险组实际违约率 29.65%、最低 0.98%；历史行为特征使 AUC 从 0.759 提升至 0.777（模块消融）。

### 2. 申请评分卡（Credit Scorecard）
- 只用**申请时点可得**的 16 个特征（含外部评分 EXT_SOURCE 1/2/3，IV 最强）。
- 标准流程：等频分箱 → WOE 编码 → IV 筛选（IV ≥ 0.02）→ 逻辑回归 → 分数刻度化（PDO=20，600 分对应 good:bad = 50:1）。
- 通过率–坏账率权衡：评分前 10% 客户坏账率 **1.58%**，全量通过坏账率 **8.07%**，为授信门槛设定提供量化依据。

## 目录结构

```
data/       原始数据（gitignore，需自行下载）
features/   特征工程产物（gitignore，可重新生成）
scripts/    阶段 1~2：分表 EDA + 特征工程（8 个脚本）
pipeline/   阶段 3~8：融合审计 → 划分 → 基线 → 筛选 → 评估 → 报告（6 个脚本）
scorecard/  申请评分卡（build_scorecard.py + 指标/分箱/系数/权衡表）
outputs/    指标、图、简历要点（大文件已 gitignore）
learnings/  每阶段学习总结 PDF（讲「为什么」）
mindmaps/   总图 + 各阶段思维导图
```

## 运行方式

### 环境
Python 3.x + pandas / numpy / scikit-learn / lightgbm

### 违约预测流水线
```bash
cd pipeline
python 01_fusion_audit.py        # 多表融合与质量审计
python 02_split_preprocess.py    # 划分与预处理
python 03_baseline_models.py     # LR + LightGBM 基线
python 04_feature_selection.py   # 交叉验证 + 特征筛选
python 05_evaluation_ablation.py # 评估 + 消融
python 06_final_report.py        # 报告 + 简历
python 00_master_mindmap.py      # 生成总思维导图
```

### 评分卡
```bash
cd scorecard
python build_scorecard.py        # 输出 iv_table / 系数 / 通过率-坏账率表
```

## 数据来源

数据来自 Kaggle「Home Credit Default Risk」，需自行下载后解压到 `data/` 目录：
https://www.kaggle.com/c/home-credit-default-risk/data
