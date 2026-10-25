# -*- coding: utf-8 -*-
"""
阶段 8：项目交付与简历总结
==========================
目标：把前面所有阶段的数字汇总成一份可写进简历、可用于面试讲解的完整报告，
     并生成可直接编辑的简历要点（markdown + docx）。
"""
import json
import os
import config
import pdf_utils

od = os.path.join(config.OUTPUTS_DIR, 'data')
mt = os.path.join(config.OUTPUTS_DIR, 'metrics')


def load(name):
    with open(os.path.join(mt, name), 'r', encoding='utf-8') as f:
        return json.load(f)


fusion = load('fusion_audit.json')
split = load('split_summary.json')
baseline = load('baseline_metrics.json')
fs = load('feature_selection.json')
ev = load('evaluation_metrics.json')
ab = load('ablation_results.json')

marginal = ab['marginal']
ablation = ab['ablation']
mod_order = sorted(config.ABLATION_MODULES, key=lambda m: -marginal[m])
top_mod = mod_order[0]

# ============================================================================
# 最终报告 PDF
# ============================================================================
d = pdf_utils.Doc('Home Credit 信贷风控项目 · 完整总结',
                  '从原始数据到可评估模型的完整链路（阶段 1–8）')
d.h1('一、项目概览')
d.body('基于 Home Credit 的信贷申请数据，构建"客户是否会违约"的信用风险模型。'
       f'共 {fusion["n_rows"]} 名申请客户，坏样本率 {fusion["target_rate"]:.2%}（不平衡分类）。'
       '数据包含 1 张申请主表 + 6 张历史行为表（外部征信、分期付款、POS/CASH、历史申请、信用卡、征信月度状态）。')

d.h1('二、方法与流程')
d.numbered([
    '业务理解与数据探索：7 张表的粒度、业务含义、单变量风险区分度。',
    '多表特征工程：从付款事件→分期、月度征信→信用记录、信用记录→客户，逐层聚合出候选特征。',
    '多表融合与质量审计：按 SK_ID_CURR 左连接，保留全部申请；完成主键/重复列/无穷大检查与 T0 防泄漏分级。',
    '数据集划分与预处理：分层 80/20 划分，LightGBM 与逻辑回归两套预处理（只在训练集拟合）。',
    '基线建模：LightGBM + 逻辑回归，拿到首轮验证 AUC。',
    '交叉验证与特征筛选：5 折 StratifiedKFold 得到 OOF AUC，剔除零贡献与高相关冗余特征。',
    '评估与消融：AUC/KS/PR-AUC + 风险分层 + 模块消融。',
])

d.h1('三、关键结果')
d.table([
    ['指标', '数值', '说明'],
    ['LightGBM 验证 AUC', f"{baseline['lgb_valid_auc']:.4f}", '基线'],
    ['逻辑回归验证 AUC', f"{baseline['lr_valid_auc']:.4f}", '可解释对照'],
    ['交叉验证 OOF AUC（删减前）', f"{fs['oof_auc_before']:.4f}", '更稳的泛化估计'],
    ['交叉验证 OOF AUC（删减后）', f"{fs['oof_auc_after']:.4f}", f"特征 {fs['n_features_before']}→{fs['n_features_after']}"],
    ['最终模型 AUC', f"{ev['auc']:.4f}", '验证集'],
    ['KS', f"{ev['ks']:.4f}", '好坏样本区分度'],
    ['PR-AUC', f"{ev['pr_auc']:.4f}", '对少数坏样本的识别'],
], col_widths=[65, 35, 55])

d.h1('四、消融结论：哪张历史表最有用')
d.body('保持划分、指标、训练流程一致，去掉单个模块看 AUC 下降（边际贡献）：')
d.table([
    ['模块', '边际贡献', '含义'],
    *[[mod, f'{marginal[mod]:+.4f}',
       '贡献最大' if mod == top_mod else ''] for mod in mod_order],
], col_widths=[45, 35, 50])
d.body(f'结论：历史模块合计把 AUC 从"仅主表"的 {ablation["app_only"]:.4f} 提升到全模型 {ablation["full"]:.4f}，'
       f'其中「{top_mod}」模块边际贡献最大（{marginal[top_mod]:+.4f}），'
       '说明多表特征工程确实带来了模型无法从主表单独得到的风险信息。')

d.h1('五、T0 防泄漏与数据质量')
d.body(f'融合后 {fusion["n_rows"]} 行、主键无重复、无重复列、无无穷大、无全空列。'
       f'T0 泄漏审计中标记 {fs["n_zero_gain"] + fs["n_corr_redundant"] + len(fs["zero_gain_features"])} 类问题特征：')
d.bullets([
    f'20 个 HIGH 风险特征（分期付款"付款结果"类，依赖 T0 之后才能观察的最终付款）——建模前排除。',
    '6 个 MEDIUM 风险特征（征信当月边界 / 极小规模更新）——保留但打标。',
    f'{fs["n_zero_gain"]} 个零贡献特征、{fs["n_corr_redundant"]} 个高相关冗余特征——交叉验证后剔除。',
])

d.h1('六、局限与改进方向')
d.bullets([
    '防泄漏是"保守排除"而非"重新计算"：理想做法是用 T0 截止日重算分期付款特征（只保留 DAYS_ENTRY_PAYMENT<=0 的付款）。',
    '逻辑回归可升级为 WOE 分箱 + 评分卡，得到可直接解释的信用评分。',
    '可加入更细的客户-月份加权聚合、额度/逾期趋势交互特征（P2 阶段）。',
    '可用 SHAP 做单客户解释，回答"为什么这个客户被拒"。',
])
d.hr()
d.image(os.path.join(config.MINDMAPS_DIR, '00_总思维导图.png'), width=165, caption='图：项目总思维导图')

report_path = d.save(os.path.join(config.LEARNINGS_DIR, '08_项目交付与简历总结.pdf'))
print('已生成最终报告:', report_path)

# ============================================================================
# 简历要点（markdown + docx）
# ============================================================================
resume_md = f"""# Home Credit 信贷违约风险建模（简历要点）

## 一句话概括
基于 30 万申请客户 × 7 张异构表的信贷数据，完成从多层级特征工程到 LightGBM/逻辑回归建模、
交叉验证特征筛选与模块消融的完整风控项目，最终模型验证集 AUC {ev['auc']:.4f}、KS {ev['ks']:.4f}。

## 三个核心能力点
1. **多层级数据重构**：把 1360 万条付款事件恢复为 1295 万条"分期"粒度，再把月度征信聚合成信用记录、
   信用记录聚合成客户，而非简单 groupby().mean()。
2. **业务驱动的特征设计**：通过分析小额尾款发现原始延期天数的缺陷；用零额度/超限记录重定义信用卡使用率；
   区分"无历史记录 / 无有效分母 / 真实为 0"三种缺失口径。
3. **用真实违约标签验证特征**：构造历史拒绝率、额度使用率、近期信贷活动与逾期趋势等特征，
   并做模块消融量化每张表的增量贡献（边际贡献最大的模块：{top_mod}）。

## 关键数字
- 数据规模：{fusion['n_rows']} 客户 × 7 表；坏样本率 {fusion['target_rate']:.2%}
- 特征工程：{fs['n_features_before']} 个候选特征，交叉验证剔除零贡献/高相关冗余后保留 {fs['n_features_after']} 个
- 最终模型：AUC {ev['auc']:.4f} / KS {ev['ks']:.4f} / PR-AUC {ev['pr_auc']:.4f}
- 消融：历史模块合计把 AUC 从 {ablation['app_only']:.4f} 提升到 {ablation['full']:.4f}

## 面试高频问题准备
- 为什么不平衡分类不能只看准确率？（用 AUC / KS / PR-AUC）
- 为什么预处理只在训练集拟合？（防止预处理阶段的数据泄漏）
- 什么是 T0 泄漏？你项目里怎么处理的？（分期付款"付款结果"依赖未来信息 → 排除）
- 交叉验证和单次划分的区别？（OOF 更稳定）
- 怎么证明每张历史表有用？（消融实验，而非单变量违约率）
"""

with open(os.path.join(config.OUTPUTS_DIR, '简历要点.md'), 'w', encoding='utf-8') as f:
    f.write(resume_md)

# 生成 docx 版本，方便直接改
import docx
doc = docx.Document()
for line in resume_md.split('\n'):
    if line.startswith('# '):
        doc.add_heading(line[2:], level=0)
    elif line.startswith('## '):
        doc.add_heading(line[3:], level=1)
    elif line.startswith('### '):
        doc.add_heading(line[4:], level=2)
    elif line.strip().startswith('- ') or line.strip().startswith('1. '):
        doc.add_paragraph(line.strip(), style='List Bullet')
    elif line.strip():
        doc.add_paragraph(line.strip())
doc.save(os.path.join(config.OUTPUTS_DIR, '简历要点.docx'))
print('已生成简历要点: 简历要点.md / 简历要点.docx')

# ============================================================================
# 项目 README（可复现说明）
# ============================================================================
readme = f"""# Home Credit Default Risk 信贷风控项目

## 运行方式（按顺序）
```
cd pipeline
python 01_fusion_audit.py        # 阶段3：多表融合与质量审计
python 02_split_preprocess.py    # 阶段4：划分与预处理
python 03_baseline_models.py     # 阶段5：LR + LightGBM 基线
python 04_feature_selection.py   # 阶段6：交叉验证 + 特征筛选
python 05_evaluation_ablation.py # 阶段7：评估 + 消融
python 06_final_report.py        # 阶段8：报告 + 简历
python 00_master_mindmap.py      # 生成总思维导图
```

## 目录
- data/          原始数据（7 张表）
- features/      特征工程产物（5 份客户级特征表）
- pipeline/      建模流水线脚本
- outputs/       宽表、模型、指标、图
- learnings/     每阶段的学习 PDF（讲"为什么"）
- mindmaps/      总图 + 各阶段分图

## 关键结果
- 最终模型：AUC {ev['auc']:.4f} / KS {ev['ks']:.4f} / PR-AUC {ev['pr_auc']:.4f}
- 特征：{fs['n_features_before']} 个候选 → 交叉验证筛选后 {fs['n_features_after']} 个
"""
with open(os.path.join(config.BASE, 'README.md'), 'w', encoding='utf-8') as f:
    f.write(readme)
print('已生成 README.md')
print('DONE 阶段8')
