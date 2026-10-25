# -*- coding: utf-8 -*-
"""
阶段 7：模型评估与消融实验
==========================
目标：
  1. 用最终模型计算核心指标：AUC、KS、PR-AUC，并画 ROC/KS 曲线、风险十分位分层图；
  2. 做消融实验，量化"七张表里到底哪些历史模块帮了最多"。

为什么消融实验比"看单变量违约率"更有说服力？
  单变量分析只能说明"这个特征单独看和违约相关"，但它放进模型后可能被别的特征取代、没有增量。
  消融实验保持数据划分、评价指标、训练流程完全一致，只去掉某个模块，观察 AUC 变化，
  才能量化"这一模块在模型里的真实增量贡献"。
"""
import json
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib

import config
import pdf_utils
import mindmap_utils
import plot_utils

od = os.path.join(config.OUTPUTS_DIR, 'data')
md = os.path.join(config.OUTPUTS_DIR, 'models')
fg = os.path.join(config.OUTPUTS_DIR, 'figures')

PARAMS = dict(
    objective='binary', metric='auc', learning_rate=0.05, num_leaves=31,
    min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, n_estimators=1500,
    random_state=config.SEED, n_jobs=-1, verbose=-1,
)

# ----------------------------------------------------------------------------
# 1. 读取数据与最终模型
# ----------------------------------------------------------------------------
X_train = pd.read_parquet(os.path.join(od, 'train_lgb.parquet'))
X_valid = pd.read_parquet(os.path.join(od, 'valid_lgb.parquet'))
y_train = np.load(os.path.join(od, 'y_train.npy'))
y_valid = np.load(os.path.join(od, 'y_valid.npy'))
selected = pd.read_csv(os.path.join(od, 'selected_features.csv'))['feature'].tolist()
audit = pd.read_csv(os.path.join(od, 'feature_audit.csv'))
cat_cols = X_train.select_dtypes('category').columns.tolist()

final_model = joblib.load(os.path.join(md, 'lgb_final.joblib'))
valid_prob = np.load(os.path.join(od, 'pred_lgb_final_valid.npy'))

# ----------------------------------------------------------------------------
# 2. 核心指标
# ----------------------------------------------------------------------------
print('=' * 70)
print('最终模型核心指标（验证集）')
auc = roc_auc_score(y_valid, valid_prob)
pr_auc = average_precision_score(y_valid, valid_prob)
# KS = max |坏样本累计占比 - 好样本累计占比|
bad = valid_prob[y_valid == 1]
good = valid_prob[y_valid == 0]
grid = np.linspace(0, 1, 2000)
cdf_bad = np.searchsorted(np.sort(bad), grid, side='right') / len(bad)
cdf_good = np.searchsorted(np.sort(good), grid, side='right') / len(good)
ks = float(np.max(np.abs(cdf_bad - cdf_good)))
print(f'  AUC   = {auc:.4f}')
print(f'  KS    = {ks:.4f}')
print(f'  PR-AUC= {pr_auc:.4f}')

# 训练集 AUC（用最终模型）——检查过拟合
train_prob = final_model.predict_proba(X_train[selected])[:, 1]
train_auc = roc_auc_score(y_train, train_prob)
print(f'  训练 AUC = {train_auc:.4f}（验证 AUC = {auc:.4f}，过拟合差 {train_auc-auc:.4f}）')

# ----------------------------------------------------------------------------
# 3. 画图：ROC、KS、风险分层
# ----------------------------------------------------------------------------
roc_path = os.path.join(fg, 'final_roc.png')
plot_utils.roc_curve_compare(y_valid, [('最终 LightGBM', valid_prob)], roc_path,
                             title='最终模型 ROC 曲线（验证集）')
ks_path = os.path.join(fg, 'final_ks.png')
_, ks_check = plot_utils.ks_curve(y_valid, valid_prob, ks_path, title='最终模型 KS 曲线')
print(f'  KS（图内复算）= {ks_check:.4f}')

# 风险十分位：按预测概率排序分 10 组，看实际坏样本率是否随分数上升
dec_df = pd.DataFrame({'y': y_valid, 'prob': valid_prob})
dec_df['decile'] = pd.qcut(dec_df['prob'], 10, labels=False, duplicates='drop') + 1
decile_stats = dec_df.groupby('decile').agg(
    actual_bad_rate=('y', 'mean'), pred_mean=('prob', 'mean'), n=('y', 'size')
).reset_index()
dec_path = os.path.join(fg, 'risk_deciles.png')
plot_utils.risk_decile_bar(decile_stats, dec_path, title='风险十分位分层（验证集）')

metrics = {
    'auc': auc, 'ks': ks, 'pr_auc': pr_auc,
    'train_auc': train_auc, 'valid_auc': auc,
    'overfit_gap': train_auc - auc,
}
with open(os.path.join(config.OUTPUTS_DIR, 'metrics', 'evaluation_metrics.json'), 'w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------------
# 4. 消融实验
# ----------------------------------------------------------------------------
print('\n' + '=' * 70)
print('消融实验（保持划分/指标/训练流程一致）')

# 每个保留特征的所属模块（application 或 5 个历史模块之一）
feat_module = dict(zip(audit['feature'], audit['module']))
sel_module = {f: feat_module.get(f, 'application') for f in selected}

def train_eval(features, label):
    feats = [f for f in selected if f in features]
    cat = [c for c in cat_cols if c in feats]
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(X_train[feats], y_train,
          eval_set=[(X_valid[feats], y_valid)], eval_metric='auc',
          categorical_feature=cat,
          callbacks=[lgb.early_stopping(150, verbose=False)])
    a = roc_auc_score(y_valid, m.predict_proba(X_valid[feats])[:, 1])
    print(f'  {label:<22}: AUC={a:.4f}  ({len(feats)} 特征)')
    return a

ablation = {}
ablation['full'] = train_eval(selected, '全模型 (F)')
ablation['app_only'] = train_eval([f for f in selected if sel_module[f] == 'application'], '仅申请主表 (A)')

for mod in config.ABLATION_MODULES:
    keep = [f for f in selected if sel_module[f] != mod]
    ablation[f'minus_{mod}'] = train_eval(keep, f'全模型 - {mod}')

# 每个模块的边际贡献 = 全模型 AUC - 去掉该模块后的 AUC
marginal = {mod: ablation['full'] - ablation[f'minus_{mod}'] for mod in config.ABLATION_MODULES}
for mod, v in marginal.items():
    print(f'  模块 {mod:<14} 边际贡献 = {v:+.4f}')

with open(os.path.join(config.OUTPUTS_DIR, 'metrics', 'ablation_results.json'), 'w', encoding='utf-8') as f:
    json.dump({'ablation': ablation, 'marginal': marginal}, f, ensure_ascii=False, indent=2)

# 消融条形图（模块边际贡献）
ab_path = os.path.join(fg, 'ablation_marginal.png')
mods = list(marginal.keys())
vals = [marginal[m] for m in mods]
order = np.argsort(vals)
mods = [mods[i] for i in order]
vals = [vals[i] for i in order]
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 3.8))
plot_utils._style_ax(ax)
colors = [plot_utils.C_ACCENT if v >= 0 else plot_utils.C_WARN for v in vals]
ax.barh(mods, vals, color=colors, height=0.6)
ax.axvline(0, color=plot_utils.C_GREY, lw=1)
ax.set_xlabel('边际贡献 = 全模型AUC - 去掉该模块后AUC')
ax.set_title('各历史模块的增量价值（消融实验）', fontsize=12, color=plot_utils.C_PRIMARY, pad=10)
for i, v in enumerate(vals):
    ax.text(v, i, f' {v:+.4f}', va='center', ha='left' if v >= 0 else 'right', fontsize=9)
ax.grid(axis='x'); ax.grid(axis='y', visible=False)
fig.tight_layout()
fig.savefig(ab_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('\n已保存: 评估指标、ROC/KS/分层图、消融结果')

# ============================================================================
# 思维导图（阶段 7 分图）
# ============================================================================
mm_tree = {
    'text': '阶段7 评估与消融',
    'children': [
        {'text': '核心指标', 'children': [
            {'text': f'AUC={auc:.4f}'},
            {'text': f'KS={ks:.4f}'},
            {'text': f'PR-AUC={pr_auc:.4f}'},
        ]},
        {'text': '图形诊断', 'children': [
            {'text': 'ROC 曲线'},
            {'text': 'KS 曲线'},
            {'text': '风险十分位分层'},
        ]},
        {'text': '消融实验', 'children': [
            {'text': '仅主表 vs 全模型'},
            {'text': '全模型 - 单模块'},
            {'text': '量化每表增量'},
        ]},
    ],
}
mm_png = os.path.join(config.MINDMAPS_DIR, '07_模型评估与消融实验.png')
mm_pdf = os.path.join(config.MINDMAPS_DIR, '07_模型评估与消融实验.pdf')
mindmap_utils.render_mindmap(mm_tree, mm_png, out_pdf=mm_pdf,
                             title='阶段 7 · 模型评估与消融实验', size=(12, 7.5), fontsize=10)

# ============================================================================
# 学习 PDF
# ============================================================================
d = pdf_utils.Doc('阶段 7：模型评估与消融实验',
                  '回答两个问题：模型效果有多好？每张历史表到底帮了多少？')
d.h1('一、模型效果：三个指标分别回答什么')
d.table([
    ['指标', '数值', '回答的问题'],
    ['ROC-AUC', f'{auc:.4f}', '模型能否把高风险客户排在低风险客户之前？'],
    ['KS', f'{ks:.4f}', '好坏样本累计分布的最大差异有多大？'],
    ['PR-AUC', f'{pr_auc:.4f}', '对少数坏样本的识别表现如何？'],
], col_widths=[30, 30, 105])
d.why('坏样本率只有 8%，是典型的不平衡分类。此时 AUC 会被大量好样本"稀释"，'
      '而 PR-AUC（精确率-召回率曲线下面积）更关注少数坏样本，能更真实地反映风控价值。'
      '所以三个指标要一起看，而不是只看 AUC。')
d.body(f'过拟合检查：训练 AUC {train_auc:.4f} vs 验证 AUC {auc:.4f}，'
       f'差距 {train_auc-auc:.4f}，属于树模型的正常水平（后续可用更强正则或更少叶子数再压）。')

d.h1('二、风险十分位分层：模型排序能力最直观的证明')
d.body('把验证集客户按预测违约概率从低到高分成 10 组，看实际坏样本率是否逐组单调上升。'
       '如果第 10 组（最高风险）的实际坏样本率远高于第 1 组，说明模型的"排序"是有效的。')
d.table([
    ['风险分位', '实际坏样本率', '预测均值', '样本数'],
    *[[f'{int(r.decile)}', f'{r.actual_bad_rate:.2%}', f'{r.pred_mean:.3f}', f'{int(r.n)}']
      for r in decile_stats.itertuples()],
], col_widths=[30, 40, 40, 30])

d.h1('三、消融实验：每张历史表的真实增量')
d.why('单变量分析只能说明"这个特征单独看和违约有关"，但放进模型后可能被别的特征取代、没有增量。'
      '消融实验保持划分、指标、训练流程完全一致，只去掉某个模块，看 AUC 变化，'
      '才能量化"这一模块在模型里的真实增量贡献"。')
d.body('实验设计：全模型(F) vs 仅申请主表(A) vs 全模型分别减去 POS / installments / bureau / previous / credit_card。')
d.table([
    ['配置', 'AUC', '特征数'],
    ['仅申请主表 (A)', f"{ablation['app_only']:.4f}", str(len([f for f in selected if sel_module[f]=='application']))],
    ['全模型 (F)', f"{ablation['full']:.4f}", str(len(selected))],
    *[[f'全模型 - {mod}', f"{ablation['minus_'+mod]:.4f}", str(len([f for f in selected if sel_module[f]!=mod]))]
      for mod in config.ABLATION_MODULES],
], col_widths=[50, 35, 30])
d.body('各模块边际贡献（全模型 AUC - 去掉该模块后 AUC）：')
d.table([
    ['模块', '边际贡献'],
    *[[mod, f'{marginal[mod]:+.4f}'] for mod in config.ABLATION_MODULES],
], col_widths=[50, 40])

d.h1('四、交付物')
d.bullets([
    'outputs/figures/final_roc.png、final_ks.png、risk_deciles.png —— 三张评估图。',
    'outputs/figures/ablation_marginal.png —— 各模块边际贡献图。',
    'outputs/metrics/evaluation_metrics.json、ablation_results.json —— 完整数字。',
])
d.hr()
d.image(roc_path, width=120, caption='图：最终模型 ROC 曲线')
d.image(ks_path, width=120, caption='图：最终模型 KS 曲线')
d.image(dec_path, width=130, caption='图：风险十分位分层')
d.image(ab_path, width=130, caption='图：各历史模块边际贡献')
d.image(mm_png, width=140, caption='图：阶段 7 思维导图')

pdf_path = d.save(os.path.join(config.LEARNINGS_DIR, '07_模型评估与消融实验.pdf'))
print('\n已生成学习 PDF:', pdf_path)
print('DONE 阶段7')
