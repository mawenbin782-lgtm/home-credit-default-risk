# -*- coding: utf-8 -*-
"""
阶段 5：建立 LR + LightGBM 基线
===============================
目标：先用 LightGBM 快速跑出"第一个验证集 AUC"，再建逻辑回归对照模型。

为什么先 LightGBM 再 LR？
  我们构造的大多是数值型聚合特征，LightGBM 这类树模型能直接吃 NaN、自动学非线性，
  最快验证这些特征有没有预测价值。拿到一个可靠的 AUC 后，再建 LR 作为"可解释基线"，
  用它来看每个变量如何线性地影响违约概率。
"""
import json
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib

import config
import pdf_utils
import mindmap_utils
import plot_utils

od = os.path.join(config.OUTPUTS_DIR, 'data')
md = os.path.join(config.OUTPUTS_DIR, 'models')

# ----------------------------------------------------------------------------
# 1. 读取预处理后的数据
# ----------------------------------------------------------------------------
X_train_lgb = pd.read_parquet(os.path.join(od, 'train_lgb.parquet'))
X_valid_lgb = pd.read_parquet(os.path.join(od, 'valid_lgb.parquet'))
y_train = np.load(os.path.join(od, 'y_train.npy'))
y_valid = np.load(os.path.join(od, 'y_valid.npy'))

from scipy import sparse
X_train_lr = sparse.load_npz(os.path.join(od, 'train_lr.npz'))
X_valid_lr = sparse.load_npz(os.path.join(od, 'valid_lr.npz'))

cat_cols = X_train_lgb.select_dtypes('category').columns.tolist()
print(f'LightGBM 训练集 {X_train_lgb.shape}，类别列 {len(cat_cols)} 个')

# ----------------------------------------------------------------------------
# 2. LightGBM 基线
# ----------------------------------------------------------------------------
print('\n' + '=' * 70)
print('训练 LightGBM 基线')
lgb_model = lgb.LGBMClassifier(
    objective='binary',
    metric='auc',
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_estimators=1500,
    random_state=config.SEED,
    n_jobs=-1,
    verbose=-1,
)
lgb_model.fit(
    X_train_lgb, y_train,
    eval_set=[(X_valid_lgb, y_valid)],
    eval_metric='auc',
    categorical_feature=cat_cols,
    callbacks=[lgb.early_stopping(150, verbose=False)],
)
lgb_train_auc = roc_auc_score(y_train, lgb_model.predict_proba(X_train_lgb)[:, 1])
lgb_valid_auc = roc_auc_score(y_valid, lgb_model.predict_proba(X_valid_lgb)[:, 1])
lgb_best_iter = lgb_model.best_iteration_
print(f'LightGBM: 训练 AUC={lgb_train_auc:.4f}  验证 AUC={lgb_valid_auc:.4f}  (最佳迭代 {lgb_best_iter})')

# ----------------------------------------------------------------------------
# 3. 逻辑回归基线
# ----------------------------------------------------------------------------
print('\n' + '=' * 70)
print('训练逻辑回归基线')
lr_model = LogisticRegression(
    C=1.0, max_iter=2000, solver='lbfgs',
    class_weight='balanced', random_state=config.SEED, n_jobs=-1,
)
lr_model.fit(X_train_lr, y_train)
lr_train_auc = roc_auc_score(y_train, lr_model.predict_proba(X_train_lr)[:, 1])
lr_valid_auc = roc_auc_score(y_valid, lr_model.predict_proba(X_valid_lr)[:, 1])
print(f'逻辑回归: 训练 AUC={lr_train_auc:.4f}  验证 AUC={lr_valid_auc:.4f}')

# ----------------------------------------------------------------------------
# 4. 保存模型、预测、指标、ROC 图
# ----------------------------------------------------------------------------
joblib.dump(lgb_model, os.path.join(md, 'lgb_baseline.joblib'))
joblib.dump(lr_model, os.path.join(md, 'lr_baseline.joblib'))
np.save(os.path.join(od, 'pred_lgb_valid.npy'), lgb_model.predict_proba(X_valid_lgb)[:, 1])
np.save(os.path.join(od, 'pred_lr_valid.npy'), lr_model.predict_proba(X_valid_lr)[:, 1])

metrics = {
    'lgb_train_auc': lgb_train_auc, 'lgb_valid_auc': lgb_valid_auc,
    'lgb_best_iter': lgb_best_iter,
    'lr_train_auc': lr_train_auc, 'lr_valid_auc': lr_valid_auc,
}
with open(os.path.join(config.OUTPUTS_DIR, 'metrics', 'baseline_metrics.json'), 'w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

roc_path = os.path.join(config.OUTPUTS_DIR, 'figures', 'baseline_roc.png')
plot_utils.roc_curve_compare(
    y_valid,
    [('LightGBM', lgb_model.predict_proba(X_valid_lgb)[:, 1]),
     ('逻辑回归', lr_model.predict_proba(X_valid_lr)[:, 1])],
    roc_path, title='基线模型 ROC 曲线（验证集）',
)
print('\n已保存: 模型、验证集预测、baseline_metrics.json、ROC 图')

# ============================================================================
# 思维导图（阶段 5 分图）
# ============================================================================
mm_tree = {
    'text': '阶段5 基线模型',
    'children': [
        {'text': 'LightGBM 基线', 'children': [
            {'text': '梯度提升树'},
            {'text': '原生吃 NaN'},
            {'text': '早停选最佳迭代'},
        ]},
        {'text': '逻辑回归对照', 'children': [
            {'text': '线性可解释'},
            {'text': 'class_weight=balanced'},
            {'text': '标准化后的输入'},
        ]},
        {'text': '评估', 'children': [
            {'text': '验证集 AUC'},
            {'text': '训练 vs 验证\n看是否过拟合'},
        ]},
    ],
}
mm_png = os.path.join(config.MINDMAPS_DIR, '05_建立基线模型.png')
mm_pdf = os.path.join(config.MINDMAPS_DIR, '05_建立基线模型.pdf')
mindmap_utils.render_mindmap(mm_tree, mm_png, out_pdf=mm_pdf,
                             title='阶段 5 · 建立 LR + LightGBM 基线', size=(12, 7.5), fontsize=10)

# ============================================================================
# 学习 PDF
# ============================================================================
d = pdf_utils.Doc('阶段 5：建立 LR + LightGBM 基线',
                  '先跑出一个可靠的验证集 AUC，再建立可解释的线性对照')
d.h1('一、这一步要做什么')
d.body('在预处理好的数据上训练两个"基线"模型：一个梯度提升树 LightGBM，一个逻辑回归 LR，'
       '用验证集 AUC 作为它们的第一份成绩单。')
d.table([
    ['模型', '训练 AUC', '验证 AUC', '说明'],
    ['LightGBM', f'{lgb_train_auc:.4f}', f'{lgb_valid_auc:.4f}', f'最佳迭代 {lgb_best_iter}'],
    ['逻辑回归', f'{lr_train_auc:.4f}', f'{lr_valid_auc:.4f}', 'C=1.0, balanced'],
], col_widths=[30, 30, 30, 70])

d.h1('二、为什么先跑 LightGBM')
d.why('我们构造的大多是数值型聚合特征。树模型能直接处理 NaN（对缺失值单独分裂）、自动学非线性交互，'
      '所以它能在最短时间里告诉我们"这些特征到底有没有预测价值"。拿到可靠的 AUC 后，'
      '再建 LR 作为可解释的对照，看变量如何线性地影响违约。')

d.h1('三、两个模型的关键差异')
d.table([
    ['维度', 'LightGBM', '逻辑回归'],
    ['模型本质', '集成决策树，非线性', '线性模型，可解释'],
    ['缺失值', '原生支持 NaN', '已中位数填补+标记'],
    ['类别变量', 'Ordinal 编码', 'One-Hot 编码'],
    ['缩放', '不需要', '已标准化'],
    ['类别不平衡', '样本加权(内置)', 'class_weight=balanced'],
    ['解释性', '特征重要性/SHAP', '系数正负可直接看'],
], col_widths=[30, 70, 70])

d.h1('四、怎么判断结果好不好')
d.body(f'两个模型的验证 AUC 分别是 {lgb_valid_auc:.4f}（LightGBM）和 {lr_valid_auc:.4f}（LR）。')
d.bullets([
    'AUC 越接近 1 越好；0.5 等于瞎猜。风控场景验证 AUC 0.75+ 已经是有区分度的水平。',
    '训练 AUC 与验证 AUC 的差距反映过拟合：差距越小越好。LightGBM 用了早停(early stopping)防止过拟合。',
    'LR 明显低于 LightGBM 是正常且预期的——线性模型没有能力学复杂的非线性关系，它的价值在于"可解释"。',
])
d.warn('注意：这里的验证集既用于早停又用于报 AUC，会略微乐观。更严格的做法是用交叉验证（阶段 6），'
       '在完全没碰过的折上评估，得到无偏的泛化估计。')

d.h1('五、交付物与下一步')
d.bullets([
    'outputs/models/lgb_baseline.joblib、lr_baseline.joblib —— 训练好的模型。',
    'outputs/data/pred_lgb_valid.npy、pred_lr_valid.npy —— 验证集预测概率，供后续画曲线。',
    'outputs/figures/baseline_roc.png —— 两模型 ROC 曲线对比。',
])
d.body('下一步（阶段 6）：用 StratifiedKFold 交叉验证得到稳定的 OOF AUC，'
       '聚合特征重要性，并据此去除冗余 / 低贡献的变量。')
d.hr()
d.image(roc_path, width=130, caption='图：基线模型 ROC 曲线')
d.image(mm_png, width=140, caption='图：阶段 5 思维导图')

pdf_path = d.save(os.path.join(config.LEARNINGS_DIR, '05_建立基线模型.pdf'))
print('\n已生成学习 PDF:', pdf_path)
print('DONE 阶段5')
