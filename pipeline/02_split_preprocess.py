# -*- coding: utf-8 -*-
"""
阶段 4：数据集划分与预处理
==========================
目标：
  1. 用 TARGET 做分层(Stratified) 80/20 划分，得到训练集与验证集；
  2. 为 LightGBM 与逻辑回归分别做预处理，且所有"需要学习参数"的步骤只在训练集拟合。

为什么"只在训练集拟合"是这一步最重要的原则？
  预处理里的填补(中位数)、缩放(均值/方差)、编码(类别取值表)，本质上都是从数据里"学"出来的参数。
  如果先对整个数据集预处理再划分，验证集的信息就会"漏"进训练集（数据泄漏），导致验证 AUC 虚高。
  正确做法：先划分，再用训练集学这些参数，然后把同一套参数套到验证集上。
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scipy import sparse
import joblib

import config
import pdf_utils
import mindmap_utils

# ----------------------------------------------------------------------------
# 1. 读取宽表与特征审计，只保留 action=KEEP 的特征
# ----------------------------------------------------------------------------
wide = pd.read_parquet(os.path.join(config.OUTPUTS_DIR, 'data', 'wide_table.parquet'))
audit = pd.read_csv(os.path.join(config.OUTPUTS_DIR, 'data', 'feature_audit.csv'))
keep_feats = audit.loc[audit['action'] == 'KEEP', 'feature'].tolist()
print(f'保留特征数: {len(keep_feats)}（排除 {len(audit) - len(keep_feats)} 个 HIGH 风险特征）')

X = wide[keep_feats]
y = wide[config.TARGET]

# 识别数值 / 类别列（类别列 = object 类型）
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]
print(f'数值特征 {len(num_cols)} 个，类别特征 {len(cat_cols)} 个')

# ----------------------------------------------------------------------------
# 2. 分层划分（固定种子）
# ----------------------------------------------------------------------------
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=config.SEED, stratify=y
)
split_info = {
    'train_n': int(len(X_train)), 'valid_n': int(len(X_valid)),
    'train_target_rate': float(y_train.mean()), 'valid_target_rate': float(y_valid.mean()),
    'n_features': len(keep_feats), 'n_numeric': len(num_cols), 'n_categorical': len(cat_cols),
    'n_excluded_high_risk': int((audit['action'] == 'EXCLUDE').sum()),
}
print(f'训练集 {split_info["train_n"]} 行(坏样本 {split_info["train_target_rate"]:.4%})，'
      f'验证集 {split_info["valid_n"]} 行(坏样本 {split_info["valid_target_rate"]:.4%})')

# ----------------------------------------------------------------------------
# 3. LightGBM 预处理：类别列做 Ordinal 编码；数值列保持原样（NaN 原生支持）
# ----------------------------------------------------------------------------
ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train_lgb = X_train.copy()
X_valid_lgb = X_valid.copy()
if cat_cols:
    X_train_lgb[cat_cols] = ord_enc.fit_transform(X_train[cat_cols])
    X_valid_lgb[cat_cols] = ord_enc.transform(X_valid[cat_cols])
    for c in cat_cols:
        X_train_lgb[c] = X_train_lgb[c].astype('category')
        X_valid_lgb[c] = X_valid_lgb[c].astype('category')
    print(f'LightGBM 类别编码完成，类别数示例: '
          f'{ {c: int(ord_enc.categories_[i].size) for i, c in enumerate(cat_cols[:3])} }')

# ----------------------------------------------------------------------------
# 4. 逻辑回归预处理：数值(中位数填补+缺失标记+标准化) + 类别(one-hot)
# ----------------------------------------------------------------------------
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
    ('scaler', StandardScaler()),
])
cat_pipe = Pipeline([
    ('fill', SimpleImputer(strategy='constant', fill_value='__MISSING__')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True)),
])
preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols),
], sparse_threshold=0)  # 强制输出稀疏矩阵，节省内存
X_train_lr = preprocessor.fit_transform(X_train)
X_valid_lr = preprocessor.transform(X_valid)
if not sparse.issparse(X_train_lr):
    X_train_lr = sparse.csr_matrix(X_train_lr)
if not sparse.issparse(X_valid_lr):
    X_valid_lr = sparse.csr_matrix(X_valid_lr)
lr_feature_names = preprocessor.get_feature_names_out().tolist()
print(f'逻辑回归特征矩阵维度: {X_train_lr.shape}（含 one-hot 展开后 {len(lr_feature_names)} 维）')

# ----------------------------------------------------------------------------
# 5. 保存所有中间产物
# ----------------------------------------------------------------------------
od = os.path.join(config.OUTPUTS_DIR, 'data')
X_train_lgb.to_parquet(os.path.join(od, 'train_lgb.parquet'))
X_valid_lgb.to_parquet(os.path.join(od, 'valid_lgb.parquet'))
np.save(os.path.join(od, 'y_train.npy'), y_train.values)
np.save(os.path.join(od, 'y_valid.npy'), y_valid.values)
sparse.save_npz(os.path.join(od, 'train_lr.npz'), X_train_lr)
sparse.save_npz(os.path.join(od, 'valid_lr.npz'), X_valid_lr)
with open(os.path.join(od, 'lr_feature_names.json'), 'w', encoding='utf-8') as f:
    json.dump(lr_feature_names, f, ensure_ascii=False)

md = os.path.join(config.OUTPUTS_DIR, 'models')
joblib.dump(ord_enc, os.path.join(md, 'lgb_ordinal.joblib'))
joblib.dump(preprocessor, os.path.join(md, 'lr_preprocessor.joblib'))
with open(os.path.join(config.OUTPUTS_DIR, 'metrics', 'split_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(split_info, f, ensure_ascii=False, indent=2)

print('\n已保存: train/valid (lgb + lr)、编码器、划分摘要')

# ============================================================================
# 思维导图（阶段 4 分图）
# ============================================================================
mm_tree = {
    'text': '阶段4 划分与预处理',
    'children': [
        {'text': '分层划分 80/20', 'children': [
            {'text': '按 TARGET 分层\n(坏样本率 8%)'},
            {'text': '固定随机种子 42'},
            {'text': '两组坏样本率接近'},
        ]},
        {'text': 'LightGBM 预处理', 'children': [
            {'text': '类别 Ordinal 编码'},
            {'text': 'NaN 原生支持'},
            {'text': '无需缩放'},
        ]},
        {'text': '逻辑回归预处理', 'children': [
            {'text': '中位数填补+缺失标记'},
            {'text': 'One-Hot 编码'},
            {'text': '标准化'},
        ]},
        {'text': '防泄漏原则', 'children': [
            {'text': '只在训练集拟合'},
            {'text': '同一套参数套验证集'},
        ]},
    ],
}
mm_png = os.path.join(config.MINDMAPS_DIR, '04_数据集划分与预处理.png')
mm_pdf = os.path.join(config.MINDMAPS_DIR, '04_数据集划分与预处理.pdf')
mindmap_utils.render_mindmap(mm_tree, mm_png, out_pdf=mm_pdf,
                             title='阶段 4 · 数据集划分与预处理', size=(13, 8), fontsize=10)

# ============================================================================
# 学习 PDF
# ============================================================================
d = pdf_utils.Doc('阶段 4：数据集划分与预处理',
                  '分层划分训练/验证集，并为两种模型做各自的预处理')
d.h1('一、这一步要做什么')
d.body('先把 229 个保留特征 + TARGET 划分成 80% 训练集 / 20% 验证集，'
       '再分别准备两套预处理：一套给 LightGBM，一套给逻辑回归。')
d.body(f'划分结果：训练集 {split_info["train_n"]} 行、验证集 {split_info["valid_n"]} 行；'
       f'坏样本率分别为 {split_info["train_target_rate"]:.2%} 和 {split_info["valid_target_rate"]:.2%}，几乎一致。')

d.h1('二、为什么要分层划分（Stratified Split）')
d.why('坏样本率只有 8.07%，样本高度不平衡。如果纯随机划分，验证集里可能分到极少甚至 0 个坏样本，'
      '那么 AUC/KS 这些指标就测不准，甚至无法计算。分层划分保证训练集和验证集里坏样本比例都≈8%，'
      '评估才公平、稳定。')
d.body('另外固定随机种子（random_state=42），是为了让每次运行划分结果完全一样——模型结果可复现，'
       '是面试时最容易被追问、也最能体现专业度的细节之一。')

d.h1('三、为什么预处理要"只在训练集拟合"')
d.why('填补用的中位数、标准化用的均值方差、编码用的类别取值表，都是从数据里"学"出来的参数。'
      '如果先把整份数据预处理再划分，验证集的信息就混进了训练集——这是最常见、最隐蔽的数据泄漏。'
      '正确做法永远是：先划分 → 只在训练集 fit → 再用同一套参数 transform 验证集。')
d.body('代码里 LightGBM 的 OrdinalEncoder 和逻辑回归的 Pipeline 都是先 fit_transform(X_train)'
       '再 transform(X_valid)，严格遵循这条原则。')

d.h1('四、为什么两种模型要两套预处理')
d.table([
    ['预处理步骤', 'LightGBM', '逻辑回归', '原因'],
    ['缺失值', '原生支持 NaN', '中位数填补 + 缺失标记', '树能对 NaN 分裂；线性模型必须有数值'],
    ['类别变量', 'Ordinal 编码', 'One-Hot 编码', '树可比较大小序；线性模型需要 0/1 哑变量'],
    ['数值缩放', '不需要', '需要标准化', '树只看切分点；线性模型受量纲影响'],
    ['非线性', '自动学习', '需要显式构造', '树的本质 vs 线性模型'],
], col_widths=[30, 42, 52, 62])
d.body('注意逻辑回归加了"缺失标记"（add_indicator=True）：因为 EDA 已发现"信息缺失本身"与违约风险有关，'
       '这个 0/1 标记让线性模型也能利用缺失信息，而不只是用中位数糊一个值。')

d.h1('五、交付物')
d.bullets([
    'train_lgb/valid_lgb.parquet —— LightGBM 用的特征（类别已编码，数值保留 NaN）。',
    'train_lr/valid_lr.npz —— 逻辑回归用的稀疏矩阵（one-hot 展开后约数百维）。',
    'lgb_ordinal.joblib / lr_preprocessor.joblib —— 保存的编码器与预处理流水线，供后续复现。',
])
d.warn('注意：ONE-HOT 展开后逻辑回归的特征数会明显多于原始 229 列（类别变量被拆成多个 0/1 列），'
       '这是正常现象，不是数据出错。')
d.hr()
d.image(mm_png, width=150, caption='图：阶段 4 思维导图')

pdf_path = d.save(os.path.join(config.LEARNINGS_DIR, '04_数据集划分与预处理.pdf'))
print('\n已生成学习 PDF:', pdf_path)
print('DONE 阶段4')
