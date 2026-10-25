# -*- coding: utf-8 -*-
"""
项目全局配置：路径、常量、随机种子、字体、模块映射。

为什么单独建一个 config.py？
—— 后续每个阶段的脚本都要读写同样的数据、用同样的随机种子、识别同样的特征模块。
   集中写在这里，改一处、全局生效，避免每个脚本里散落重复的硬编码。
"""
import os

# 项目根目录：本文件在 pipeline/ 下，往上一级就是项目根
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR     = os.path.join(BASE, 'data')
FEATURES_DIR = os.path.join(BASE, 'features')
OUTPUTS_DIR  = os.path.join(BASE, 'outputs')
LEARNINGS_DIR = os.path.join(BASE, 'learnings')
MINDMAPS_DIR = os.path.join(BASE, 'mindmaps')

# 输出子目录（模型、指标、图、中间宽表）
for d in [OUTPUTS_DIR, LEARNINGS_DIR, MINDMAPS_DIR,
          os.path.join(OUTPUTS_DIR, 'data'),
          os.path.join(OUTPUTS_DIR, 'models'),
          os.path.join(OUTPUTS_DIR, 'metrics'),
          os.path.join(OUTPUTS_DIR, 'figures')]:
    os.makedirs(d, exist_ok=True)

# 固定随机种子：保证划分、交叉验证、模型训练全部可复现
SEED = 42

TARGET = 'TARGET'          # 违约标签列
ID_COL = 'SK_ID_CURR'      # 客户主键

APPLICATION_CSV = os.path.join(DATA_DIR, 'application_train.csv')

# 融合用的 5 份历史特征文件 -> (文件名, 是否 parquet)
# 注意：bureau_features.parquet 内部已包含 bureau_balance 的 BUR_BB_* 特征，
#       所以 bureau_balance_features.parquet 是冗余中间文件，融合时不要重复合并。
FEATURE_FILES = {
    'bureau':       ('bureau_features.parquet',       True),
    'credit_card':  ('credit_card_features.parquet',  True),
    'installments': ('installments_features.parquet', True),
    'pos':          ('pos_features.csv',              False),
    'previous':     ('previous_application_features.parquet', True),
}

# 消融实验里的 5 个"业务模块"（bureau 模块 = 外部征信，同时包含 BUR_ 与 BUR_BB_ 两类前缀）
ABLATION_MODULES = ['pos', 'installments', 'bureau', 'previous', 'credit_card']

# 列名前缀 -> 业务模块（用于识别某列属于哪个历史模块）。
# 顺序敏感：'BUR_BB_' 必须排在 'BUR_' 之前，否则 BUR_BB_xxx 会被误判成 bureau。
ABLATION_PREFIXES = [
    ('BUR_BB_', 'bureau'),
    ('BUR_',    'bureau'),
    ('CC_',     'credit_card'),
    ('INS_',    'installments'),
    ('POS_',    'pos'),
    ('PREV_',   'previous'),
]

# 中文字体（Windows 自带，用于 matplotlib 思维导图 与 reportlab PDF）
FONT_SIMHEI = r'C:\Windows\Fonts\simhei.ttf'
FONT_MSYH   = r'C:\Windows\Fonts\msyh.ttc'


def module_of_column(col: str):
    """根据列名前缀判断它属于哪个历史模块；不属于任何模块返回 None（即主表特征）。"""
    for prefix, mod in ABLATION_PREFIXES:  # 已按 BUR_BB_ 在 BUR_ 之前排好序
        if col.startswith(prefix):
            return mod
    return None
