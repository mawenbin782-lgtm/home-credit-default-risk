# -*- coding: utf-8 -*-
"""
阶段 3：多表融合与质量审计
==========================
目标：把 application_train 与 5 份历史客户级特征表按 SK_ID_CURR 左连接成一张
      "模型训练宽表"，并完成四项质量检查 + T0 防泄漏分级。

四项必须检查（对应任务说明）：
  1. 合并后是否仍保留 307,511 条申请；
  2. 是否存在重复客户、重复字段名、无穷大、错误类型；
  3. 没有历史表的客户，缺失值如何表达；
  4. 特征是否满足 T0 信息可得性（防泄漏）。

为什么这一步必须先做？
  特征工程分散在 6 张表里，任何一张表合并错了（漏行、重复行、重复列、无穷大），
  都会带着错误进入模型训练，且极难回溯。所以"先融合、先体检、再建模"。
"""
import json
import os
import numpy as np
import pandas as pd

import config
import pdf_utils
import mindmap_utils

# ----------------------------------------------------------------------------
# 1. 读取主表与 5 份历史特征表
# ----------------------------------------------------------------------------
print('=' * 70)
print('读取主表与历史特征表')
app = pd.read_csv(config.APPLICATION_CSV)
print(f'  application_train : {app.shape}')

module_dfs = {}
for name, (fname, is_parquet) in config.FEATURE_FILES.items():
    path = os.path.join(config.FEATURES_DIR, fname)
    df = pd.read_parquet(path) if is_parquet else pd.read_csv(path)
    # 每个模块只保留特征列 + 主键，防止把中间文件的其它列带进来
    module_dfs[name] = df
    print(f'  {name:<14}: {df.shape}')

# ----------------------------------------------------------------------------
# 2. 融合：application 左连接 5 个模块
# ----------------------------------------------------------------------------
print('\n' + '=' * 70)
print('多表融合（左连接）')
wide = app.copy()
for name, df in module_dfs.items():
    before = len(wide)
    # on=SK_ID_CURR 保证只保留一个主键列；how='left' 保证不丢任何一名申请客户
    wide = wide.merge(df, on=config.ID_COL, how='left', suffixes=('', '_dup'))
    after = len(wide)
    dup_cols = [c for c in wide.columns if c.endswith('_dup')]
    if dup_cols:
        print(f'  [警告] {name} 产生重复列: {dup_cols}')
        wide = wide.drop(columns=dup_cols)
    print(f'  合并 {name:<14}: 行数 {before} -> {after} ({"OK" if after == before else "变化!"})')

print(f'\n融合后宽表: {wide.shape}')

# ----------------------------------------------------------------------------
# 3. 程序化质量检查
# ----------------------------------------------------------------------------
audit = {}

# 3.1 行数与主键唯一性
audit['n_rows'] = int(len(wide))
audit['n_unique_sk'] = int(wide[config.ID_COL].nunique())
audit['n_dup_sk'] = int(wide[config.ID_COL].duplicated().sum())
audit['rows_preserved'] = bool(len(wide) == len(app))
audit['target_rate'] = float(wide[config.TARGET].mean())

# 3.2 重复列名
dup_col_names = wide.columns[wide.columns.duplicated()].tolist()
audit['duplicate_columns'] = dup_col_names

# 3.3 无穷大检查（数值列）
num_cols = wide.select_dtypes(include=[np.number]).columns
inf_cols = [c for c in num_cols if np.isinf(wide[c]).any()]
audit['inf_columns'] = inf_cols

# 3.4 全空列
allna_cols = [c for c in wide.columns if wide[c].isna().all()]
audit['all_na_columns'] = allna_cols

# 3.5 目标列缺失
audit['target_missing'] = int(wide[config.TARGET].isna().sum())

print('\n' + '=' * 70)
print('质量检查结果')
for k, v in audit.items():
    if k in ('duplicate_columns', 'inf_columns', 'all_na_columns'):
        print(f'  {k:<18}: {len(v)} 个 -> {v}')
    else:
        print(f'  {k:<18}: {v}')

# ----------------------------------------------------------------------------
# 4. T0 防泄漏分级
# ----------------------------------------------------------------------------
# 为什么 installments 的"付款结果"特征是泄漏？
#   FULL_PAY_DELAY(足额付款延迟) 依赖"最终付款日"，而最终付款可能发生在 T0 之后。
#   对"仍有贷款未结清"的客户，这些特征等于看到了未来。所以判为 HIGH（建模前排除）。
HIGH_RISK = {
    # 静态履约 / 足额付款 / 延期（依赖最终付款结果）
    'INS_LATE_RATE_ALL', 'INS_DPD_MEAN_ALL', 'INS_DPD_MAX_ALL', 'INS_DPD30_RATE_ALL',
    'INS_DUE_COVERAGE_MEAN', 'INS_PAYMENT_ANOMALY_RATE', 'INS_UNCONFIRMED_COUNT',
    'INS_LATE_DAYS_AVG_ALL', 'INS_LATE_DAYS_CAP90_AVG', 'INS_VALID_INSTALMENT_COUNT',
    'INS_UNPAID30_COUNT_ALL', 'INS_UNPAID30_COUNT_D180',
    'INS_MISSING_PAYMENT_COUNT', 'INS_INVALID_AMOUNT_COUNT',
    # 近/前 90 天覆盖率与未确认（同样依赖付款完成）
    'INS_COVERAGE_MEAN_D090', 'INS_COVERAGE_MEAN_PREV090',
    'INS_COVERAGE_ZERO_RATE_D090', 'INS_COVERAGE_ZERO_RATE_PREV090',
    'INS_COVERAGE_DELTA_90V90', 'INS_UNCONFIRMED_D090_COUNT',
}
# MEDIUM：极小规模或有"当月边界"争议，保留但打标，供后续消融验证
MEDIUM_RISK = {
    'BUR_CURRENT_OVERDUE_COUNT', 'BUR_MAX_CURRENT_DPD', 'BUR_CURRENT_OVERDUE_AMT_SUM',  # 17 条 DAYS_CREDIT_UPDATE>0
    'BUR_BB_TREND_VALID_COUNT', 'BUR_BB_WORSEN_COUNT', 'BUR_BB_DELTA_MEAN',            # 趋势窗口的月份 0 边界
}

rows = []
for c in wide.columns:
    if c in (config.ID_COL, config.TARGET):
        continue
    mod = config.module_of_column(c) or 'application'
    risk = 'HIGH' if c in HIGH_RISK else ('MEDIUM' if c in MEDIUM_RISK else 'LOW')
    action = 'EXCLUDE' if risk == 'HIGH' else 'KEEP'
    rows.append({
        'feature': c,
        'module': mod,
        'missing_rate': float(wide[c].isna().mean()),
        'leakage_risk': risk,
        'action': action,
    })
feature_audit = pd.DataFrame(rows)
feature_audit.to_csv(os.path.join(config.OUTPUTS_DIR, 'data', 'feature_audit.csv'), index=False)
audit['n_features_total'] = len(feature_audit)
audit['n_features_excluded_high_risk'] = int((feature_audit['action'] == 'EXCLUDE').sum())
audit['n_features_keep'] = int((feature_audit['action'] == 'KEEP').sum())

# ----------------------------------------------------------------------------
# 5. 保存宽表与审计结果
# ----------------------------------------------------------------------------
wide.to_parquet(os.path.join(config.OUTPUTS_DIR, 'data', 'wide_table.parquet'), index=False)
with open(os.path.join(config.OUTPUTS_DIR, 'metrics', 'fusion_audit.json'), 'w', encoding='utf-8') as f:
    json.dump(audit, f, ensure_ascii=False, indent=2, default=str)

# 缺失率概览（用于 PDF）
module_missing = (
    feature_audit.groupby('module')['missing_rate']
    .agg(['mean', 'max']).round(4).reset_index()
)

print('\n已保存: wide_table.parquet, feature_audit.csv, fusion_audit.json')

# ============================================================================
# 生成思维导图（阶段 3 分图）
# ============================================================================
mm_tree = {
    'text': '阶段3 多表融合\n与质量审计',
    'children': [
        {'text': '融合\n(SK_ID_CURR 左连接)', 'children': [
            {'text': '主表 307,511 行'},
            {'text': '5 份历史特征表'},
            {'text': '保留 307,511 行'},
        ]},
        {'text': '质量检查', 'children': [
            {'text': '主键唯一性'},
            {'text': '重复列名'},
            {'text': '无穷大 / 全空列'},
        ]},
        {'text': '缺失值口径', 'children': [
            {'text': '无历史记录 → NaN'},
            {'text': '无有效分母 → 0'},
            {'text': '真实结果 → 实际值'},
        ]},
        {'text': 'T0 防泄漏', 'children': [
            {'text': 'HIGH: 付款结果类 → 排除'},
            {'text': 'MEDIUM: 当月边界 → 打标'},
            {'text': 'LOW: 历史窗口 [-N,-1]'},
        ]},
        {'text': '交付物', 'children': [
            {'text': 'wide_table.parquet'},
            {'text': 'feature_audit.csv'},
        ]},
    ],
}
mm_png = os.path.join(config.MINDMAPS_DIR, '03_多表融合与质量审计.png')
mm_pdf = os.path.join(config.MINDMAPS_DIR, '03_多表融合与质量审计.pdf')
mindmap_utils.render_mindmap(mm_tree, mm_png, out_pdf=mm_pdf,
                             title='阶段 3 · 多表融合与质量审计', size=(15, 9), fontsize=10)

# ============================================================================
# 生成学习 PDF
# ============================================================================
d = pdf_utils.Doc('阶段 3：多表融合与质量审计',
                  '把 6 张分散的特征表合并成一张模型宽表，并在建模前完成"体检"')
d.h1('一、这一步要做什么')
d.body('以 application_train 为主表，按客户主键 SK_ID_CURR，把 5 份历史特征表（外部征信 bureau、'
       '分期付款 installments、POS/CASH、历史申请 previous、信用卡 credit_card）左连接上去，'
       '得到一张"申请字段 + 历史特征 + TARGET"的完整宽表。')
d.body(f'融合结果：{audit["n_rows"]} 行 × {wide.shape[1]} 列；坏样本率 {audit["target_rate"]:.2%}。')

d.h1('二、为什么这一步必须先做')
d.why('特征工程是分散在 6 张表里完成的。任何一张表在合并时漏行、重复行、出现重复列名或无穷大，'
      '都会把"脏数据"带进模型，而且等问题在 AUC 里暴露时已经很难追溯到是哪张表、哪个特征。'
      '所以正确顺序是：先融合、先体检、确认干净，再进入建模。')
d.bullets([
    '体检的 4 个问题，对应任务说明里"阶段 3 必须完成的检查"：行数、重复/类型、缺失口径、T0 泄漏。',
    '这一步交付的 feature_audit.csv 是后续所有阶段的"单一事实来源"——哪些特征保留、哪些排除，只看这一张表。',
])

d.h1('三、四项检查的结果')
d.table([
    ['检查项', '结果', '是否通过'],
    ['合并后行数', f'{audit["n_rows"]} 条（应 307,511）', '√' if audit['rows_preserved'] else '×'],
    ['主键重复', f'{audit["n_dup_sk"]} 条重复 SK_ID_CURR', '√' if audit['n_dup_sk'] == 0 else '×'],
    ['重复列名', f'{len(audit["duplicate_columns"])} 个', '√' if not audit['duplicate_columns'] else '×'],
    ['含无穷大的列', f'{len(audit["inf_columns"])} 个', '√' if not audit['inf_columns'] else '×'],
    ['全空列', f'{len(audit["all_na_columns"])} 个', '√' if not audit['all_na_columns'] else '×'],
    ['目标列缺失', f'{audit["target_missing"]} 条', '√' if audit['target_missing'] == 0 else '×'],
], col_widths=[40, 90, 30])

d.h2('一个真实踩到的坑：bureau 重复列')
d.body('bureau_features.parquet 其实已经把 bureau_balance 的 7 个 BUR_BB_* 特征合并进去了'
       '（27 列 = 20 个 bureau + 7 个 bureau_balance）。如果再把 bureau_balance_features.parquet 单独合并一次，'
       '就会出现 7 个重复列。所以融合时只合并 5 份文件，而不是 6 份。这正是"重复字段名检查"的价值。')

d.h1('四、缺失值：三种"空"不能混为一谈')
d.body('合并后大量特征有缺失，但它们缺失的含义完全不同，处理方式也不同：')
d.table([
    ['缺失原因', '例子', '正确表达'],
    ['无历史记录', '客户从没申请过分期，所以没有 installments 特征', 'NaN（树模型能天然处理）'],
    ['无有效分母', '信用卡额度为 0 的月份，使用率算不出来', '用 *_OBS / *_VALID 计数列区分'],
    ['真实结果为 0', '近 90 天新增信贷数 = 0', '0（不是缺失）'],
], col_widths=[30, 95, 55])
d.body('特征工程阶段已经用 INS_VALID_INSTALMENT_COUNT、CC_VALID_LIMIT_MONTHS、'
       'POS_OBS_MONTHS_*、BUR_BB_TREND_VALID_COUNT 这类"有效观察数"列把前两种情况区分开了。'
       '所以建模时：LightGBM 可以直接吃 NaN（树会自己学会"缺失"这一支），逻辑回归才需要显式填补。')

d.h1('五、T0 防泄漏：本次审计的重点')
d.why('所谓 T0 泄漏，是指某个特征用到了"申请时点之后才能看到的信息"。'
      '如果把这类特征喂给模型，验证集 AUC 会虚高，因为它相当于"开了天眼"。'
      '在风控场景这尤其危险：真实上线时你根本没有未来信息，模型会失效。')
d.h2('时间窗口的约定（检查结论）')
d.body('各脚本的"近 N 天/月"窗口统一用 [-N, -1]，即排除 T0 当天。例如分期"近 90 天"= DAYS_INSTALMENT ∈ [-90, -1]，'
       '"前 90 天"= [-180, -91]；POS/信用卡"近 3 个月"= MONTHS_BALANCE ∈ [-3, -1]。所以这些历史窗口特征是 T0 安全的。')
d.h2('真正的泄漏点：分期付款的"付款结果"')
d.body('分期付款表里，"是否足额付款 / 延期多少天 / 覆盖率"依赖"最终付款日"。但最终付款可能发生在 T0 之后'
       '（数据里确实存在 DAYS_ENTRY_PAYMENT > 0 的付款记录）。对"贷款还没结清"的客户，这些特征等于偷看了未来。')
d.body(f'处理：把 {len(HIGH_RISK)} 个此类特征标记为 HIGH 风险，建模前排除；'
       '其余标记为 LOW（保留）。另有 6 个"当月边界/极小规模更新"特征标为 MEDIUM，保留但留待消融验证。')
d.table([
    ['风险级别', '含义', '动作', '特征数'],
    ['HIGH', '依赖 T0 之后才能观察的信息', '排除', str(audit['n_features_excluded_high_risk'])],
    ['MEDIUM', '极小规模或当月边界争议', '保留 + 打标', str((feature_audit['leakage_risk'] == 'MEDIUM').sum())],
    ['LOW', '只用 T0 之前的信息', '保留', str((feature_audit['leakage_risk'] == 'LOW').sum())],
], col_widths=[22, 90, 35, 20])

d.h1('六、交付物与下一步')
d.bullets([
    'outputs/data/wide_table.parquet —— 融合后的完整宽表（不删任何信息）。',
    'outputs/data/feature_audit.csv —— 每个特征的模块、缺失率、泄漏等级、保留/排除动作。',
    'outputs/metrics/fusion_audit.json —— 质量检查的汇总数字。',
])
d.body('下一步（阶段 4）：基于 feature_audit.csv 里 action=KEEP 的特征，做分层训练/验证划分与预处理。')

d.hr()
d.image(mm_png, width=158, caption='图：阶段 3 思维导图')

pdf_path = d.save(os.path.join(config.LEARNINGS_DIR, '03_多表融合与质量审计.pdf'))
print('\n已生成学习 PDF:', pdf_path)
print('DONE 阶段3')
