# -*- coding: utf-8 -*-
"""
阶段 1~2 总结：EDA 与特征工程
=============================
每张表固定四个板块，方便查阅：
  【包含的变量】 原始表有哪些字段、含义是什么
  【处理方式】   做了哪些 EDA / 特征工程步骤
  【中间的问题】 过程中发现并要处理的坑
  【最后得到的特征】 压成"一人一行"后产出了哪些特征
"""
import os
import config
import pdf_utils

d = pdf_utils.Doc('EDA 与特征工程总结（阶段 1~2）',
                  '七张表 · 每张四段：包含的变量 / 处理方式 / 中间的问题 / 最后得到的特征')

C_VAR = pdf_utils.C_PRIMARY   # 变量：深蓝
C_STEP = pdf_utils.C_ACCENT   # 处理：青绿
C_ISSUE = pdf_utils.C_WARN    # 问题：橙
C_FEAT = pdf_utils.C_PRIMARY  # 特征：深蓝

# ----------------------------------------------------------------------------
# 主线（简述）
# ----------------------------------------------------------------------------
d.h1('主线：把"一人多行"压成"一人一行"')
d.body('模型要预测的 TARGET 是一人一行（307,511 人），但历史数据都是一人多行。'
       '特征工程做的唯一一件事，就是把一个人散落在几千行里的行为翻译成一行数字。'
       '万能三板斧：<b>聚合</b>（count/sum/mean/max）、<b>比率</b>（除以机会数，灵魂是分母）、'
       '<b>时间窗口</b>（最近 3/6/12 月 vs 更早，看恶化）。'
       '判断特征好坏：把它十分位分组，看每组违约率是否单调。')
d.hr()

# ============================================================================
# ① application_train
# ============================================================================
d.h1('① application_train（申请主表）· 307,511 行 × 122 列 · 粒度 = 人')
d.label('【包含的变量】', C_VAR)
d.table([
    ['字段（组）', '含义'],
    ['SK_ID_CURR / TARGET', '客户 ID / 违约标签（1=逾期困难，0=其他），全项目唯一的答案'],
    ['AMT_INCOME_TOTAL / AMT_CREDIT / AMT_ANNUITY / AMT_GOODS_PRICE', '收入 / 贷款额 / 年金 / 商品价'],
    ['CODE_GENDER / CNT_CHILDREN / CNT_FAM_MEMBERS', '性别 / 子女数 / 家庭人口'],
    ['NAME_EDUCATION_TYPE / NAME_FAMILY_STATUS / NAME_INCOME_TYPE / NAME_HOUSING_TYPE / OCCUPATION_TYPE', '学历 / 婚姻 / 收入类型 / 住房 / 职业（类别变量）'],
    ['DAYS_BIRTH / DAYS_EMPLOYED / DAYS_REGISTRATION / DAYS_ID_PUBLISH', '年龄 / 工龄 / 登记时长 / 证件时长（相对申请日的天数，负数=过去）'],
    ['EXT_SOURCE_1 / 2 / 3', '外部征信评分（归一化），最关键的三列'],
    ['APARTMENTS_AVG/MODE/MEDI 等 40+ 列', '居住建筑/区域的归一化信息（面积、房龄、电梯、楼层…）'],
    ['AMT_REQ_CREDIT_BUREAU_HOUR/DAY/WEEK/MON/QRT/YEAR', '申请前各时间窗内被查询征信的次数'],
    ['OBS/DEF_30/60_CNT_SOCIAL_CIRCLE', '社交圈中逾期的人数'],
    ['FLAG_DOCUMENT_2~21 / FLAG_MOBIL / FLAG_EMP_PHONE / FLAG_EMAIL', '是否提供了各类证件 / 手机 / 公司电话 / 邮箱'],
], col_widths=[84, 90])
d.label('【处理方式】', C_STEP)
d.numbered([
    '基础 EDA：查 307,511 行、122 列，确认 SK_ID_CURR 唯一；统计 TARGET 分布（违约率 8.07%）和缺失率（41 个字段缺失 >50%）。',
    '单变量风险分层：对 EXT_SOURCE_1/2/3 十等分，看每组违约率；再算单变量 AUC。',
    '缺失关联分析：对比"某字段缺失 vs 不缺失"两组的违约率（Lift）。',
])
d.label('【中间的问题】', C_ISSUE)
d.warnlist([
    '类别不平衡：违约率仅 8.07%，决定后面不能用准确率当指标（全猜"不违约"也有 92%）。',
    '缺失本身是弱信号：缺失客户违约率更高（Lift 1.05~1.06），但差距仅约 1 个百分点，不是决定性因子。',
    '41 个字段缺失 >50%，处理时要区分"无记录 / 无有效分母 / 真实 0 / 信息未知"。',
])
d.label('【最后得到的特征】', C_FEAT)
d.body('主表本来就是"一人一行"，<b>120 个原始字段直接进模型</b>，不额外做聚合（除 ID 和 TARGET 外全部保留）。'
       '真正有区分力的代表：EXT_SOURCE_1/2/3（单变量 AUC 0.666 / 0.656 / 0.679，覆盖率 43.6% / 99.8% / 80.2%）。')

# ============================================================================
# ② previous_application
# ============================================================================
d.h1('② previous_application（历史贷款申请）· 167 万行 · 粒度 = 申请事件')
d.label('【包含的变量】', C_VAR)
d.table([
    ['字段（组）', '含义'],
    ['SK_ID_PREV / SK_ID_CURR', '历史申请 ID / 客户 ID'],
    ['NAME_CONTRACT_STATUS', '审批结果：Approved 批 / Refused 拒 / Canceled 取消 / Unused offer 未用'],
    ['AMT_APPLICATION / AMT_CREDIT', '申请金额 / 最终授信金额（可能被银行砍额度）'],
    ['AMT_DOWN_PAYMENT / AMT_ANNUITY / AMT_GOODS_PRICE', '首付 / 年金 / 商品价'],
    ['DAYS_DECISION', '审批日（相对申请日的天数，负数=过去）'],
    ['CODE_REJECT_REASON / NAME_CONTRACT_TYPE / NAME_PORTFOLIO / CHANNEL_TYPE', '拒绝原因 / 合同类型 / 产品线 / 获客渠道'],
    ['CNT_PAYMENT / DAYS_FIRST_DRAWING / DAYS_LAST_DUE 等', '期数 / 首次放款日 / 最后到期日等'],
], col_widths=[84, 90])
d.label('【处理方式】', C_STEP)
d.numbered([
    '审批状态打标：IS_APPROVED / IS_REFUSED / IS_CANCELED 三个独立 0/1（不合并）。',
    '时间窗口打标：APP_D090（近90天）、APP_PREV090（前90天）、APP_D365（近365天）。',
    '聚合到客户级：历史申请次数、获批/拒绝/取消次数、各窗口申请与拒绝次数、距最近申请天数。',
    '算比率：全历史拒绝率、近90天拒绝率、前90天拒绝率、两者之差（趋势）。',
    '金额特征：正申请金额均值/最大、获批/被拒申请金额均值、授信申请比、授信<申请的比例。',
])
d.label('【中间的问题】', C_ISSUE)
d.warnlist([
    '分母为 0：近 90 天没申请时，拒绝率 0÷0 要标成缺失（replace 0 → NaN），不能错算成 0。',
    'Canceled ≠ Refused：取消申请（客户自己放弃，金额常为 0）和被拒绝（银行不信任）业务相反，绝不合并。',
    '授信申请比：授信 < 申请（比值 <0.99）说明银行砍了额度，本身就是风险信号。',
    '趋势特征覆盖极低：PREV_REFUSED_RATE_DELTA_90V90 要"两个窗口都申请过"才能算，缺失 96.35%。',
])
d.label('【最后得到的特征】（20 个）', C_FEAT)
d.table([
    ['特征', '含义'],
    ['PREV_APP_COUNT / APPROVED_COUNT / REFUSED_COUNT / CANCELED_COUNT', '历史申请总次数 / 获批 / 被拒 / 取消次数'],
    ['PREV_APP_D090 / PREV090 / D365', '近90天 / 前90天 / 近365天申请次数'],
    ['PREV_REFUSED_D090 / PREV090', '近90天 / 前90天被拒次数'],
    ['PREV_REFUSED_RATE / RATE_D090 / RATE_PREV090', '全历史拒绝率 / 近90天拒绝率 / 前90天拒绝率'],
    ['PREV_REFUSED_RATE_DELTA_90V90', '近90天 vs 前90天拒绝率变化（趋势）'],
    ['PREV_DAYS_SINCE_LAST_APP', '距最近一次历史申请的天数'],
    ['PREV_APP_AMT_MEAN / MAX', '历史申请金额均值 / 最大'],
    ['PREV_APPROVED_AMT_MEAN / REFUSED_AMT_MEAN', '获批申请金额均值 / 被拒申请金额均值'],
    ['PREV_APPROVED_CREDIT_RATIO_MEAN', '授信申请比均值'],
    ['PREV_APPROVED_CREDIT_LT_APP_RATE', '授信 < 申请（被砍额度）的比例'],
], col_widths=[96, 78])

# ============================================================================
# ③ bureau
# ============================================================================
d.h1('③ bureau（外部征信主表）· 171 万行 · 粒度 = 账户')
d.label('【包含的变量】', C_VAR)
d.table([
    ['字段（组）', '含义'],
    ['SK_ID_CURR / SK_BUREAU_ID', '客户 ID / 征信账户 ID（一个客户可有多个账户）'],
    ['CREDIT_ACTIVE', '账户状态：Active（未结清）/ Closed（已结清）等'],
    ['CREDIT_TYPE / CREDIT_CURRENCY', '信贷类型（消费贷/信用卡/车贷…）/ 币种'],
    ['DAYS_CREDIT / DAYS_CREDIT_ENDDATE / DAYS_ENDDATE_FACT / DAYS_CREDIT_UPDATE', '开户日 / 计划到期日 / 实际结束日 / 信息更新时间（相对申请日）'],
    ['CREDIT_DAY_OVERDUE / AMT_CREDIT_MAX_OVERDUE', '当前逾期天数 / 历史最大逾期金额'],
    ['AMT_CREDIT_SUM / AMT_CREDIT_SUM_DEBT / AMT_CREDIT_SUM_LIMIT / AMT_CREDIT_SUM_OVERDUE', '授信额 / 当前欠款 / 额度 / 逾期金额'],
    ['CNT_CREDIT_PROLONG / AMT_ANNUITY', '展期次数 / 年金'],
], col_widths=[84, 90])
d.label('【处理方式】', C_STEP)
d.numbered([
    '状态打标：IS_ACTIVE / IS_CLOSED 两个 0/1，分开统计。',
    '债务异常打标：DEBT_NEGATIVE（欠款为负）、ACTIVE_DEBT（活跃账户欠款，负值 clip 到 0）、ACTIVE_DEBT_MISSING（活跃账户欠款缺失）。',
    '聚合到客户级：账户总数、活跃/关闭账户数、活跃债务合计/最大、负债务账户数。',
    '算比率：活跃账户占比、活跃账户债务缺失率。',
    '时间窗口（先排除 DAYS_CREDIT_UPDATE>0 的记录）：近90/前90/近365天新开账户数、变化量、距最近信贷天数。',
    '静态补充：当前逾期账户数、最大当前逾期天数、逾期金额合计、消费贷/信用卡账户数。',
])
d.label('【中间的问题】', C_ISSUE)
d.warnlist([
    'Active / Closed 风险天差地别，不能把"账户数"一锅端。',
    '债务负值不能直接加进总欠款（会抵消正欠款），要 clip(lower=0) 再求和，负值单独当异常标记。',
    '"无记录 ≠ 欠款为 0"：所有活跃账户欠款都缺失的客户，总欠款设 NaN，不能填 0。',
    '泄漏苗头：DAYS_CREDIT_UPDATE > 0（更新时间在申请之后）的 17 条记录要排除——申请时点不可能知道未来的更新。',
])
d.label('【最后得到的特征】（bureau 19 + bureau_balance 8 = 27 个）', C_FEAT)
d.table([
    ['特征', '含义'],
    ['BUR_CREDIT_COUNT / ACTIVE_COUNT / CLOSED_COUNT', '征信账户总数 / 活跃账户数 / 已关闭账户数'],
    ['BUR_ACTIVE_DEBT_SUM / MAX', '活跃账户欠款合计 / 最大'],
    ['BUR_ACTIVE_RATIO', '活跃账户占比'],
    ['BUR_ACTIVE_DEBT_MISSING_RATE / NEGATIVE_DEBT_COUNT', '活跃债务缺失率 / 负债务账户数'],
    ['BUR_CREDIT_NEW_D090 / PREV090 / D365', '近90天 / 前90天 / 近365天新开账户数'],
    ['BUR_CREDIT_NEW_DELTA_90V90 / DAYS_SINCE_LAST_CREDIT', '新开账户数变化 / 距最近信贷天数'],
    ['BUR_CURRENT_OVERDUE_COUNT / MAX_CURRENT_DPD / OVERDUE_AMT_SUM', '当前逾期账户数 / 最大逾期天数 / 逾期金额合计'],
    ['BUR_CONSUMER_COUNT / CREDIT_CARD_COUNT', '消费贷账户数 / 信用卡账户数'],
    ['BUR_BB_RECORD_COUNT / EVER_OVERDUE_COUNT / MAX_STATUS', '有月度历史账户数 / 曾逾期账户数 / 最大逾期等级'],
    ['BUR_BB_OVERDUE_RATE_MEAN / WORSEN_COUNT / DELTA_MEAN', '逾期比例均值 / 近期恶化账户数 / 趋势变化均值'],
], col_widths=[96, 78])

# ============================================================================
# ④ bureau_balance
# ============================================================================
d.h1('④ bureau_balance（征信月度状态）· 2730 万行 · 粒度 = 账户 × 月')
d.label('【包含的变量】', C_VAR)
d.table([
    ['字段', '含义'],
    ['SK_BUREAU_ID', '征信账户 ID（靠它才能把"月"归到"人"）'],
    ['MONTHS_BALANCE', '月份（相对申请日，-1=最近一个月）'],
    ['STATUS', '当月状态：0=正常，1~5=逾期 1~5 个月（越大越严重），C=关闭，X=未知'],
], col_widths=[60, 114])
d.label('【处理方式】', C_STEP)
d.numbered([
    '状态打标：IS_KNOWN（只把 0~5 当"明确状态"，X/C 排除）、IS_OVERDUE（1~5 逾期）、STATUS_NUM（把 X/C 转成 NaN）。',
    '第一级聚合（月 → 账户）：观察月数、明确状态月数、逾期月数、最大逾期等级、逾期比例、是否曾逾期。',
    '时间窗口（近6月 vs 前6月）：各窗口逾期比例，两者之差；只对"两个窗口都有≥2个明确状态月"的账户算趋势。',
    '第二级聚合（账户 → 人）：有月度历史的账户数、曾逾期账户数、最大逾期等级、逾期比例均值、近期恶化账户数、趋势均值。',
])
d.label('【中间的问题】', C_ISSUE)
d.warnlist([
    'X（未知）和 C（关闭）都不是"正常还款 0"，只把 0~5 当"明确状态"做分母，否则逾期比例被稀释。',
    '两级聚合不能跳级：月份只知道自己属于哪个账户，必须靠 bureau 的 SK_BUREAU_ID→SK_ID_CURR 映射把账户归到人。',
    '171,269 个客户没有任何月度历史：计数特征填 0，但"最大逾期等级/逾期比例"保留 NaN（无记录 ≠ 0）。',
    '趋势特征 BUR_BB_DELTA_MEAN 缺失约 76%（要双窗口都够月数）。',
])
d.label('【最后得到的特征】（已并入 bureau，见上表 BUR_BB_* 8 个）', C_FEAT)
d.body('验证结果：违约率从"有历史未逾期 7.22%" → "曾逾期未恶化 9.41%" → "曾逾期且恶化 12.03%"，'
       '单调上升，证明两级特征造对了。')

# ============================================================================
# ⑤ POS_CASH_balance
# ============================================================================
d.h1('⑤ POS_CASH_balance（内部贷款月度逾期）· 1000 万行 · 粒度 = 贷款 × 月')
d.label('【包含的变量】', C_VAR)
d.table([
    ['字段', '含义'],
    ['SK_ID_PREV / SK_ID_CURR', '历史贷款 ID / 客户 ID'],
    ['MONTHS_BALANCE', '月份（相对申请日）'],
    ['CNT_INSTALMENT / CNT_INSTALMENT_FUTURE', '总期数 / 剩余期数'],
    ['NAME_CONTRACT_STATUS', '合同状态（Active/Completed 等）'],
    ['SK_DPD / SK_DPD_DEF', '当月逾期天数 / 容忍口径逾期天数（小额逾期不计）'],
], col_widths=[60, 114])
d.label('【处理方式】', C_STEP)
d.numbered([
    '先解决粒度：一个人同月可能有多笔贷款，先聚合成"客户 × 月"的最大 DPD，避免同月重复计数。',
    '静态特征：历史最大/平均 DPD、逾期比例、≥30天逾期比例、观察月数。',
    '时间窗口：近 3/6/12 月的 DPD 均值、最大值、逾期比例、≥30天比例。',
    '趋势：近3月 vs 前3月，造原始 DPD 趋势、90天截尾 DPD 趋势、逾期频率趋势三个。',
    '连续逾期：从月份序列里找"最长连续逾期"、"当前连续逾期"、"恢复次数"。',
    '严重逾期新近度：距最近一次 >90 天逾期有多少个月。',
])
d.label('【中间的问题】', C_ISSUE)
d.warnlist([
    '极端 DPD：原始值可达几百天，一个值就带偏均值/趋势，要 90 天截尾并同时保留原始/截尾两版。',
    '两种 DPD 口径（SK_DPD vs SK_DPD_DEF）不同，不能互相替代。',
    '连续逾期要识别"真正相邻的日历月份"才算连续，不能只看排序相邻。',
])
d.label('【最后得到的特征】（27 个）', C_FEAT)
d.table([
    ['特征', '含义'],
    ['POS_DPD_MAX_ALL / MEAN_ALL / RATE_ALL / DPD30_RATE_ALL / OBS_MONTHS_ALL', '历史最大/平均 DPD、逾期比例、≥30天比例、观察月数'],
    ['POS_DPD_MEAN/MAX/RATE/DPD30_RATE_M03·M06·M12', '近 3/6/12 月的 DPD 均值/最大/逾期比例/≥30天比例'],
    ['POS_DPD_MEAN_DELTA_3V3', '近3月 vs 前3月 DPD 均值变化（趋势）'],
    ['POS_DPD_CAP90_DELTA_3V3', '90天截尾 DPD 趋势'],
    ['POS_OVERDUE_RATE_DELTA_3V3', '逾期频率趋势'],
    ['POS_OVERDUE_STREAK_MAX / CURRENT', '最长连续逾期 / 当前连续逾期月数'],
    ['POS_DPD_RECOVERY_COUNT', '从逾期恢复正常的次数'],
    ['POS_MONTHS_SINCE_LAST_DPD90', '距最近一次 >90 天逾期的月数'],
], col_widths=[96, 78])

# ============================================================================
# ⑥ credit_card_balance
# ============================================================================
d.h1('⑥ credit_card_balance（信用卡月度表现）· 384 万行 · 粒度 = 卡 × 月')
d.label('【包含的变量】', C_VAR)
d.table([
    ['字段（组）', '含义'],
    ['SK_ID_PREV / SK_ID_CURR / MONTHS_BALANCE', '信用卡 ID / 客户 ID / 月份'],
    ['AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL', '当月余额 / 实际信用额度（核心两列）'],
    ['AMT_DRAWINGS_ATM/CURRENT/OTHER/POS_CURRENT + CNT_DRAWINGS_*', '当月各类取现/消费金额与次数'],
    ['AMT_INST_MIN_REGULARITY / AMT_PAYMENT_TOTAL_CURRENT', '最低应还额 / 实际总付款'],
    ['AMT_RECEIVABLE_PRINCIPAL / AMT_RECIVABLE / AMT_TOTAL_RECEIVABLE', '应收本金 / 应收 / 总应收'],
    ['CNT_INSTALMENT_MATURE_CUM / NAME_CONTRACT_STATUS', '已还期数 / 合同状态'],
    ['SK_DPD / SK_DPD_DEF', '当月逾期天数 / 容忍口径逾期天数'],
], col_widths=[84, 90])
d.label('【处理方式】', C_STEP)
d.numbered([
    '额度使用率：UTIL_RAW = 余额 ÷ 额度（只对额度>0 的月份）；UTIL_CAP1 = 截尾到 0~1。',
    '异常打标：ZERO_LIMIT（额度=0）、OVER_LIMIT_VALID（额度>0 且余额>额度）、NEGATIVE_BALANCE（余额<0）。',
    '聚合：观察月数、账户数、使用率均值/最大、截尾使用率均值、超限比例、零额度比例、负余额次数、余额均值/最大。',
    '趋势：近3月 vs 前3月使用率变化（CC_UTIL_DELTA_3V3）。',
    '还款与逾期：最低应还覆盖率、付款不足比例、逾期比例、≥30天比例、最大 DPD、DPD_DEF 比例、截尾 DPD 均值、DPD 趋势。',
])
d.label('【中间的问题】', C_ISSUE)
d.warnlist([
    '零额度陷阱：额度=0 时"余额 ÷ 0"是无穷大，必须排除在分母外，使用率设 NaN。',
    '超限 ≠ 零额度：额度>0 且余额>额度（真刷爆）和额度=0 是两回事，分别造特征。',
    '截尾丢信息：使用率 clip 到 0~1 后"超限了多少"就没了，要同时保留原始/截尾/超限比例三个。',
    '最低应还覆盖率 ≠ 逾期率：付款和账单统计周期不同；SK_DPD vs SK_DPD_DEF 口径不同。',
])
d.label('【最后得到的特征】（31 个）', C_FEAT)
d.table([
    ['特征', '含义'],
    ['CC_OBS_MONTHS / ACCOUNT_COUNT', '观察月数 / 信用卡账户数'],
    ['CC_UTIL_MEAN_ALL / MAX_ALL / CAP1_MEAN_ALL', '原始使用率均值/最大 / 截尾使用率均值'],
    ['CC_OVERLIMIT_RATE_ALL / ZERO_LIMIT_RATE_ALL', '超限比例 / 零额度比例'],
    ['CC_BALANCE_MEAN_ALL / MAX_ALL', '余额均值 / 最大'],
    ['CC_UTIL_DELTA_3V3', '近3月 vs 前3月使用率变化'],
    ['CC_MIN_PAY_COVERAGE_MEAN / MIN_PAY_SHORT_RATE', '最低应还覆盖率均值 / 付款不足比例'],
    ['CC_DPD_RATE_ALL / DPD30_RATE_ALL / DPD_MAX_ALL / DPD_DEF_RATE_ALL', '逾期比例 / ≥30天比例 / 最大DPD / DPD_DEF比例'],
    ['CC_DPD_CAP90_MEAN / DPD_DELTA_3V3', '90天截尾DPD均值 / DPD趋势'],
], col_widths=[96, 78])

# ============================================================================
# ⑦ installments_payments
# ============================================================================
d.h1('⑦ installments_payments（历史分期付款）· 1360 万行 · 粒度 = 付款事件')
d.label('【包含的变量】', C_VAR)
d.table([
    ['字段', '含义'],
    ['SK_ID_PREV / SK_ID_CURR', '历史贷款 ID / 客户 ID'],
    ['NUM_INSTALMENT_VERSION / NUM_INSTALMENT_NUMBER', '分期版本 / 期次（两者一起唯一定位一期）'],
    ['DAYS_INSTALMENT', '应还日（相对申请日）'],
    ['DAYS_ENTRY_PAYMENT', '实际付款日（相对申请日）'],
    ['AMT_INSTALMENT / AMT_PAYMENT', '应还金额 / 实际付款金额'],
], col_widths=[70, 104])
d.label('【处理方式】', C_STEP)
d.numbered([
    '分期重构：按分期（贷款+版本+期次）分组、按付款日期排序、累计求和，找"累计付款第一次达到应还金额"的日期=足额还款日（金额比较用 0.01 容差）。',
    '足额付款延迟 = 足额还款日 - 应还日（负数提前、0 按时、正数延期）。',
    '三阈值：找达到 95% / 99% / 100% 的日期，分别算 PAY95_DELAY / PAY99_DELAY / FULL_PAY_DELAY。',
    '到期覆盖率：到期日（含）前累计付款 ÷ 应还金额，截尾到 0~1。',
    '聚合到客户级：分期数、延期比例、正向延迟均值/最大、≥30天比例、到期覆盖率均值、异常付款比例等。',
    '趋势：近90天 vs 前90天覆盖率变化（INS_COVERAGE_DELTA_90V90）。',
])
d.label('【中间的问题】', C_ISSUE)
d.warnlist([
    '小额尾款坑：还了 99.9 块、差 1 毛尾款拖几个月，"足额付款延迟"会飙到几百天但风险很低，'
    '所以造 95/99/100% 三阈值把"只差尾款"和"从头没还"分开。',
    '幸存者偏差：只统计"最终还清"的分期，会漏掉最危险的"还不起"的人。',
    '付款缺失 ≠ 未付款；观测截止日（到期才 5 天的分期不能和 90 天的比 30 天以上延期）。',
    '提前还款不算延迟（延迟计 0，不是负数）。',
])
d.label('【最后得到的特征】（24 个）', C_FEAT)
d.table([
    ['特征', '含义'],
    ['INS_INSTALLMENT_COUNT', '历史分期总数'],
    ['INS_LATE_RATE_ALL / DPD30_RATE_ALL', '延期比例 / 延期>30天比例'],
    ['INS_DPD_MEAN_ALL / MAX_ALL', '正向延迟均值 / 最大'],
    ['INS_LATE_DAYS_AVG_ALL / CAP90_AVG', '平均延迟天数 / 90天截尾平均延迟'],
    ['INS_DUE_COVERAGE_MEAN', '到期日覆盖率均值'],
    ['INS_PAYMENT_ANOMALY_RATE / MISSING_PAYMENT_COUNT', '付款异常比例 / 付款缺失数'],
    ['INS_UNCONFIRMED_COUNT / D090_COUNT', '未确认足额分期数 / 近90天未确认数'],
    ['INS_UNPAID30_COUNT_ALL / D180', '到期30天仍未足额数（全历史 / 近180天）'],
    ['INS_COVERAGE_DELTA_90V90', '近90天 vs 前90天覆盖率变化'],
], col_widths=[96, 78])
d.why('这张表埋了全项目最大的一颗雷——T0 泄漏：那些"付款结果"字段（后来到底还没还、还多少）'
      '很多发生在申请日之后，拿它预测违约就是用未来预测过去。所以阶段 3 大审计里'
      '20 个 installments"付款结果"特征被直接排除，这也是消融实验里 installments 贡献 ≈ 0 的原因'
      '——不是这张表没用，是它最值钱的"结果"特征被防泄漏原则拿掉了。')

# ----------------------------------------------------------------------------
# 全景回顾
# ----------------------------------------------------------------------------
d.hr()
d.h1('全景回顾：七张表，一条主线')
d.table([
    ['数据表', '粒度', '核心字段', '独门坑'],
    ['application_train', '人', 'EXT_SOURCE_1/2/3、收入、年龄', '缺失是弱信号、类别不平衡'],
    ['previous_application', '申请事件', '审批结果、申请/授信金额、DAYS_DECISION', '分母为0、Canceled≠Refused、趋势96%缺失'],
    ['bureau', '账户', '账户状态、欠款、DAYS_CREDIT_UPDATE', 'Active/Closed区分、债务负值、无记录≠0'],
    ['bureau_balance', '账户×月', 'STATUS(0~5/C/X)', '两级聚合、X/C语义、无月度历史'],
    ['POS_CASH', '贷款×月', 'SK_DPD', '客户-月去重、连续逾期、DPD截尾'],
    ['credit_card', '卡×月', '余额、额度、SK_DPD', '零额度陷阱、超限≠零额度、截尾丢信息'],
    ['installments', '付款事件', '应还/实付日、应还/实付金额', '事件重构、小额尾款、幸存者偏差、T0泄漏'],
], col_widths=[32, 24, 66, 52])
d.body('')
d.h2('产出与去向')
d.bullets([
    '五份历史模块合计 <b>129 个客户级特征</b>：POS 27、Installments 24、Bureau 27、Previous 20、Credit Card 31。',
    '模型输入 = <b>120 个主表原始字段</b>（主表本来就是一人一行）+ <b>129 个历史工程特征</b> = 249，'
    '其中 20 个 installments 泄漏特征在阶段 3 被排除，剩 229 进入建模。',
    '阶段 3~8 做的事，本质就是把这 129 个特征"验货"：哪些真有用、哪些泄漏、每张表贡献多大。',
])

pdf_path = d.save(os.path.join(config.LEARNINGS_DIR, '01-02_EDA与特征工程总结.pdf'))
print('已生成学习 PDF:', pdf_path)
print('DONE')
