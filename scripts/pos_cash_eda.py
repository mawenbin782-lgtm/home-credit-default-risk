import pandas as pd

# 读取 POS/CASH 历史月度数据
pos = pd.read_csv(r"D:\Home Credit Default Risk\data\POS_CASH_balance.csv")

# 1. 数据规模
print("数据维度：", pos.shape)

# 2. 字段名称
print("\n字段名称：")
print(pos.columns.tolist())

# 3. 前5行
print("\n前5行数据：")
print(pos.head().to_string())

# 4. 客户数量
print("\n客户数量：", pos["SK_ID_CURR"].nunique())

# 5. 历史贷款数量
print("历史贷款数量：", pos["SK_ID_PREV"].nunique())

# 6. 检查贷款-月份重复记录
duplicates = pos.duplicated(
    subset=["SK_ID_PREV", "MONTHS_BALANCE"]
).sum()

print("贷款-月份重复记录数：", duplicates)

# 7. 月份范围
print("最早历史月份：", pos["MONTHS_BALANCE"].min())
print("最近历史月份：", pos["MONTHS_BALANCE"].max())

# ==========================================
# EDA 02：历史月度记录覆盖分析
# ==========================================

# 1. 转为客户-月份粒度
# 同一个客户在同一个月可能有多笔历史贷款
customer_month = (
    pos.groupby(
        ["SK_ID_CURR", "MONTHS_BALANCE"],
        as_index=False
    )
    .agg(
        DPD_MAX=("SK_DPD", "max"),
        LOAN_COUNT=("SK_ID_PREV", "nunique")
    )
)

print("\n客户-月份记录数：", len(customer_month))

# 2. 每位客户的有效历史月份数
history_months = (
    customer_month.groupby("SK_ID_CURR")
    ["MONTHS_BALANCE"]
    .nunique()
)

print("\n每位客户的历史月份数分布：")
print(history_months.describe())

# 3. 检查近期两个窗口的覆盖情况
customer_month["recent_3"] = (
    customer_month["MONTHS_BALANCE"].between(-3, -1)
)

customer_month["previous_3"] = (
    customer_month["MONTHS_BALANCE"].between(-6, -4)
)

coverage = (
    customer_month.groupby("SK_ID_CURR")
    [["recent_3", "previous_3"]]
    .sum()
)

# 每个窗口至少有2个月记录
eligible = (
    (coverage["recent_3"] >= 2) &
    (coverage["previous_3"] >= 2)
)

print("\n近3个月至少有2个月记录的客户数：", (coverage["recent_3"] >= 2).sum())

print("前3个月至少有2个月记录的客户数：", (coverage["previous_3"] >= 2).sum())

print("两个窗口都满足条件的客户数：", eligible.sum())

print("两个窗口都满足条件的客户比例：", f"{eligible.mean():.2%}")

# ==========================================
# EDA 03：历史覆盖情况与 TARGET 的关系
# ==========================================

application = pd.read_csv(
    r"D:\Home Credit Default Risk\data\application_train.csv",
    usecols=["SK_ID_CURR", "TARGET"]
)

# 保留客户的观察窗口覆盖信息
coverage_features = coverage.rename(
    columns={
        "recent_3": "RECENT_3_OBS",
        "previous_3": "PREVIOUS_3_OBS"
    }
).reset_index()

# 左连接，确保申请主表客户不丢失
merged = application.merge(
    coverage_features,
    on="SK_ID_CURR",
    how="left",
    validate="one_to_one"
)

# 无POS记录的客户，窗口观察数设为0
merged[["RECENT_3_OBS", "PREVIOUS_3_OBS"]] = (
    merged[["RECENT_3_OBS", "PREVIOUS_3_OBS"]]
    .fillna(0)
)

# 两个窗口是否都满足至少2个月
merged["TREND_ELIGIBLE"] = (
    (merged["RECENT_3_OBS"] >= 2) &
    (merged["PREVIOUS_3_OBS"] >= 2)
)

print("合并后样本数：", len(merged))

print("\n趋势特征可计算情况：")
print(merged["TREND_ELIGIBLE"].value_counts())

print("\n不同覆盖客群的坏样本率：")
print(
    merged.groupby("TREND_ELIGIBLE")["TARGET"]
    .agg(
        样本数="count",
        坏样本数="sum",
        坏样本率="mean"
    ).to_string()
)

# ==========================================
# EDA 04：POS/CASH 逾期分布
# ==========================================

print("\n===== SK_DPD 描述统计 =====")
print(pos["SK_DPD"].describe(percentiles=[0.5, 0.9, 0.95, 0.99, 0.999]))

print("\n===== SK_DPD_DEF 描述统计 =====")
print(pos["SK_DPD_DEF"].describe(percentiles=[0.5, 0.9, 0.95, 0.99, 0.999]))

# 将月度记录按照逾期程度分组
dpd_bins = [-float("inf"), 0, 7, 30, 60, 90, float("inf")]

dpd_labels = [
    "DPD=0",
    "1-7天",
    "8-30天",
    "31-60天",
    "61-90天",
    ">90天"
]

dpd_group = pd.cut(
    pos["SK_DPD"],
    bins=dpd_bins,
    labels=dpd_labels
)

print("\n===== 月度记录逾期分布 =====")

dpd_distribution = (
    dpd_group.value_counts(sort=False)
    .to_frame("记录数")
)

dpd_distribution["占比"] = (
    dpd_distribution["记录数"] / len(pos) * 100
)

print(dpd_distribution.round(2).to_string())

# 比较两种DPD口径
print("\n===== 两种DPD口径比较 =====")
print("SK_DPD > 0 的记录数：", (pos["SK_DPD"] > 0).sum())

print("SK_DPD_DEF > 0 的记录数：", (pos["SK_DPD_DEF"] > 0).sum())

print("两种口径不同的记录数：", (pos["SK_DPD"] != pos["SK_DPD_DEF"]).sum())

# EDA 05：DPD 与历史贷款状态的关系

pos["DPD_GROUP"] = pd.cut(
    pos["SK_DPD"],
    bins=[-1, 0, 7, 30, 60, 90, float("inf")],
    labels=[
        "0",
        "1-7",
        "8-30",
        "31-60",
        "61-90",
        ">90"
    ]
)

# 各贷款状态的样本量和逾期比例
status_analysis = (
    pos.groupby("NAME_CONTRACT_STATUS")
    .agg(
        记录数=("SK_DPD", "size"),
        平均DPD=("SK_DPD", "mean"),
        最大DPD=("SK_DPD", "max"),
        逾期比例=("SK_DPD", lambda x: (x > 0).mean()),
        严重逾期比例=("SK_DPD", lambda x: (x > 90).mean())
    )
)

print("\n===== 不同贷款状态的DPD表现 =====")
print(status_analysis.round(4).to_string())

# 检查极端DPD记录集中在哪些状态
print("\n===== DPD>90 的贷款状态分布 =====")
print(pos.loc[pos["SK_DPD"] > 90, "NAME_CONTRACT_STATUS"].value_counts())

# ==========================================
# EDA 06：不同历史时间窗口的 DPD 分布
# ==========================================

pos["TIME_GROUP"] = pd.cut(
    pos["MONTHS_BALANCE"],
    bins=[-97, -37, -13, -7, -4, -1],
    labels=[
        "37-96个月前",
        "13-36个月前",
        "7-12个月前",
        "4-6个月前",
        "1-3个月前"
    ]
)

time_analysis = (
    pos.groupby("TIME_GROUP", observed=True)
    .agg(
        记录数=("SK_DPD", "size"),
        平均DPD=("SK_DPD", "mean"),
        逾期比例=("SK_DPD", lambda x: (x > 0).mean()),
        严重逾期比例=("SK_DPD", lambda x: (x > 90).mean()),
        最大DPD=("SK_DPD", "max")
    )
)

print("\n===== 不同历史窗口的逾期表现 =====")
print(time_analysis.round(4).to_string())

# ==========================================
# Feature Engineering 01：历史风险变化
# ==========================================

# 1. 选取近6个月的客户月度记录
recent_history = customer_month[
    customer_month["MONTHS_BALANCE"].between(-6, -1)
].copy()

# 2. 划分两个观察窗口
recent_history["WINDOW"] = "PREV03"

recent_history.loc[
    recent_history["MONTHS_BALANCE"].between(-3, -1),
    "WINDOW"
] = "RECENT03"

# 3. 计算每位客户在各窗口的平均DPD及观察月数
window_features = (
    recent_history.groupby(["SK_ID_CURR", "WINDOW"])
    .agg(
        DPD_MEAN=("DPD_MAX", "mean"),
        OBS_MONTHS=("MONTHS_BALANCE", "nunique")
    )
    .unstack("WINDOW")
)

# 4. 展平列名
window_features.columns = [
    f"{metric}_{window}"
    for metric, window in window_features.columns
]

window_features = window_features.reset_index()

# 5. 判断是否满足最低观察要求
eligible = (
    (window_features["OBS_MONTHS_RECENT03"] >= 2) &
    (window_features["OBS_MONTHS_PREV03"] >= 2)
)

# 6. 构造风险变化特征
window_features["POS_DPD_MEAN_DELTA_3V3"] = (
    window_features["DPD_MEAN_RECENT03"] -
    window_features["DPD_MEAN_PREV03"]
).where(eligible)

# 7. 特征质量检查
print("\n===== 风险趋势特征描述统计 =====")

print(
    window_features[
        [
            "DPD_MEAN_RECENT03",
            "DPD_MEAN_PREV03",
            "POS_DPD_MEAN_DELTA_3V3"
        ]
    ].describe().to_string()
)

print("\n趋势特征有效客户数：", window_features["POS_DPD_MEAN_DELTA_3V3"].notna().sum())

print("\n前5条结果：")
print(window_features.head().to_string())

# ==========================================
# Feature Engineering 02：风险趋势方向分析
# ==========================================

trend = window_features[
    "POS_DPD_MEAN_DELTA_3V3"
].dropna()

print("\n===== 风险趋势方向分布 =====")

print("有效客户数：", len(trend))

print("风险恶化客户数：", (trend > 0).sum())
print("风险改善客户数：", (trend < 0).sum())
print("风险不变客户数：", (trend == 0).sum())

print("\n风险恶化比例：", f"{(trend > 0).mean():.2%}")
print("风险改善比例：", f"{(trend < 0).mean():.2%}")
print("风险不变比例：", f"{(trend == 0).mean():.2%}")

print("\n===== 趋势差值分位数 =====")
print(trend.quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]))

# ==========================================
# Feature Engineering 03：
# 风险趋势与 TARGET 的关系
# ==========================================

import numpy as np

# 1. 读取主表标签
application = pd.read_csv(
    r"D:\Home Credit Default Risk\data\application_train.csv",
    usecols=["SK_ID_CURR", "TARGET"]
)

# 2. 关联风险趋势特征
trend_data = application.merge(
    window_features[
        ["SK_ID_CURR",
         "DPD_MEAN_PREV03",
         "DPD_MEAN_RECENT03",
         "POS_DPD_MEAN_DELTA_3V3"]
    ],
    on="SK_ID_CURR",
    how="left",
    validate="one_to_one"
)

# 3. 风险趋势分类
delta = trend_data["POS_DPD_MEAN_DELTA_3V3"]

trend_data["TREND_GROUP"] = np.select(
    [
        delta < 0,
        delta == 0,
        delta > 0
    ],
    [
        "改善",
        "不变",
        "恶化"
    ],
    default="历史不足"
)

# 4. 比较不同群体的坏样本率
result = (
    trend_data.groupby("TREND_GROUP")["TARGET"]
    .agg(
        样本数="count",
        坏样本数="sum",
        坏样本率="mean"
    )
)

result["Lift"] = (
    result["坏样本率"] /
    trend_data["TARGET"].mean()
)

print("\n===== 风险趋势与坏样本率 =====")
print(result.round(4).to_string())

# 5. 进一步拆分零变化客户
zero_group = trend_data[
    trend_data["TREND_GROUP"] == "不变"
].copy()

zero_group["ZERO_TYPE"] = np.where(
    (zero_group["DPD_MEAN_PREV03"] == 0) &
    (zero_group["DPD_MEAN_RECENT03"] == 0),
    "持续零DPD",
    "非零但均值不变"
)

print("\n===== 零变化客户进一步分析 =====")
print(
    zero_group.groupby("ZERO_TYPE")["TARGET"]
    .agg(
        样本数="count",
        坏样本数="sum",
        坏样本率="mean"
    ).round(4).to_string()
)

# 6. 检查样本数量
print("\n合并后申请样本数：", len(trend_data))

# ==========================================
# EDA 07：近期风险水平 × 风险趋势
# ==========================================

#%%
# 只研究具备有效趋势的客户
analysis = trend_data[
    trend_data["TREND_GROUP"].isin(["改善", "不变", "恶化"])
].copy()

# 按近期平均 DPD 进行风险分层
analysis["RECENT_DPD_GROUP"] = pd.cut(
    analysis["DPD_MEAN_RECENT03"],
    bins=[-0.001, 0, 7, 30, 90, float("inf")],
    labels=[
        "DPD=0",
        "0<DPD<=7",
        "7<DPD<=30",
        "30<DPD<=90",
        "DPD>90"
    ]
)

# 交叉统计坏样本率
result = (
    analysis.groupby(
        ["RECENT_DPD_GROUP", "TREND_GROUP"],
        observed=True
    )["TARGET"]
    .agg(
        样本数="count",
        坏样本数="sum",
        坏样本率="mean"
    )
)

print("\n===== 近期风险水平 × 趋势方向 =====")
print(result.round(4).to_string())
# %%
# EDA 08：调查近期极端 DPD 客群

# 筛选近期 DPD>90 且风险恶化的客户
extreme_ids = analysis.loc[
    (analysis["DPD_MEAN_RECENT03"] > 90) &
    (analysis["TREND_GROUP"] == "恶化"),
    "SK_ID_CURR"
]

# 查看这些客户近三个月的历史贷款状态
extreme_records = pos.loc[
    pos["SK_ID_CURR"].isin(extreme_ids) &
    pos["MONTHS_BALANCE"].between(-3, -1)
].copy()

print("极端风险客户数：", extreme_ids.nunique())

print("\n===== 合同状态分布 =====")
print(extreme_records["NAME_CONTRACT_STATUS"].value_counts().to_string())

print("\n===== 不同合同状态的DPD =====")
print(
    extreme_records.groupby("NAME_CONTRACT_STATUS")["SK_DPD"]
    .agg(["count", "mean", "median", "max"])
    .round(2)
    .to_string()
)

print("\n===== 极端风险客户的DPD分位数 =====")
print(extreme_records["SK_DPD"].quantile([0, .25, .5, .75, .9, .99, 1]))
# %%
# ==========================================
# Feature Engineering 04：
# 截尾 DPD 趋势与逾期频率趋势
# ==========================================

# 1. 取最近6个月的客户月度数据
temp = customer_month[
    customer_month["MONTHS_BALANCE"].between(-6, -1)
].copy()

# 2. 构造截尾 DPD 和逾期标记
temp["DPD_CAP90"] = temp["DPD_MAX"].clip(upper=90)
temp["IS_OVERDUE"] = (temp["DPD_MAX"] > 0).astype(int)

# 3. 划分近期与前期窗口
temp["WINDOW"] = "PREV03"

temp.loc[
    temp["MONTHS_BALANCE"].between(-3, -1),
    "WINDOW"
] = "RECENT03"

# 4. 按客户、窗口聚合
features = (
    temp.groupby(["SK_ID_CURR", "WINDOW"])
    .agg(
        DPD_CAP90_MEAN=("DPD_CAP90", "mean"),
        OVERDUE_RATE=("IS_OVERDUE", "mean"),
        OBS_MONTHS=("MONTHS_BALANCE", "nunique")
    )
    .unstack("WINDOW")
)

features.columns = [
    f"{metric}_{window}"
    for metric, window in features.columns
]

features = features.reset_index()

# 5. 两个窗口均至少有2个月
eligible = (
    (features["OBS_MONTHS_RECENT03"] >= 2) &
    (features["OBS_MONTHS_PREV03"] >= 2)
)

# 6. 构造两个风险趋势指标
features["POS_DPD_CAP90_DELTA_3V3"] = (
    features["DPD_CAP90_MEAN_RECENT03"] -
    features["DPD_CAP90_MEAN_PREV03"]
).where(eligible)

features["POS_OVERDUE_RATE_DELTA_3V3"] = (
    features["OVERDUE_RATE_RECENT03"] -
    features["OVERDUE_RATE_PREV03"]
).where(eligible)

# 7. 描述统计
print("\n===== 新趋势特征描述统计 =====")

print(
    features[
        [
            "POS_DPD_CAP90_DELTA_3V3",
            "POS_OVERDUE_RATE_DELTA_3V3"
        ]
    ].describe().to_string()
)

print("\n===== 新趋势特征正负分布 =====")

for col in [
    "POS_DPD_CAP90_DELTA_3V3",
    "POS_OVERDUE_RATE_DELTA_3V3"
]:
    s = features[col].dropna()

    print(f"\n{col}")
    print("恶化：", (s > 0).sum())
    print("不变：", (s == 0).sum())
    print("改善：", (s < 0).sum())
# %%
# ==========================================
# Feature Engineering 05：
# 三种风险趋势特征的坏样本率对比
# ==========================================

import numpy as np

# 1. 将新特征合并到有标签申请主表
compare = application.merge(
    window_features[
        ["SK_ID_CURR", "POS_DPD_MEAN_DELTA_3V3"]
    ],
    on="SK_ID_CURR",
    how="left",
    validate="one_to_one"
)

compare = compare.merge(
    features[
        [
            "SK_ID_CURR",
            "POS_DPD_CAP90_DELTA_3V3",
            "POS_OVERDUE_RATE_DELTA_3V3"
        ]
    ],
    on="SK_ID_CURR",
    how="left",
    validate="one_to_one"
)

# 2. 三种趋势特征
trend_cols = [
    "POS_DPD_MEAN_DELTA_3V3",
    "POS_DPD_CAP90_DELTA_3V3",
    "POS_OVERDUE_RATE_DELTA_3V3"
]

# 3. 分别计算风险分层情况
for col in trend_cols:

    delta = compare[col]

    group = np.select(
        [
            delta < 0,
            delta == 0,
            delta > 0
        ],
        [
            "改善",
            "不变",
            "恶化"
        ],
        default="历史不足"
    )

    result = (
        compare.groupby(group)["TARGET"]
        .agg(
            样本数="count",
            坏样本数="sum",
            坏样本率="mean"
        )
    )

    result["Lift"] = (
        result["坏样本率"] /
        compare["TARGET"].mean()
    )

    print(f"\n===== {col} =====")
    print(result.round(4).to_string())

print("\n最终样本数：", len(compare))
# %%
# ==========================================
# Feature Engineering 06：
# 原始DPD与截尾DPD的差异分析
# ==========================================

import numpy as np
import pandas as pd

# 使用之前已经生成的 compare
temp = compare[
    [
        "SK_ID_CURR",
        "TARGET",
        "POS_DPD_MEAN_DELTA_3V3",
        "POS_DPD_CAP90_DELTA_3V3"
    ]
].copy()

def get_trend_group(s):
    return np.select(
        [s < 0, s == 0, s > 0],
        ["改善", "不变", "恶化"],
        default="历史不足"
    )

temp["RAW_GROUP"] = get_trend_group(
    temp["POS_DPD_MEAN_DELTA_3V3"]
)

temp["CAP90_GROUP"] = get_trend_group(
    temp["POS_DPD_CAP90_DELTA_3V3"]
)

# 交叉比较两种趋势分类
cross = pd.crosstab(
    temp["RAW_GROUP"],
    temp["CAP90_GROUP"]
)

print("\n===== 原始DPD vs 截尾DPD 趋势分类 =====")
print(cross.to_string())

# 组合分类
temp["COMBINATION"] = (
    temp["RAW_GROUP"] + " → " +
    temp["CAP90_GROUP"]
)

result = (
    temp.groupby("COMBINATION")["TARGET"]
    .agg(
        样本数="count",
        坏样本数="sum",
        坏样本率="mean"
    )
    .sort_values("样本数", ascending=False)
)

print("\n===== 不同趋势变化组合的风险表现 =====")
print(result.round(4).to_string())
# %%
# Feature Engineering 07：历史静态风险特征

static_features = (
    customer_month.groupby("SK_ID_CURR")
    .agg(
        POS_DPD_MAX_ALL=("DPD_MAX", "max"),
        POS_DPD_MEAN_ALL=("DPD_MAX", "mean"),
        POS_DPD_RATE_ALL=("DPD_MAX", lambda s: (s > 0).mean()),
        POS_DPD30_RATE_ALL=("DPD_MAX", lambda s: (s >= 30).mean()),
        POS_OBS_MONTHS_ALL=("MONTHS_BALANCE", "nunique")
    )
    .reset_index()
)

print("\n===== 静态风险特征描述统计 =====")
print(static_features.drop(columns="SK_ID_CURR").describe().round(4).to_string())

print("\n===== 静态风险特征前5行 =====")
print(static_features.head().to_string())

print("\n===== 唯一客户检查 =====")
print("客户数：", len(static_features))
print("重复客户ID：", static_features["SK_ID_CURR"].duplicated().sum())
# %%
# ==========================================
# Feature Engineering 08：
# 历史是否逾期与 TARGET 的关系
# ==========================================

# 1. 关联主表与静态风险特征
static_analysis = application.merge(
    static_features,
    on="SK_ID_CURR",
    how="left",
    validate="one_to_one"
)

# 2. 构造历史是否逾期标记
# 无历史记录的客户先保留缺失，不当作无逾期
static_analysis["POS_EVER_OVERDUE"] = (
    static_analysis["POS_DPD_MAX_ALL"] > 0
).astype("Int64")

static_analysis.loc[
    static_analysis["POS_DPD_MAX_ALL"].isna(),
    "POS_EVER_OVERDUE"
] = pd.NA

# 3. 计算各客群坏样本率
result = (
    static_analysis.groupby(
        "POS_EVER_OVERDUE",
        dropna=False
    )["TARGET"]
    .agg(
        样本数="count",
        坏样本数="sum",
        坏样本率="mean"
    )
)

result["Lift"] = (
    result["坏样本率"] /
    static_analysis["TARGET"].mean()
)

print("\n===== 历史逾期与 TARGET =====")
print(result.round(4).to_string())

print("\n合并后申请数：", len(static_analysis))
print("无 POS 历史申请数：", static_analysis["POS_DPD_MAX_ALL"].isna().sum())
# %%
# ==========================================
# EDA 09：历史逾期 × 风险趋势
# ==========================================

import numpy as np

# 在主表中合并静态与趋势特征
joint = static_analysis.merge(
    window_features[
        ["SK_ID_CURR", "POS_DPD_MEAN_DELTA_3V3"]
    ],
    on="SK_ID_CURR",
    how="left",
    validate="one_to_one"
)

delta = joint["POS_DPD_MEAN_DELTA_3V3"]

joint["TREND_GROUP"] = np.select(
    [delta < 0, delta == 0, delta > 0],
    ["改善", "不变", "恶化"],
    default="历史不足"
)

joint["HISTORY_GROUP"] = joint["POS_EVER_OVERDUE"].map({
    0: "无历史逾期",
    1: "曾发生逾期"
}).fillna("无POS记录")

result = (
    joint.groupby(
        ["HISTORY_GROUP", "TREND_GROUP"]
    )["TARGET"]
    .agg(
        样本数="count",
        坏样本数="sum",
        坏样本率="mean"
    )
)

print("\n===== 历史逾期 × 风险趋势 =====")
print(result.round(4).to_string())

print("\n合并后申请数：", len(joint))
# %%
# EDA 10：历史最大 DPD 风险分层

import pandas as pd

analysis_df = static_analysis.copy()

analysis_df["HIST_DPD_GROUP"] = pd.cut(
    analysis_df["POS_DPD_MAX_ALL"],
    bins=[-1, 0, 7, 30, 90, float("inf")],
    labels=[
        "DPD=0",
        "1-7天",
        "8-30天",
        "31-90天",
        ">90天"
    ]
)

# 无 POS 历史客户单列
analysis_df["HIST_DPD_GROUP"] = (
    analysis_df["HIST_DPD_GROUP"]
    .cat.add_categories(["无POS历史"])
    .fillna("无POS历史")
)

result = (
    analysis_df.groupby(
        "HIST_DPD_GROUP",
        observed=True
    )["TARGET"]
    .agg(
        样本数="count",
        坏样本数="sum",
        坏样本率="mean"
    )
)

result["Lift"] = (
    result["坏样本率"] /
    analysis_df["TARGET"].mean()
)

print("\n===== 历史最大DPD风险分层 =====")
print(result.round(4).to_string())

print("\n总样本数：", result["样本数"].sum())
# %%
# ==========================================
# EDA 11：严重逾期距当前申请的时间
# ==========================================

# 1. 筛选 DPD > 90 的历史客户月记录
severe = customer_month.loc[
    customer_month["DPD_MAX"] > 90,
    ["SK_ID_CURR", "MONTHS_BALANCE"]
]

# 2. 找出每位客户最近一次严重逾期的月份
last_severe = (
    severe.groupby("SK_ID_CURR")["MONTHS_BALANCE"]
    .max()
    .reset_index()
)

last_severe["MONTHS_SINCE_SEVERE"] = (
    -last_severe["MONTHS_BALANCE"]
)

# 3. 关联申请主表
severe_analysis = static_analysis.merge(
    last_severe[
        ["SK_ID_CURR", "MONTHS_SINCE_SEVERE"]
    ],
    on="SK_ID_CURR",
    how="left",
    validate="one_to_one"
)

# 4. 只分析历史最大 DPD > 90 的客户
severe_analysis = severe_analysis.loc[
    severe_analysis["POS_DPD_MAX_ALL"] > 90
].copy()

# 5. 按严重逾期距今时间分箱
severe_analysis["RECENCY_GROUP"] = pd.cut(
    severe_analysis["MONTHS_SINCE_SEVERE"],
    bins=[0, 3, 6, 12, 24, 36, float("inf")],
    labels=[
        "近1-3个月",
        "4-6个月前",
        "7-12个月前",
        "13-24个月前",
        "25-36个月前",
        "36个月以前"
    ]
)

# 6. 计算各组坏样本率
result = (
    severe_analysis.groupby(
        "RECENCY_GROUP", observed=True
    )["TARGET"]
    .agg(
        样本数="count",
        坏样本数="sum",
        坏样本率="mean"
    )
)

print("\n===== 严重逾期新近程度分析 =====")
print(result.round(4).to_string())

print("\n分析客户总数：", result["样本数"].sum())
# %%
# ==========================================
# Feature Engineering 09：
# 连续逾期与风险恢复特征
# ==========================================

import pandas as pd

# 1. 提取客户月度数据，并按时间从旧到新排序
monthly = customer_month[
    ["SK_ID_CURR", "MONTHS_BALANCE", "DPD_MAX"]
].sort_values(
    ["SK_ID_CURR", "MONTHS_BALANCE"]
).copy()

# 2. 判断当月是否逾期
monthly["IS_OVERDUE"] = monthly["DPD_MAX"] > 0

# 3. 获取同一客户的上一个观测月份及逾期状态
g = monthly.groupby("SK_ID_CURR", sort=False)

monthly["PREV_MONTH"] = g["MONTHS_BALANCE"].shift(1)
monthly["PREV_OVERDUE"] = g["IS_OVERDUE"].shift(1)

# 只有真正相邻的日历月份，才算连续
monthly["IS_CONSECUTIVE"] = (
    monthly["MONTHS_BALANCE"] - monthly["PREV_MONTH"] == 1
)

# 4. 识别每段连续逾期的起点
monthly["STREAK_START"] = (
    monthly["IS_OVERDUE"] &
    (
        ~monthly["IS_CONSECUTIVE"] |
        ~monthly["PREV_OVERDUE"].fillna(False).astype(bool)
    )
)

# 每位客户的连续逾期段编号
monthly["STREAK_ID"] = (
    monthly.groupby("SK_ID_CURR")["STREAK_START"]
    .cumsum()
)

# 5. 计算每段连续逾期的长度
overdue_rows = monthly[monthly["IS_OVERDUE"]].copy()

streak_lengths = (
    overdue_rows.groupby(["SK_ID_CURR", "STREAK_ID"])
    .size()
    .reset_index(name="STREAK_LENGTH")
)

max_streak = (
    streak_lengths.groupby("SK_ID_CURR")["STREAK_LENGTH"]
    .max()
    .rename("POS_OVERDUE_STREAK_MAX")
)

# 6. 计算截至最近观测月的连续逾期长度
latest = monthly.groupby("SK_ID_CURR").tail(1).copy()

current_streak = latest[
    ["SK_ID_CURR", "MONTHS_BALANCE", "IS_OVERDUE", "STREAK_ID"]
].merge(
    streak_lengths,
    on=["SK_ID_CURR", "STREAK_ID"],
    how="left"
)

# 必须观察到 -1 月，才认为可以计算截至最近月份的连续逾期
current_streak["POS_OVERDUE_STREAK_CURRENT"] = (
    current_streak["STREAK_LENGTH"]
    .where(
        (current_streak["MONTHS_BALANCE"] == -1) &
        current_streak["IS_OVERDUE"],
        0
    )
)

current_streak.loc[
    current_streak["MONTHS_BALANCE"] != -1,
    "POS_OVERDUE_STREAK_CURRENT"
] = float("nan")

current_streak = current_streak.set_index("SK_ID_CURR")[
    "POS_OVERDUE_STREAK_CURRENT"
]

# 7. 识别从逾期恢复正常的事件
monthly["IS_RECOVERY"] = (
    monthly["IS_CONSECUTIVE"] &
    ~monthly["IS_OVERDUE"] &
    monthly["PREV_OVERDUE"].fillna(False).astype(bool)
)

recovery = (
    monthly.groupby("SK_ID_CURR")["IS_RECOVERY"]
    .sum()
    .rename("POS_DPD_RECOVERY_COUNT")
)

# 8. 合并特征
streak_features = pd.DataFrame({
    "POS_OVERDUE_STREAK_MAX": max_streak,
    "POS_OVERDUE_STREAK_CURRENT": current_streak,
    "POS_DPD_RECOVERY_COUNT": recovery
}).reindex(customer_month["SK_ID_CURR"].unique())

# 从未逾期的客户，最长连续逾期设为0
streak_features["POS_OVERDUE_STREAK_MAX"] = (
    streak_features["POS_OVERDUE_STREAK_MAX"].fillna(0)
)

streak_features = streak_features.reset_index()
streak_features = streak_features.rename(
    columns={"index": "SK_ID_CURR"}
)

# 9. 质量检查
print("\n===== 连续逾期与风险恢复特征 =====")

print(streak_features.describe().round(2).to_string())

print("\n前5行：")
print(streak_features.head().to_string())

print("\n客户数：", len(streak_features))
print("重复客户ID：", streak_features["SK_ID_CURR"].duplicated().sum())

print("最长连续逾期超过0的客户数：", (streak_features["POS_OVERDUE_STREAK_MAX"] > 0).sum())

print("发生过恢复事件的客户数：", (streak_features["POS_DPD_RECOVERY_COUNT"] > 0).sum())
# %%
# ==========================================
# Feature Engineering 10：
# 批量构建 3/6/12 个月风险特征
# ==========================================

import pandas as pd

window_result = None

for months in [3, 6, 12]:

    # 1. 选择对应历史窗口
    temp = customer_month[
        customer_month["MONTHS_BALANCE"].between(
            -months, -1
        )
    ].copy()

    # 2. 构造逾期标记
    temp["IS_OVERDUE"] = temp["DPD_MAX"] > 0
    temp["IS_DPD30"] = temp["DPD_MAX"] >= 30

    # 3. 按客户聚合
    agg = (
        temp.groupby("SK_ID_CURR")
        .agg(
            **{
                f"POS_DPD_MEAN_M{months:02d}":
                    ("DPD_MAX", "mean"),

                f"POS_DPD_MAX_M{months:02d}":
                    ("DPD_MAX", "max"),

                f"POS_DPD_RATE_M{months:02d}":
                    ("IS_OVERDUE", "mean"),

                f"POS_DPD30_RATE_M{months:02d}":
                    ("IS_DPD30", "mean"),

                f"POS_OBS_MONTHS_M{months:02d}":
                    ("MONTHS_BALANCE", "nunique")
            }
        )
        .reset_index()
    )

    # 4. 合并不同时间窗口
    if window_result is None:
        window_result = agg
    else:
        window_result = window_result.merge(
            agg,
            on="SK_ID_CURR",
            how="outer",
            validate="one_to_one"
        )

# 5. 检查生成的特征
print("\n===== 时间窗口特征结果 =====")
print("特征表维度：", window_result.shape)
print("重复客户ID：", window_result["SK_ID_CURR"].duplicated().sum())

print("\n特征名称：")
print(window_result.columns.tolist())

print("\n前5行：")
print(window_result.head().to_string())
# %%
# ==========================================
# Feature Engineering 11：
# 整合 POS/CASH 客户级特征宽表
# ==========================================

# 1. 以静态特征表为基础，保留所有POS历史客户
pos_features = static_features.copy()

# 2. 合并时间窗口特征
pos_features = pos_features.merge(
    window_result,
    on="SK_ID_CURR",
    how="left",
    validate="one_to_one"
)

# 3. 合并原始DPD趋势
pos_features = pos_features.merge(
    window_features[
        ["SK_ID_CURR", "POS_DPD_MEAN_DELTA_3V3"]
    ],
    on="SK_ID_CURR",
    how="left",
    validate="one_to_one"
)

# 4. 合并截尾DPD和逾期频率趋势
pos_features = pos_features.merge(
    features[
        [
            "SK_ID_CURR",
            "POS_DPD_CAP90_DELTA_3V3",
            "POS_OVERDUE_RATE_DELTA_3V3"
        ]
    ],
    on="SK_ID_CURR",
    how="left",
    validate="one_to_one"
)

# 5. 合并连续逾期和恢复特征
pos_features = pos_features.merge(
    streak_features,
    on="SK_ID_CURR",
    how="left",
    validate="one_to_one"
)

# 6. 合并严重逾期新近程度特征
pos_features = pos_features.merge(
    last_severe[
        ["SK_ID_CURR", "MONTHS_SINCE_SEVERE"]
    ].rename(columns={
        "MONTHS_SINCE_SEVERE":
        "POS_MONTHS_SINCE_LAST_DPD90"
    }),
    on="SK_ID_CURR",
    how="left",
    validate="one_to_one"
)

# 7. 验证特征表
print("\n===== POS/CASH 特征宽表 =====")

print("数据维度：", pos_features.shape)

print("重复客户ID：", pos_features["SK_ID_CURR"].duplicated().sum())

print("重复字段名：", pos_features.columns.duplicated().sum())

print("\n各字段缺失率：")
print(pos_features.isna().mean().sort_values(ascending=False).round(4).to_string())

# 8. 检查主表关联
model_check = application.merge(
    pos_features,
    on="SK_ID_CURR",
    how="left",
    validate="one_to_one"
)

print("\n关联后申请样本数：", len(model_check))

print("没有POS历史的申请数：", model_check["POS_OBS_MONTHS_ALL"].isna().sum())

# 9. 保存独立的POS/CASH特征表
pos_features.to_csv(
    r"D:\Home Credit Default Risk\features\pos_features.csv",
    index=False
)

print("\nPOS/CASH 特征表已保存")
# %%
