# %% Feature Engineering 01：分期级还款记录重构

import pandas as pd
import numpy as np

path = r"D:\Home Credit Default Risk"

cols = [
    "SK_ID_CURR",
    "SK_ID_PREV",
    "NUM_INSTALMENT_VERSION",
    "NUM_INSTALMENT_NUMBER",
    "DAYS_INSTALMENT",
    "DAYS_ENTRY_PAYMENT",
    "AMT_INSTALMENT",
    "AMT_PAYMENT"
]

ins = pd.read_csv(
    path + r"\data\installments_payments.csv",
    usecols=cols
)

key = [
    "SK_ID_PREV",
    "NUM_INSTALMENT_VERSION",
    "NUM_INSTALMENT_NUMBER"
]

# 1. 按分期及实际付款日期排序
ins = ins.sort_values(
    key + ["DAYS_ENTRY_PAYMENT"],
    kind="mergesort"
).reset_index(drop=True)

# 2. 记录付款缺失情况
ins["PAY_MISSING"] = (
    ins["AMT_PAYMENT"].isna() |
    ins["DAYS_ENTRY_PAYMENT"].isna()
)

# 3. 计算分期内累计付款
ins["CUM_PAYMENT"] = (
    ins["AMT_PAYMENT"].fillna(0)
    .groupby(
        [ins[k] for k in key],
        sort=False
    )
    .cumsum()
)

# 4. 找出首次足额还款的记录
# 金额容差 0.01，用于处理小额浮点误差
full_mask = (
    (ins["AMT_INSTALMENT"] > 0) &
    ins["DAYS_ENTRY_PAYMENT"].notna() &
    (ins["CUM_PAYMENT"] >= ins["AMT_INSTALMENT"] - 0.01)
)

# 已按日期排序，取每期首次达到应还额的日期
full_dates = (
    ins.loc[full_mask]
    .groupby(key, sort=False)["DAYS_ENTRY_PAYMENT"]
    .first()
    .rename("FULL_PAYMENT_DAY")
)

# 5. 聚合到分期级
inst_features = (
    ins.groupby(key, sort=False)
    .agg(
        SK_ID_CURR=("SK_ID_CURR", "first"),
        DAYS_INSTALMENT=("DAYS_INSTALMENT", "first"),
        AMT_INSTALMENT=("AMT_INSTALMENT", "first"),
        PAYMENT_COUNT=("AMT_PAYMENT", "count"),
        PAYMENT_TOTAL=("AMT_PAYMENT", "sum"),
        HAS_MISSING_PAYMENT=("PAY_MISSING", "max")
    )
)

# 6. 加入首次足额付款日期
inst_features = inst_features.join(full_dates)

# 7. 是否足额支付
inst_features["IS_FULLY_PAID"] = (
    inst_features["FULL_PAYMENT_DAY"].notna()
)

# 8. 足额付款延迟天数
inst_features["FULL_PAY_DELAY"] = (
    inst_features["FULL_PAYMENT_DAY"] -
    inst_features["DAYS_INSTALMENT"]
)

# 9. 付款覆盖率
inst_features["PAYMENT_COVERAGE"] = (
    inst_features["PAYMENT_TOTAL"] /
    inst_features["AMT_INSTALMENT"].replace(0, np.nan)
)

# 10. 还原普通列
inst_features = inst_features.reset_index()

print("\n===== 分期级特征表 =====")
print("数据维度：", inst_features.shape)

print("重复分期组合：", inst_features.duplicated(subset=key).sum())

print("\n足额付款情况：")
print(inst_features["IS_FULLY_PAID"].value_counts(dropna=False))

print("\n足额付款延迟天数：")
print(inst_features["FULL_PAY_DELAY"].describe(percentiles=[0.5, 0.9, 0.99]))

# 11. 保存分期级中间成果
inst_features.to_parquet(
    path + r"\features\installments_level.parquet",
    index=False
)

print("\n分期级数据已保存")
# %%
inst_features.to_parquet(
    path + r"\features\installments_level.parquet",
    index=False
)

print("分期级数据已保存")
# %%
# %% Feature Engineering 02：分期重构质量检查

print("===== 足额付款时间分类 =====")

delay = inst_features["FULL_PAY_DELAY"]

print("提前足额付款：", (delay < 0).sum())
print("按时足额付款：", (delay == 0).sum())
print("延期足额付款：", (delay > 0).sum())
print("未识别足额付款日期：", delay.isna().sum())

print("\n===== 延迟严重程度 =====")

print("延期超过7天：", (delay > 7).sum())
print("延期超过30天：", (delay > 30).sum())
print("延期超过90天：", (delay > 90).sum())

print("\n===== 极端时间差 =====")

print("提前超过365天：", (delay < -365).sum())
print("延期超过365天：", (delay > 365).sum())

print("\n===== 未识别足额付款的原因 =====")

unpaid = inst_features.loc[
    ~inst_features["IS_FULLY_PAID"]
]

print("未识别足额分期数：", len(unpaid))

print("应还金额为0或负数：", (unpaid["AMT_INSTALMENT"] <= 0).sum())

print("存在付款字段缺失：", unpaid["HAS_MISSING_PAYMENT"].sum())

print("付款合计小于应还金额：", (unpaid["PAYMENT_TOTAL"] < unpaid["AMT_INSTALMENT"] - 0.01).sum())

print("\n===== 极端延迟记录示例 =====")

print(
    inst_features.nlargest(5, "FULL_PAY_DELAY")[
        [
            "SK_ID_PREV",
            "DAYS_INSTALMENT",
            "FULL_PAYMENT_DAY",
            "AMT_INSTALMENT",
            "PAYMENT_TOTAL",
            "FULL_PAY_DELAY"
        ]
    ].to_string(index=False)
)
# %%
# %% EDA 05：极端延迟分期原始记录检查

extreme_ids = [1985467, 1088177, 2495473, 1564008, 2247294]

extreme_records = ins.loc[
    ins["SK_ID_PREV"].isin(extreme_ids)
].copy()

# 只选出这些贷款中 FULL_PAY_DELAY > 365 的分期组合
extreme_keys = inst_features.loc[
    inst_features["FULL_PAY_DELAY"] > 365,
    [
        "SK_ID_PREV",
        "NUM_INSTALMENT_VERSION",
        "NUM_INSTALMENT_NUMBER"
    ]
]

extreme_keys = extreme_keys.loc[
    extreme_keys["SK_ID_PREV"].isin(extreme_ids)
]

extreme_records = extreme_records.merge(
    extreme_keys,
    on=[
        "SK_ID_PREV",
        "NUM_INSTALMENT_VERSION",
        "NUM_INSTALMENT_NUMBER"
    ],
    how="inner"
)

print("===== 极端延迟分期的原始付款记录 =====")

print(
    extreme_records[
        [
            "SK_ID_PREV",
            "NUM_INSTALMENT_VERSION",
            "NUM_INSTALMENT_NUMBER",
            "DAYS_INSTALMENT",
            "DAYS_ENTRY_PAYMENT",
            "AMT_INSTALMENT",
            "AMT_PAYMENT"
        ]
    ].sort_values(
        ["SK_ID_PREV", "NUM_INSTALMENT_NUMBER",
         "DAYS_ENTRY_PAYMENT"]
    ).head(30).to_string(index=False)
)
# %%
# %% Feature Engineering 03：
# 95%、99%付款延迟 + 到期日付款覆盖率

import pandas as pd
import numpy as np

key = [
    "SK_ID_PREV",
    "NUM_INSTALMENT_VERSION",
    "NUM_INSTALMENT_NUMBER"
]

# 1. 仅选择日期和付款金额有效的记录
pay = ins.loc[
    ins["DAYS_ENTRY_PAYMENT"].notna()
    & ins["AMT_PAYMENT"].notna(),
    key + [
        "DAYS_INSTALMENT",
        "DAYS_ENTRY_PAYMENT",
        "AMT_INSTALMENT",
        "AMT_PAYMENT"
    ]
].copy()

# 2. 按分期、付款日期排序
pay = pay.sort_values(
    key + ["DAYS_ENTRY_PAYMENT"]
)

# 3. 计算每一期的累计付款
pay["CUM_PAYMENT"] = (
    pay.groupby(key, sort=False)["AMT_PAYMENT"]
    .cumsum()
)

# 4. 定义一个计算指定付款比例达到日期的函数
def get_threshold_day(data, threshold, name):

    mask = (
        (data["AMT_INSTALMENT"] > 0)
        & (
            data["CUM_PAYMENT"]
            >= data["AMT_INSTALMENT"] * threshold
        )
    )

    return (
        data.loc[mask]
        .groupby(key, sort=False)["DAYS_ENTRY_PAYMENT"]
        .first()
        .rename(name)
    )

# 5. 找到首次达到95%、99%的付款日期
day_95 = get_threshold_day(
    pay, 0.95, "PAY95_DAY"
)

day_99 = get_threshold_day(
    pay, 0.99, "PAY99_DAY"
)

# 6. 计算到期日前（含当天）累计付款金额
due_payment = (
    pay.loc[
        pay["DAYS_ENTRY_PAYMENT"]
        <= pay["DAYS_INSTALMENT"]
    ]
    .groupby(key)["AMT_PAYMENT"]
    .sum()
    .rename("PAYMENT_AT_DUE")
)

# 7. 连接到已有的分期级特征表
inst_features = inst_features.set_index(key)

inst_features = inst_features.join(
    [day_95, day_99, due_payment]
)

# 无有效的到期日前付款记录，累计已知付款额为0
inst_features["PAYMENT_AT_DUE"] = (
    inst_features["PAYMENT_AT_DUE"].fillna(0)
)

# 8. 计算95%、99%付款延迟天数
inst_features["PAY95_DELAY"] = (
    inst_features["PAY95_DAY"]
    - inst_features["DAYS_INSTALMENT"]
)

inst_features["PAY99_DELAY"] = (
    inst_features["PAY99_DAY"]
    - inst_features["DAYS_INSTALMENT"]
)

# 9. 到期日付款覆盖率
inst_features["DUE_DATE_COVERAGE"] = (
    inst_features["PAYMENT_AT_DUE"]
    / inst_features["AMT_INSTALMENT"].where(
        inst_features["AMT_INSTALMENT"] > 0
    )
)

# 10. 标记是否达到付款阈值
inst_features["REACHED_95"] = (
    inst_features["PAY95_DAY"].notna()
)

inst_features["REACHED_99"] = (
    inst_features["PAY99_DAY"].notna()
)

# 11. 还原普通列
inst_features = inst_features.reset_index()

print("===== 新增分期特征检查 =====")

print(
    inst_features[
        [
            "PAY95_DELAY",
            "PAY99_DELAY",
            "FULL_PAY_DELAY",
            "DUE_DATE_COVERAGE"
        ]
    ].describe(
        percentiles=[0.5, 0.9, 0.99]
    ).round(4).to_string()
)

print("\n达到95%付款的分期数：", inst_features["REACHED_95"].sum())

print("达到99%付款的分期数：", inst_features["REACHED_99"].sum())

print("\n分期级记录数：", len(inst_features))
print("重复分期组合：", inst_features.duplicated(subset=key).sum())
# %%
# %% Feature Engineering 04：付款覆盖率质量检查

import numpy as np

ratio = inst_features["DUE_DATE_COVERAGE"]

print("===== 到期日覆盖率检查 =====")

print("覆盖率缺失：", ratio.isna().sum())
print("覆盖率等于0：", (ratio == 0).sum())
print("覆盖率大于1：", (ratio > 1).sum())
print("覆盖率大于2：", (ratio > 2).sum())
print("覆盖率大于10：", (ratio > 10).sum())

# 查看最高的5条记录
print("\n===== 覆盖率最高的5条分期 =====")

print(
    inst_features.nlargest(5, "DUE_DATE_COVERAGE")[
        [
            "SK_ID_PREV",
            "NUM_INSTALMENT_NUMBER",
            "AMT_INSTALMENT",
            "PAYMENT_AT_DUE",
            "DUE_DATE_COVERAGE"
        ]
    ].to_string(index=False)
)

# 构造用于建模的候选指标
inst_features["DUE_COVERAGE_CAP1"] = (
    ratio.clip(lower=0, upper=1)
)

inst_features["DUE_OVERPAY_FLAG"] = (
    ratio > 1
).astype("Int64")

inst_features.loc[
    ratio.isna(),
    "DUE_OVERPAY_FLAG"
] = pd.NA

# 保留原始金额异常标记
inst_features["INVALID_DUE_AMOUNT"] = (
    inst_features["AMT_INSTALMENT"] <= 0
)

# 检查付款阈值的时间顺序
invalid_order = (
    (
        inst_features["PAY95_DAY"] >
        inst_features["PAY99_DAY"]
    ) |
    (
        inst_features["PAY99_DAY"] >
        inst_features["FULL_PAYMENT_DAY"]
    )
)

print("\n阈值日期顺序异常数：", invalid_order.sum())

print("\n截尾覆盖率描述统计：")
print(inst_features["DUE_COVERAGE_CAP1"].describe().round(4))
# %%
# %% Feature Engineering 05：客户级静态履约特征

import pandas as pd

# 1. 判断付款数据是否完整可靠
inst_features["VALID_PAYMENT"] = (
    (inst_features["AMT_INSTALMENT"] > 0)
    & (~inst_features["HAS_MISSING_PAYMENT"])
)

# 2. 延迟特征，只对可确认足额付款的分期计算
delay = inst_features["FULL_PAY_DELAY"].where(
    inst_features["VALID_PAYMENT"]
)

inst_features["LATE_FLAG"] = (delay > 0).where(
    delay.notna()
)

inst_features["DPD30_FLAG"] = (delay > 30).where(
    delay.notna()
)

# 只计算正向延迟，提前还款不计入延迟严重程度
inst_features["LATE_DAYS"] = (
    delay.where(delay > 0)
)

# 3. 异常付款标记
inst_features["PAYMENT_ANOMALY"] = (
    (inst_features["AMT_INSTALMENT"] <= 0)
    | inst_features["HAS_MISSING_PAYMENT"]
    | (inst_features["DUE_DATE_COVERAGE"] > 1.1)
)

# 4. 客户级聚合
ins_customer = (
    inst_features.groupby("SK_ID_CURR")
    .agg(
        INS_INSTALLMENT_COUNT=(
            "SK_ID_PREV", "size"
        ),
        INS_LATE_RATE_ALL=(
            "LATE_FLAG", "mean"
        ),
        INS_DPD_MEAN_ALL=(
            "LATE_DAYS", "mean"
        ),
        INS_DPD_MAX_ALL=(
            "LATE_DAYS", "max"
        ),
        INS_DPD30_RATE_ALL=(
            "DPD30_FLAG", "mean"
        ),
        INS_DUE_COVERAGE_MEAN=(
            "DUE_COVERAGE_CAP1", "mean"
        ),
        INS_PAYMENT_ANOMALY_RATE=(
            "PAYMENT_ANOMALY", "mean"
        ),
        INS_UNCONFIRMED_COUNT=(
            "IS_FULLY_PAID",
            lambda s: (~s).sum()
        )
    )
    .reset_index()
)

print("===== 客户级静态履约特征 =====")

print("数据维度：", ins_customer.shape)

print("重复客户ID：", ins_customer["SK_ID_CURR"].duplicated().sum())

print("\n特征描述统计：")
print(ins_customer.drop(columns="SK_ID_CURR").describe().round(4).to_string())

print("\n前5行：")
print(ins_customer.head().to_string())
# %%
# %% Feature Engineering 06：
# 补充正向延迟与客户级特征质量检查

import pandas as pd

# 仅对有效且能够确认足额付款日的分期计算
valid_delay = inst_features["FULL_PAY_DELAY"].where(
    inst_features["VALID_PAYMENT"]
)

# 提前/按时付款计0，延期保留正向天数
inst_features["LATE_DAYS_ALL"] = valid_delay.clip(lower=0)

# 为减弱极端尾款延期的影响，构造90天截尾版本
inst_features["LATE_DAYS_CAP90"] = (
    inst_features["LATE_DAYS_ALL"].clip(upper=90)
)

# 客户级统计
extra_features = (
    inst_features.groupby("SK_ID_CURR")
    .agg(
        INS_LATE_DAYS_AVG_ALL=("LATE_DAYS_ALL", "mean"),
        INS_LATE_DAYS_CAP90_AVG=("LATE_DAYS_CAP90", "mean"),
        INS_VALID_INSTALMENT_COUNT=("LATE_DAYS_ALL", "count")
    )
    .reset_index()
)

# 合并到已有客户级表
ins_customer = ins_customer.merge(
    extra_features,
    on="SK_ID_CURR",
    how="left",
    validate="one_to_one"
)

print("===== 补充特征结果 =====")
print(
    ins_customer[
        [
            "INS_LATE_DAYS_AVG_ALL",
            "INS_LATE_DAYS_CAP90_AVG",
            "INS_VALID_INSTALMENT_COUNT"
        ]
    ].describe().round(4).to_string()
)

print("\n延期比例字段类型：")
print(ins_customer["INS_LATE_RATE_ALL"].dtype)

print("\n延期超过30天比例字段类型：")
print(ins_customer["INS_DPD30_RATE_ALL"].dtype)

print("\n客户数：", len(ins_customer))
print("重复客户ID：", ins_customer["SK_ID_CURR"].duplicated().sum())
# %%
# %% Feature Engineering 07：修正比例特征的数据类型

import pandas as pd

# 1. 将两个比例特征转成数值类型
cols = [
    "INS_LATE_RATE_ALL",
    "INS_DPD30_RATE_ALL"
]

for col in cols:
    ins_customer[col] = pd.to_numeric(
        ins_customer[col],
        errors="coerce"
    )

# 2. 检查修正后的数据类型
print("===== 字段类型 =====")
print(ins_customer[cols].dtypes)

# 3. 查看实际分布
print("\n===== 延期比例描述统计 =====")
print(ins_customer[cols].describe().round(4).to_string())

# 4. 检查比例范围
for col in cols:
    invalid = (
        (ins_customer[col] < 0) |
        (ins_customer[col] > 1)
    ).sum()

    print(f"{col} 超出0-1范围的客户数：", invalid)

# 5. 确认客户粒度
print("\n客户数：", len(ins_customer))
print("重复客户ID：", ins_customer["SK_ID_CURR"].duplicated().sum())
# %%
# %% Feature Engineering 08：近期分期覆盖情况

# 只使用应还金额有效且付款字段完整的分期
valid = inst_features.loc[
    (inst_features["AMT_INSTALMENT"] > 0) &
    (~inst_features["HAS_MISSING_PAYMENT"])
].copy()

# 最近90天：-90 至 -1
recent = valid[
    valid["DAYS_INSTALMENT"].between(-90, -1)
]

# 前90天：-180 至 -91
previous = valid[
    valid["DAYS_INSTALMENT"].between(-180, -91)
]

print("===== 近90天 =====")
print("分期数量：", len(recent))
print("客户数量：", recent["SK_ID_CURR"].nunique())

print("\n===== 前90天 =====")
print("分期数量：", len(previous))
print("客户数量：", previous["SK_ID_CURR"].nunique())

# 各窗口每个客户有多少期分期
recent_count = recent.groupby("SK_ID_CURR").size()
previous_count = previous.groupby("SK_ID_CURR").size()

coverage = pd.concat(
    [
        recent_count.rename("RECENT_COUNT"),
        previous_count.rename("PREVIOUS_COUNT")
    ],
    axis=1
).fillna(0)

# 暂定两窗口各至少有2期
eligible = (
    (coverage["RECENT_COUNT"] >= 2) &
    (coverage["PREVIOUS_COUNT"] >= 2)
)

print("\n===== 趋势特征覆盖情况 =====")
print("两个窗口均至少2期的客户数：", eligible.sum())
print("拥有任一窗口记录的客户数：", len(coverage))
print("占全部Installments客户比例：", f"{eligible.sum() / 339587:.2%}")
# %%
# %% Feature Engineering 09：近期覆盖率趋势特征

import pandas as pd
import numpy as np

# 1. 保留计算所需字段
trend_base = inst_features.loc[
    (inst_features["AMT_INSTALMENT"] > 0)
    & (~inst_features["HAS_MISSING_PAYMENT"])
    & inst_features["DAYS_INSTALMENT"].between(-180, -1),
    [
        "SK_ID_CURR",
        "DAYS_INSTALMENT",
        "DUE_COVERAGE_CAP1"
    ]
].copy()

# 2. 划分观察窗口
trend_base["WINDOW"] = np.where(
    trend_base["DAYS_INSTALMENT"] >= -90,
    "D090",
    "PREV090"
)

# 3. 每个客户、每个窗口聚合
trend_agg = (
    trend_base.groupby(["SK_ID_CURR", "WINDOW"])
    .agg(
        OBS_COUNT=("DUE_COVERAGE_CAP1", "count"),
        COVERAGE_MEAN=("DUE_COVERAGE_CAP1", "mean"),
        COVERAGE_ZERO_RATE=(
            "DUE_COVERAGE_CAP1",
            lambda s: (s == 0).mean()
        )
    )
    .unstack("WINDOW")
)

# 4. 展开成客户级宽表
trend_agg.columns = [
    f"INS_{metric}_{window}"
    for metric, window in trend_agg.columns
]

trend_agg = trend_agg.reset_index()

# 5. 补齐没有记录的窗口计数
for col in [
    "INS_OBS_COUNT_D090",
    "INS_OBS_COUNT_PREV090"
]:
    trend_agg[col] = trend_agg[col].fillna(0).astype(int)

# 6. 判断是否满足双窗口比较条件
eligible = (
    (trend_agg["INS_OBS_COUNT_D090"] >= 2)
    & (trend_agg["INS_OBS_COUNT_PREV090"] >= 2)
)

trend_agg["INS_TREND_ELIGIBLE"] = eligible.astype(int)

# 7. 计算近期与前期覆盖率变化
trend_agg["INS_COVERAGE_DELTA_90V90"] = (
    trend_agg["INS_COVERAGE_MEAN_D090"]
    - trend_agg["INS_COVERAGE_MEAN_PREV090"]
).where(eligible)

# 8. 合并回客户级静态特征
ins_customer = ins_customer.merge(
    trend_agg,
    on="SK_ID_CURR",
    how="left",
    validate="one_to_one"
)

# 没有任一窗口记录的客户，观察数和可比较标记设为0
for col in [
    "INS_OBS_COUNT_D090",
    "INS_OBS_COUNT_PREV090",
    "INS_TREND_ELIGIBLE"
]:
    ins_customer[col] = (
        ins_customer[col].fillna(0).astype(int)
    )

# 9. 验收
print("===== 客户级近期趋势特征 =====")
print("客户数：", len(ins_customer))
print("重复客户ID：", ins_customer["SK_ID_CURR"].duplicated().sum())

print("满足趋势条件的客户：", ins_customer["INS_TREND_ELIGIBLE"].sum())

print("\n===== 覆盖率变化描述统计 =====")
print(
    ins_customer["INS_COVERAGE_DELTA_90V90"]
    .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    .round(4)
)

delta = ins_customer["INS_COVERAGE_DELTA_90V90"]

print("\n===== 趋势方向 =====")
print("近期覆盖率下降：", (delta < 0).sum())
print("近期覆盖率上升：", (delta > 0).sum())
print("两个窗口相同：", (delta == 0).sum())
print("无法计算趋势：", delta.isna().sum())
# %%
# %% Feature Engineering 10：结合TARGET验证近期覆盖率趋势

import pandas as pd
import numpy as np

# 1. 读取申请主表中的客户ID和违约标签
app = pd.read_csv(
    path + r"\data\application_train.csv",
    usecols=["SK_ID_CURR", "TARGET"]
)

# 2. 只连接本次研究的趋势特征
trend_cols = [
    "SK_ID_CURR",
    "INS_TREND_ELIGIBLE",
    "INS_COVERAGE_DELTA_90V90"
]

analysis = app.merge(
    ins_customer[trend_cols],
    on="SK_ID_CURR",
    how="left",
    validate="one_to_one"
)

# 3. 对没有Installments历史的客户也标记为无法比较
delta = analysis["INS_COVERAGE_DELTA_90V90"]

analysis["TREND_GROUP"] = np.select(
    [
        delta < -1e-10,
        delta > 1e-10,
        delta.notna()
    ],
    [
        "近期恶化",
        "近期改善",
        "基本不变"
    ],
    default="无法比较"
)

# 4. 统计各组违约率
result = (
    analysis.groupby("TREND_GROUP")
    .agg(
        CUSTOMER_COUNT=("SK_ID_CURR", "size"),
        BAD_COUNT=("TARGET", "sum"),
        BAD_RATE=("TARGET", "mean")
    )
)

result["BAD_RATE"] = (
    result["BAD_RATE"] * 100
).round(2)

result = result.reindex([
    "近期恶化",
    "基本不变",
    "近期改善",
    "无法比较"
])

print("===== 近期还款覆盖率趋势与TARGET =====")
print(result.to_string())

print("\n===== 数据检查 =====")
print("申请主表客户数：", len(app))
print("合并后客户数：", len(analysis))
print("重复客户ID：", analysis["SK_ID_CURR"].duplicated().sum())
print("总体违约率：", f'{analysis["TARGET"].mean():.2%}')
# %%
# %% Feature Engineering 11：近期绝对覆盖率与违约率

# 给分析表补充近期覆盖率
analysis = analysis.merge(
    ins_customer[
        ["SK_ID_CURR", "INS_COVERAGE_MEAN_D090"]
    ],
    on="SK_ID_CURR",
    how="left",
    validate="one_to_one"
)

# 分组：仅在能够计算趋势的客户中分析
trend_analysis = analysis.loc[
    analysis["TREND_GROUP"] != "无法比较"
].copy()

# 近期绝对覆盖率分组
trend_analysis["RECENT_LEVEL"] = pd.cut(
    trend_analysis["INS_COVERAGE_MEAN_D090"],
    bins=[-0.001, 0.8, 0.95, 0.999999, 1.000001],
    labels=["<=80%", "80%-95%", "95%-<100%", "100%"],
    include_lowest=True
)

# 同时考虑趋势方向和近期覆盖率
result = (
    trend_analysis.groupby(
        ["TREND_GROUP", "RECENT_LEVEL"],
        observed=True
    )
    .agg(
        CUSTOMER_COUNT=("SK_ID_CURR", "size"),
        BAD_RATE=("TARGET", "mean")
    )
    .reset_index()
)

result["BAD_RATE"] = (
    result["BAD_RATE"] * 100
).round(2)

print("===== 趋势方向 × 近期覆盖率 × 违约率 =====")
print(result.to_string(index=False))

print("\n===== 数据检查 =====")
print("趋势可计算客户数：", len(trend_analysis))
print("近期覆盖率分组缺失数：", trend_analysis["RECENT_LEVEL"].isna().sum())
# %%
# %% Feature Engineering 12：时间口径与未足额分期检查

import pandas as pd

print("===== 1. 时间范围检查 =====")

print(
    "计划还款日范围：",
    inst_features["DAYS_INSTALMENT"].min(),
    inst_features["DAYS_INSTALMENT"].max()
)

print("实际付款日范围：", ins["DAYS_ENTRY_PAYMENT"].min(), ins["DAYS_ENTRY_PAYMENT"].max())

print("申请日之后的计划还款分期数：", (inst_features["DAYS_INSTALMENT"] > 0).sum())

print("申请日之后的实际付款记录数：", (ins["DAYS_ENTRY_PAYMENT"] > 0).sum())

print("\n===== 2. 已到期但未确认足额的分期 =====")

# 当前到期日已经过去、但未识别到足额付款
unpaid_due = (
    (inst_features["DAYS_INSTALMENT"] <= -1)
    & (~inst_features["IS_FULLY_PAID"])
)

print("已到期且未识别足额付款：", unpaid_due.sum())

print("\n===== 3. 近期未足额分期 =====")

recent_mask = (
    inst_features["DAYS_INSTALMENT"].between(-90, -1)
)

recent_unpaid = (
    recent_mask
    & (~inst_features["IS_FULLY_PAID"])
)

print("近90天到期分期数：", recent_mask.sum())
print("其中未确认足额付款：", recent_unpaid.sum())

print("涉及客户数：", inst_features.loc[recent_unpaid, "SK_ID_CURR"].nunique())

print("\n===== 4. 数据有效性检查 =====")

print("非正数应还金额分期：", (inst_features["AMT_INSTALMENT"] <= 0).sum())

print("付款信息存在缺失的分期：", inst_features["HAS_MISSING_PAYMENT"].sum())

print(
    "分期组合重复数：",
    inst_features.duplicated(
        subset=[
            "SK_ID_PREV",
            "NUM_INSTALMENT_VERSION",
            "NUM_INSTALMENT_NUMBER"
        ]
    ).sum()
)
# %%
# %% Feature Engineering 13：最终客户级特征表整理与保存

import pandas as pd
import numpy as np

# 1. 构造分期级补充标记

due = inst_features["DAYS_INSTALMENT"]

# 未识别到足额付款，但不直接定义为违约
inst_features["INS_UNCONFIRMED_D090_FLAG"] = (
    due.between(-90, -1)
    & ~inst_features["IS_FULLY_PAID"]
).astype("int8")

# 观察期完整、付款字段完整、应还金额有效
valid_base = (
    (inst_features["AMT_INSTALMENT"] > 0)
    & (~inst_features["HAS_MISSING_PAYMENT"])
)

# 到期已经至少过去30天，仍未足额支付
# 或足额付款实际发生在到期30天之后
# 注意：这是到期后30天未足额的履约标记，不等同于监管DPD定义
inst_features["INS_UNPAID30_FLAG"] = (
    (
        inst_features["FULL_PAYMENT_DAY"].isna()
        | (
            inst_features["FULL_PAYMENT_DAY"]
            > due + 30
        )
    )
    & valid_base
    & (due <= -31)
).astype("int8")

# 近180天到期、且已有完整30天观察期：
# 到期日在 -180 至 -31
inst_features["INS_UNPAID30_D180_FLAG"] = (
    inst_features["INS_UNPAID30_FLAG"].eq(1)
    & due.between(-180, -31)
).astype("int8")


# 2. 按客户聚合新增特征

final_extra = (
    inst_features.groupby("SK_ID_CURR")
    .agg(
        INS_UNCONFIRMED_D090_COUNT=(
            "INS_UNCONFIRMED_D090_FLAG", "sum"
        ),
        INS_UNPAID30_COUNT_ALL=(
            "INS_UNPAID30_FLAG", "sum"
        ),
        INS_UNPAID30_COUNT_D180=(
            "INS_UNPAID30_D180_FLAG", "sum"
        ),
        INS_MISSING_PAYMENT_COUNT=(
            "HAS_MISSING_PAYMENT", "sum"
        ),
        INS_INVALID_AMOUNT_COUNT=(
            "INVALID_DUE_AMOUNT", "sum"
        )
    )
    .reset_index()
)


# 3. 合并到客户级表

# 避免重复运行Cell时产生 _x / _y 后缀
extra_cols = final_extra.columns.drop("SK_ID_CURR")

ins_customer = ins_customer.drop(
    columns=extra_cols,
    errors="ignore"
)

ins_customer = ins_customer.merge(
    final_extra,
    on="SK_ID_CURR",
    how="left",
    validate="one_to_one"
)


# 4. 检查字段类型

for col in ins_customer.columns:
    if col == "SK_ID_CURR":
        continue

    if ins_customer[col].dtype == "object":
        ins_customer[col] = pd.to_numeric(
            ins_customer[col],
            errors="raise"
        )


# 5. 检查最终表

print("===== Installments最终客户级特征表 =====")

print("数据维度：", ins_customer.shape)

print("重复客户ID：", ins_customer["SK_ID_CURR"].duplicated().sum())

print("重复字段名：", ins_customer.columns.duplicated().sum())

print("目标标签是否混入特征表：", "TARGET" in ins_customer.columns)

print("object类型字段数：", len(ins_customer.select_dtypes(include="object").columns))

print("近期未确认足额付款分期总数：", ins_customer["INS_UNCONFIRMED_D090_COUNT"].sum())

print("\n===== 缺失率最高的10个特征 =====")

print(
    ins_customer.isna()
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .round(4)
    .to_string()
)


# 6. 通过基础验收后保存

assert ins_customer["SK_ID_CURR"].is_unique
assert not ins_customer.columns.duplicated().any()
assert "TARGET" not in ins_customer.columns
assert len(ins_customer) == 339587

output_path = path + r"\features\installments_features.parquet"

ins_customer.to_parquet(
    output_path,
    index=False
)

print("\n保存成功：", output_path)
# %%
