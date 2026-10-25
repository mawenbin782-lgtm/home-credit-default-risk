# %% EDA 01：credit_card_balance基础结构检查

import pandas as pd
import numpy as np

path = r"D:\Home Credit Default Risk"

cc = pd.read_csv(
    path + r"\data\credit_card_balance.csv"
)

print("===== 数据规模 =====")
print("数据维度：", cc.shape)

print("\n===== ID检查 =====")
print("客户数量：", cc["SK_ID_CURR"].nunique())
print("信用卡账户数量：", cc["SK_ID_PREV"].nunique())

print("重复账户-月份组合：", cc.duplicated(["SK_ID_PREV", "MONTHS_BALANCE"]).sum())

print("\n===== 历史月份 =====")
print("最早月份：", cc["MONTHS_BALANCE"].min())
print("最近月份：", cc["MONTHS_BALANCE"].max())

print("\n===== 字段名称 =====")
print(cc.columns.tolist())

print("\n===== 关键字段缺失率 =====")

key_cols = [
    "AMT_BALANCE",
    "AMT_CREDIT_LIMIT_ACTUAL",
    "AMT_PAYMENT_TOTAL_CURRENT",
    "AMT_INST_MIN_REGULARITY",
    "SK_DPD",
    "SK_DPD_DEF"
]

for col in key_cols:
    if col in cc.columns:
        print(f"{col}：{cc[col].isna().mean():.2%}")

print("\n===== 全表缺失概览 =====")

missing = cc.isna().mean()

print("超过50%缺失的字段：", (missing > 0.5).sum())
print("无缺失字段：", (missing == 0).sum())
# %%
# %% EDA 02：信用卡余额与信用额度分布

cols = [
    "AMT_BALANCE",
    "AMT_CREDIT_LIMIT_ACTUAL"
]

print("===== 余额与信用额度分布 =====")

print(cc[cols].describe(percentiles=[0.5, 0.9, 0.99]).round(2).to_string())

print("\n===== 信用额度检查 =====")

print("额度等于0：", (cc["AMT_CREDIT_LIMIT_ACTUAL"] == 0).sum())

print("额度小于0：", (cc["AMT_CREDIT_LIMIT_ACTUAL"] < 0).sum())

print("\n===== 信用卡余额检查 =====")

print("余额等于0：", (cc["AMT_BALANCE"] == 0).sum())

print("余额小于0：", (cc["AMT_BALANCE"] < 0).sum())

print("余额大于信用额度：", (cc["AMT_BALANCE"] > cc["AMT_CREDIT_LIMIT_ACTUAL"]).sum())

print("\n===== 月度账户状态 =====")

print(cc["NAME_CONTRACT_STATUS"].value_counts(dropna=False).to_string())
# %%
# %% EDA 03：信用额度异常与账户状态

import numpy as np
import pandas as pd

limit = cc["AMT_CREDIT_LIMIT_ACTUAL"]
balance = cc["AMT_BALANCE"]

# 1. 将零额度与真正的超额度情况分开
cc["ZERO_LIMIT"] = limit.eq(0)

cc["ZERO_LIMIT_POS_BAL"] = (
    limit.eq(0) & balance.gt(0)
)

cc["OVER_LIMIT_VALID"] = (
    limit.gt(0) & balance.gt(limit)
)

cc["NEGATIVE_BALANCE"] = balance.lt(0)

print("===== 额度异常分类 =====")
print("额度为0、余额为正：", cc["ZERO_LIMIT_POS_BAL"].sum())
print("额度大于0、余额超限：", cc["OVER_LIMIT_VALID"].sum())
print("余额为负：", cc["NEGATIVE_BALANCE"].sum())

# 2. 按账户状态检查
status_quality = (
    cc.groupby("NAME_CONTRACT_STATUS")
    .agg(
        MONTH_COUNT=("SK_ID_PREV", "size"),
        ZERO_LIMIT_RATE=("ZERO_LIMIT", "mean"),
        ZERO_LIMIT_POS_BAL_RATE=("ZERO_LIMIT_POS_BAL", "mean"),
        OVER_LIMIT_RATE=("OVER_LIMIT_VALID", "mean")
    )
)

rate_cols = [
    "ZERO_LIMIT_RATE",
    "ZERO_LIMIT_POS_BAL_RATE",
    "OVER_LIMIT_RATE"
]

status_quality[rate_cols] = (
    status_quality[rate_cols] * 100
).round(2)

print("\n===== 各账户状态的额度异常比例（%） =====")
print(status_quality.to_string())

# 3. 仅对额度大于0的月份计算使用率
cc["UTIL_RAW"] = (
    balance / limit.where(limit > 0)
)

print("\n===== 有效额度月份的原始使用率 =====")
print(cc["UTIL_RAW"].describe(percentiles=[0.5, 0.9, 0.99]).round(4).to_string())

print("\n额度大于0的月份：", cc["UTIL_RAW"].notna().sum())
print("使用率超过100%的月份：", (cc["UTIL_RAW"] > 1).sum())
print("使用率超过200%的月份：", (cc["UTIL_RAW"] > 2).sum())
# %%
# %% Feature Engineering 01：客户级信用额度使用特征

import numpy as np
import pandas as pd

limit = cc["AMT_CREDIT_LIMIT_ACTUAL"]
balance = cc["AMT_BALANCE"]

# 1. 有效额度月份
cc["VALID_LIMIT"] = limit.gt(0).astype("int8")

# 2. 原始使用率（上一轮已计算，这里保证可独立执行）
cc["UTIL_RAW"] = (
    balance / limit.where(limit > 0)
)

# 3. 截尾使用率：用于反映额度使用程度
cc["UTIL_CAP1"] = cc["UTIL_RAW"].clip(
    lower=0, upper=1
)

# 4. 真正超限：额度大于0，且余额超过额度
cc["OVERLIMIT_FLAG"] = (
    (limit > 0) & (balance > limit)
).astype("int8")

# 无有效额度时不参与超限比例计算
cc["OVERLIMIT_VALID"] = (
    cc["OVERLIMIT_FLAG"]
    .astype(float)
    .where(limit > 0)
)

# 5. 零额度、负余额标记
cc["ZERO_LIMIT_FLAG"] = limit.eq(0).astype("int8")
cc["NEGATIVE_BALANCE_FLAG"] = balance.lt(0).astype("int8")

# 6. 客户级聚合
cc_customer = (
    cc.groupby("SK_ID_CURR")
    .agg(
        CC_OBS_MONTHS=("MONTHS_BALANCE", "size"),
        CC_ACCOUNT_COUNT=("SK_ID_PREV", "nunique"),

        CC_UTIL_MEAN_ALL=("UTIL_RAW", "mean"),
        CC_UTIL_MAX_ALL=("UTIL_RAW", "max"),
        CC_UTIL_CAP1_MEAN_ALL=("UTIL_CAP1", "mean"),

        CC_OVERLIMIT_RATE_ALL=("OVERLIMIT_VALID", "mean"),
        CC_ZERO_LIMIT_RATE_ALL=("ZERO_LIMIT_FLAG", "mean"),
        CC_NEGATIVE_BALANCE_COUNT=("NEGATIVE_BALANCE_FLAG", "sum"),

        CC_BALANCE_MEAN_ALL=("AMT_BALANCE", "mean"),
        CC_BALANCE_MAX_ALL=("AMT_BALANCE", "max"),

        CC_VALID_LIMIT_MONTHS=("VALID_LIMIT", "sum")
    )
    .reset_index()
)

# 7. 基础验收
print("===== 信用卡客户级静态特征 =====")
print("数据维度：", cc_customer.shape)
print("重复客户ID：", cc_customer["SK_ID_CURR"].duplicated().sum())

print("\n===== 核心特征描述统计 =====")
cols = [
    "CC_OBS_MONTHS",
    "CC_ACCOUNT_COUNT",
    "CC_UTIL_MEAN_ALL",
    "CC_UTIL_CAP1_MEAN_ALL",
    "CC_OVERLIMIT_RATE_ALL",
    "CC_ZERO_LIMIT_RATE_ALL"
]

print(cc_customer[cols].describe(percentiles=[0.5, 0.9, 0.99]).round(4).to_string())

print("\n===== 额度使用率覆盖情况 =====")
print("没有有效额度月份的客户：", (cc_customer["CC_VALID_LIMIT_MONTHS"] == 0).sum())
# %%
# %% Feature Engineering 02：近期额度使用率趋势

import pandas as pd
import numpy as np

# 1. 只取最近6个月
cc6 = cc.loc[
    cc["MONTHS_BALANCE"].between(-6, -1),
    ["SK_ID_CURR", "MONTHS_BALANCE", "UTIL_CAP1"]
].copy()

# 2. 划分窗口
cc6["WINDOW"] = np.where(
    cc6["MONTHS_BALANCE"] >= -3,
    "D3",
    "PREV3"
)

# 3. 按客户、窗口计算平均使用率和有效记录数
trend = (
    cc6.groupby(["SK_ID_CURR", "WINDOW"])
    .agg(
        UTIL_MEAN=("UTIL_CAP1", "mean"),
        UTIL_VALID=("UTIL_CAP1", "count")
    )
    .unstack("WINDOW")
)

# 4. 展开列名
trend.columns = [
    f"CC_{metric}_{window}"
    for metric, window in trend.columns
]

trend = trend.reset_index()

# 5. 没有有效额度的窗口，计数设为0
for col in ["CC_UTIL_VALID_D3", "CC_UTIL_VALID_PREV3"]:
    trend[col] = trend[col].fillna(0).astype(int)

# 6. 双窗口各至少2条有效额度记录
eligible = (
    (trend["CC_UTIL_VALID_D3"] >= 2)
    & (trend["CC_UTIL_VALID_PREV3"] >= 2)
)

# 7. 计算近期变化
trend["CC_UTIL_DELTA_3V3"] = (
    trend["CC_UTIL_MEAN_D3"]
    - trend["CC_UTIL_MEAN_PREV3"]
).where(eligible)

# 8. 合并回客户级特征表
# 允许重复运行Cell，避免生成_x、_y列
new_cols = trend.columns.drop("SK_ID_CURR")

cc_customer = cc_customer.drop(
    columns=new_cols,
    errors="ignore"
)

cc_customer = cc_customer.merge(
    trend,
    on="SK_ID_CURR",
    how="left",
    validate="one_to_one"
)

for col in ["CC_UTIL_VALID_D3", "CC_UTIL_VALID_PREV3"]:
    cc_customer[col] = cc_customer[col].fillna(0).astype(int)

# 9. 检查
print("===== 信用卡近期使用率趋势 =====")

print("客户级维度：", cc_customer.shape)
print("重复客户ID：", cc_customer["SK_ID_CURR"].duplicated().sum())

delta = cc_customer["CC_UTIL_DELTA_3V3"]

print("\n===== 趋势覆盖 =====")
print("可计算趋势客户：", delta.notna().sum())
print("无法计算趋势客户：", delta.isna().sum())

print("\n===== 趋势方向 =====")
print("近期使用率上升：", (delta > 1e-10).sum())
print("近期使用率下降：", (delta < -1e-10).sum())
print("基本不变：", delta.notna().sum() - (delta > 1e-10).sum() - (delta < -1e-10).sum())

print("\n===== 变化量分布 =====")
print(delta.describe(percentiles=[0.1, 0.5, 0.9]).round(4).to_string())
# %%
# %% EDA 04：信用卡还款金额与逾期检查

cols = [
    "AMT_PAYMENT_TOTAL_CURRENT",
    "AMT_INST_MIN_REGULARITY",
    "SK_DPD",
    "SK_DPD_DEF"
]

print("===== 还款与逾期分布 =====")
print(cc[cols].describe(percentiles=[0.5, 0.9, 0.99]).round(2).to_string())

print("\n===== 还款金额质量 =====")

print("最低应还金额缺失：", cc["AMT_INST_MIN_REGULARITY"].isna().sum())

print("最低应还金额大于0：", (cc["AMT_INST_MIN_REGULARITY"] > 0).sum())

print("实际付款金额小于0：", (cc["AMT_PAYMENT_TOTAL_CURRENT"] < 0).sum())

print("\n===== 逾期情况 =====")

print("DPD大于0：", (cc["SK_DPD"] > 0).sum())
print("DPD大于30：", (cc["SK_DPD"] > 30).sum())
print("DPD大于90：", (cc["SK_DPD"] > 90).sum())

print("DPD_DEF大于0：", (cc["SK_DPD_DEF"] > 0).sum())

print("两个DPD字段不相等：", (cc["SK_DPD"] != cc["SK_DPD_DEF"]).sum())
# %%
# %% Feature Engineering 03：客户级还款与逾期特征

import pandas as pd
import numpy as np

# 1. 最低应还金额为正，才能计算付款覆盖率
min_due = cc["AMT_INST_MIN_REGULARITY"]
payment = cc["AMT_PAYMENT_TOTAL_CURRENT"]

valid_min = min_due > 0

cc["MIN_PAY_COVERAGE"] = (
    payment / min_due.where(valid_min)
).clip(lower=0, upper=1)

# 付款少于最低应还金额
# 无有效最低应还金额的月份保持 NaN
cc["MIN_PAY_SHORT"] = (
    (payment < min_due)
    .astype(float)
    .where(valid_min)
)

cc["MIN_PAY_VALID"] = valid_min.astype("int8")

# 2. DPD 特征
cc["DPD_FLAG"] = (cc["SK_DPD"] > 0).astype("int8")
cc["DPD30_FLAG"] = (cc["SK_DPD"] > 30).astype("int8")
cc["DPD_DEF_FLAG"] = (cc["SK_DPD_DEF"] > 0).astype("int8")

cc["DPD_CAP90"] = cc["SK_DPD"].clip(
    lower=0, upper=90
)

# 3. 聚合到客户级
payment_dpd_features = (
    cc.groupby("SK_ID_CURR")
    .agg(
        CC_MIN_PAY_COVERAGE_MEAN=(
            "MIN_PAY_COVERAGE", "mean"
        ),
        CC_MIN_PAY_SHORT_RATE=(
            "MIN_PAY_SHORT", "mean"
        ),
        CC_MIN_PAY_VALID_MONTHS=(
            "MIN_PAY_VALID", "sum"
        ),
        CC_DPD_RATE_ALL=(
            "DPD_FLAG", "mean"
        ),
        CC_DPD30_RATE_ALL=(
            "DPD30_FLAG", "mean"
        ),
        CC_DPD_MAX_ALL=(
            "SK_DPD", "max"
        ),
        CC_DPD_DEF_RATE_ALL=(
            "DPD_DEF_FLAG", "mean"
        ),
        CC_DPD_CAP90_MEAN=(
            "DPD_CAP90", "mean"
        )
    )
    .reset_index()
)

# 4. 合并回已有客户特征表
new_cols = payment_dpd_features.columns.drop("SK_ID_CURR")

cc_customer = cc_customer.drop(
    columns=new_cols,
    errors="ignore"
)

cc_customer = cc_customer.merge(
    payment_dpd_features,
    on="SK_ID_CURR",
    how="left",
    validate="one_to_one"
)

# 5. 验收
print("===== 信用卡还款与逾期客户级特征 =====")
print("数据维度：", cc_customer.shape)

print("重复客户ID：", cc_customer["SK_ID_CURR"].duplicated().sum())

print("\n===== 核心特征分布 =====")

cols = [
    "CC_MIN_PAY_COVERAGE_MEAN",
    "CC_MIN_PAY_SHORT_RATE",
    "CC_MIN_PAY_VALID_MONTHS",
    "CC_DPD_RATE_ALL",
    "CC_DPD30_RATE_ALL",
    "CC_DPD_MAX_ALL",
    "CC_DPD_DEF_RATE_ALL",
    "CC_DPD_CAP90_MEAN"
]

print(cc_customer[cols].describe(percentiles=[0.5, 0.9, 0.99]).round(4).to_string())

print("\n===== 最低应还覆盖情况 =====")

print("没有有效最低应还月份的客户：", (cc_customer["CC_MIN_PAY_VALID_MONTHS"] == 0).sum())

print("至少一个月发生DPD逾期的客户：", (cc_customer["CC_DPD_MAX_ALL"] > 0).sum())

print("至少一个月DPD超过30天的客户：", (cc_customer["CC_DPD_MAX_ALL"] > 30).sum())
# %%# %% Feature Engineering 04：近期信用卡DPD趋势

import pandas as pd
import numpy as np

# 1. 取最近6个月的历史数据
dpd6 = cc.loc[
    cc["MONTHS_BALANCE"].between(-6, -1),
    ["SK_ID_CURR", "MONTHS_BALANCE", "SK_DPD"]
].copy()

# 2. 划分两个时间窗口
dpd6["WINDOW"] = np.where(
    dpd6["MONTHS_BALANCE"] >= -3,
    "D3",
    "PREV3"
)

# 3. 构造月度逾期标记
dpd6["IS_OVERDUE"] = (
    dpd6["SK_DPD"] > 0
).astype("int8")

# 4. 按客户、窗口聚合
dpd_trend = (
    dpd6.groupby(["SK_ID_CURR", "WINDOW"])
    .agg(
        DPD_OBS=("SK_DPD", "count"),
        DPD_RATE=("IS_OVERDUE", "mean"),
        DPD_MAX=("SK_DPD", "max")
    )
    .unstack("WINDOW")
)

# 5. 展开列名
dpd_trend.columns = [
    f"CC_{metric}_{window}"
    for metric, window in dpd_trend.columns
]

dpd_trend = dpd_trend.reset_index()

# 6. 没有记录的窗口，观察数量填0
for col in ["CC_DPD_OBS_D3", "CC_DPD_OBS_PREV3"]:
    dpd_trend[col] = (
        dpd_trend[col].fillna(0).astype(int)
    )

# 7. 两窗口各至少2条记录
eligible = (
    (dpd_trend["CC_DPD_OBS_D3"] >= 2)
    & (dpd_trend["CC_DPD_OBS_PREV3"] >= 2)
)

# 8. 近期减前期
dpd_trend["CC_DPD_DELTA_3V3"] = (
    dpd_trend["CC_DPD_RATE_D3"]
    - dpd_trend["CC_DPD_RATE_PREV3"]
).where(eligible)

# 9. 合并到已有客户级特征
new_cols = dpd_trend.columns.drop("SK_ID_CURR")

cc_customer = cc_customer.drop(
    columns=new_cols,
    errors="ignore"
)

cc_customer = cc_customer.merge(
    dpd_trend,
    on="SK_ID_CURR",
    how="left",
    validate="one_to_one"
)

# 10. 验收
print("===== 信用卡近期DPD趋势 =====")
print("数据维度：", cc_customer.shape)
print("重复客户ID：", cc_customer["SK_ID_CURR"].duplicated().sum())

delta = cc_customer["CC_DPD_DELTA_3V3"]

print("\n===== 趋势覆盖 =====")
print("可计算趋势客户：", delta.notna().sum())
print("无法计算趋势客户：", delta.isna().sum())

print("\n===== 趋势方向 =====")
print("近期恶化：", (delta > 0).sum())
print("近期改善：", (delta < 0).sum())
print("保持不变：", (delta == 0).sum())

print("\n===== 变化量描述统计 =====")
print(delta.describe(percentiles=[0.1, 0.5, 0.9]).round(4).to_string())

# %%
# %% EDA 05：信用卡额度使用率与DPD趋势的TARGET验证

import pandas as pd
import numpy as np

path = r"D:\Home Credit Default Risk"

app = pd.read_csv(
    path + r"\data\application_train.csv",
    usecols=["SK_ID_CURR", "TARGET"]
)

# 1. 关联客户级特征
cols = [
    "SK_ID_CURR",
    "CC_UTIL_CAP1_MEAN_ALL",
    "CC_DPD_DELTA_3V3",
    "CC_DPD_RATE_D3"
]

analysis = app.merge(
    cc_customer[cols],
    on="SK_ID_CURR",
    how="left",
    validate="one_to_one"
)

# 2. 历史平均额度使用率分组
u = analysis["CC_UTIL_CAP1_MEAN_ALL"]

analysis["UTIL_GROUP"] = pd.cut(
    u,
    bins=[-0.001, 0.2, 0.5, 0.8, 1.000001],
    labels=["0%-20%", "20%-50%", "50%-80%", "80%-100%"],
    include_lowest=True
)

analysis["UTIL_GROUP"] = (
    analysis["UTIL_GROUP"]
    .cat.add_categories(["无有效额度历史"])
    .fillna("无有效额度历史")
)

# 3. DPD趋势分组
d = analysis["CC_DPD_DELTA_3V3"]

analysis["DPD_TREND_GROUP"] = np.select(
    [
        d > 1e-10,
        d < -1e-10,
        d.notna()
    ],
    [
        "近期恶化",
        "近期改善",
        "基本不变"
    ],
    default="无法比较"
)

# 4. 统计风险
def risk_summary(group_col):
    result = (
        analysis.groupby(group_col, observed=True)
        .agg(
            CUSTOMER_COUNT=("SK_ID_CURR", "size"),
            BAD_COUNT=("TARGET", "sum"),
            BAD_RATE=("TARGET", "mean")
        )
    )

    result["BAD_RATE"] = (
        result["BAD_RATE"] * 100
    ).round(2)

    return result

print("===== 信用卡历史额度使用率与TARGET =====")
print(risk_summary("UTIL_GROUP").to_string())

print("\n===== 近期DPD趋势与TARGET =====")
print(risk_summary("DPD_TREND_GROUP").to_string())

print("\n===== 数据检查 =====")
print("申请客户数：", len(analysis))
print("重复客户ID：", analysis["SK_ID_CURR"].duplicated().sum())
print("总体违约率：", f'{analysis["TARGET"].mean():.2%}')
# %%
# %% Feature Engineering 05：最终验收与保存

import pandas as pd
import numpy as np

path = r"D:\Home Credit Default Risk"

print("===== 信用卡最终客户级特征表 =====")

print("数据维度：", cc_customer.shape)
print("重复客户ID：", cc_customer["SK_ID_CURR"].duplicated().sum())
print("重复字段名：", cc_customer.columns.duplicated().sum())
print("是否包含TARGET：", "TARGET" in cc_customer.columns)
print("object类型字段数：", len(cc_customer.select_dtypes(include="object").columns))

print("\n===== 缺失率最高的10个特征 =====")
print(
    cc_customer.isna()
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .round(4)
    .to_string()
)

print("\n===== 无穷大检查 =====")
numeric = cc_customer.select_dtypes(include="number")
print("无穷大数量：", np.isinf(numeric.to_numpy(dtype="float64")).sum())

# 基础结构验收
assert cc_customer.shape == (103558, 32)
assert cc_customer["SK_ID_CURR"].is_unique
assert not cc_customer.columns.duplicated().any()
assert "TARGET" not in cc_customer.columns
assert len(cc_customer.select_dtypes(include="object").columns) == 0
assert not np.isinf(numeric.to_numpy(dtype="float64")).any()

# 保存
output_path = path + r"\features\credit_card_features.parquet"

cc_customer.to_parquet(
    output_path,
    index=False
)

# 重新读取进行验证
check = pd.read_parquet(output_path)

assert check.shape == cc_customer.shape
assert check["SK_ID_CURR"].is_unique

print("\n信用卡客户级特征表保存并读取验证成功！")
print("保存路径：", output_path)