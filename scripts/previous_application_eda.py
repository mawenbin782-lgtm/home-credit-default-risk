# %% EDA 01：previous_application基础检查

import pandas as pd
import numpy as np

path = r"D:\Home Credit Default Risk"

prev = pd.read_csv(
    path + r"\data\previous_application.csv"
)

print("===== 数据规模 =====")
print("数据维度：", prev.shape)

print("\n===== ID检查 =====")
print("客户数量：", prev["SK_ID_CURR"].nunique())
print("历史申请数量：", prev["SK_ID_PREV"].nunique())
print("重复申请ID：", prev["SK_ID_PREV"].duplicated().sum())

print("\n===== 审批状态 =====")
print(prev["NAME_CONTRACT_STATUS"].value_counts(dropna=False).to_string())

print("\n===== 关键字段类型 =====")
cols = [
    "SK_ID_CURR",
    "SK_ID_PREV",
    "NAME_CONTRACT_STATUS",
    "AMT_APPLICATION",
    "AMT_CREDIT",
    "DAYS_DECISION"
]

print(prev[cols].dtypes.to_string())

print("\n===== 缺失率概览 =====")
missing = prev.isna().mean()

print("缺失率超过50%的字段数：", (missing > 0.5).sum())
print("完全无缺失的字段数：", (missing == 0).sum())
# %%
# %% EDA 02：不同审批状态下的申请金额

print("===== 不同审批状态的金额情况 =====")

amount_stats = (
    prev.groupby("NAME_CONTRACT_STATUS")
    .agg(
        RECORD_COUNT=("SK_ID_PREV", "size"),
        APPLICATION_MEDIAN=("AMT_APPLICATION", "median"),
        CREDIT_MEDIAN=("AMT_CREDIT", "median"),
        APPLICATION_MISSING=("AMT_APPLICATION",
                             lambda s: s.isna().mean()),
        CREDIT_MISSING=("AMT_CREDIT",
                        lambda s: s.isna().mean())
    )
)

print(amount_stats.round(4).to_string())

print("\n===== Approved申请金额分布 =====")

approved = prev.loc[
    prev["NAME_CONTRACT_STATUS"] == "Approved"
]

print(
    approved[["AMT_APPLICATION", "AMT_CREDIT"]]
    .describe(percentiles=[0.5, 0.9, 0.99])
    .round(2)
    .to_string()
)

print("\n===== 金额异常检查 =====")

print("申请金额为0：", (prev["AMT_APPLICATION"] == 0).sum())

print("授信金额为0：", (prev["AMT_CREDIT"] == 0).sum())

print("Approved且申请金额大于0的记录：",
      (
          (prev["NAME_CONTRACT_STATUS"] == "Approved")
          & (prev["AMT_APPLICATION"] > 0)
      ).sum())
# %%
# %% EDA 03：Approved申请的金额调整情况

import numpy as np

approved_valid = prev.loc[
    (prev["NAME_CONTRACT_STATUS"] == "Approved")
    & (prev["AMT_APPLICATION"] > 0)
    & (prev["AMT_CREDIT"] > 0)
].copy()

approved_valid["CREDIT_APP_RATIO"] = (
    approved_valid["AMT_CREDIT"]
    / approved_valid["AMT_APPLICATION"]
)

ratio = approved_valid["CREDIT_APP_RATIO"]

print("===== 有效Approved申请 =====")
print("记录数：", len(approved_valid))

print("\n===== 授信申请比分布 =====")
print(ratio.describe(percentiles=[0.1, 0.5, 0.9, 0.99]).round(4).to_string())

print("\n===== 金额调整方向 =====")
print("授信金额低于申请金额：", (ratio < 0.99).sum())
print("基本相同（0.99～1.01）：", ratio.between(0.99, 1.01).sum())
print("授信金额高于申请金额：", (ratio > 1.01).sum())

print("\n===== 极端比值 =====")
print("比值大于2：", (ratio > 2).sum())
print("比值小于0.5：", (ratio < 0.5).sum())
# %%
# %% EDA 04：历史申请时间与近期拒绝行为

print("===== DAYS_DECISION时间分布 =====")

print(
    prev["DAYS_DECISION"]
    .describe(percentiles=[0.5, 0.9, 0.99])
    .round(2)
    .to_string()
)

print("\n===== 近期历史申请数量 =====")

for days in [90, 180, 365]:
    mask = prev["DAYS_DECISION"].between(-days, -1)

    print(f"最近{days}天申请数：", mask.sum())

print("\n===== 近期拒绝记录 =====")

refused = prev["NAME_CONTRACT_STATUS"].eq("Refused")

for days in [90, 180, 365]:
    mask = prev["DAYS_DECISION"].between(-days, -1)

    print(f"最近{days}天拒绝数：", (mask & refused).sum())

print("\n===== 时间异常检查 =====")

print("申请日之后的历史决策记录：", (prev["DAYS_DECISION"] > 0).sum())

print("申请当天的历史决策记录：", (prev["DAYS_DECISION"] == 0).sum())
# %%
# %% Feature Engineering 01：历史申请与审批行为特征

import pandas as pd
import numpy as np

# 1. 审批状态标记
prev["IS_APPROVED"] = (
    prev["NAME_CONTRACT_STATUS"] == "Approved"
).astype("int8")

prev["IS_REFUSED"] = (
    prev["NAME_CONTRACT_STATUS"] == "Refused"
).astype("int8")

prev["IS_CANCELED"] = (
    prev["NAME_CONTRACT_STATUS"] == "Canceled"
).astype("int8")

# 2. 时间窗口标记
prev["APP_D090"] = (
    prev["DAYS_DECISION"].between(-90, -1)
).astype("int8")

prev["APP_PREV090"] = (
    prev["DAYS_DECISION"].between(-180, -91)
).astype("int8")

prev["APP_D365"] = (
    prev["DAYS_DECISION"].between(-365, -1)
).astype("int8")

# 3. 窗口内拒绝标记
prev["REFUSED_D090"] = (
    prev["IS_REFUSED"] & prev["APP_D090"]
).astype("int8")

prev["REFUSED_PREV090"] = (
    prev["IS_REFUSED"] & prev["APP_PREV090"]
).astype("int8")

# 4. 聚合到客户级
prev_customer = (
    prev.groupby("SK_ID_CURR")
    .agg(
        PREV_APP_COUNT=("SK_ID_PREV", "size"),
        PREV_APPROVED_COUNT=("IS_APPROVED", "sum"),
        PREV_REFUSED_COUNT=("IS_REFUSED", "sum"),
        PREV_CANCELED_COUNT=("IS_CANCELED", "sum"),

        PREV_APP_D090=("APP_D090", "sum"),
        PREV_APP_PREV090=("APP_PREV090", "sum"),
        PREV_APP_D365=("APP_D365", "sum"),

        PREV_REFUSED_D090=("REFUSED_D090", "sum"),
        PREV_REFUSED_PREV090=("REFUSED_PREV090", "sum"),

        PREV_LAST_DECISION_DAY=("DAYS_DECISION", "max")
    )
    .reset_index()
)

# 5. 全历史拒绝率
prev_customer["PREV_REFUSED_RATE"] = (
    prev_customer["PREV_REFUSED_COUNT"]
    / prev_customer["PREV_APP_COUNT"]
)

# 6. 最近90天拒绝率
prev_customer["PREV_REFUSED_RATE_D090"] = (
    prev_customer["PREV_REFUSED_D090"]
    / prev_customer["PREV_APP_D090"].replace(0, np.nan)
)

# 7. 前90天拒绝率
prev_customer["PREV_REFUSED_RATE_PREV090"] = (
    prev_customer["PREV_REFUSED_PREV090"]
    / prev_customer["PREV_APP_PREV090"].replace(0, np.nan)
)

# 8. 两窗口都有申请，才计算拒绝率变化
eligible = (
    (prev_customer["PREV_APP_D090"] > 0)
    & (prev_customer["PREV_APP_PREV090"] > 0)
)

prev_customer["PREV_REFUSED_RATE_DELTA_90V90"] = (
    prev_customer["PREV_REFUSED_RATE_D090"]
    - prev_customer["PREV_REFUSED_RATE_PREV090"]
).where(eligible)

# 9. 距离最近一次历史决策的天数
prev_customer["PREV_DAYS_SINCE_LAST_APP"] = (
    -prev_customer["PREV_LAST_DECISION_DAY"]
)

prev_customer = prev_customer.drop(
    columns="PREV_LAST_DECISION_DAY"
)

# 10. 验收
print("===== previous_application第一批客户级特征 =====")

print("数据维度：", prev_customer.shape)
print("重复客户ID：", prev_customer["SK_ID_CURR"].duplicated().sum())

print("\n===== 核心特征分布 =====")
cols = [
    "PREV_APP_COUNT",
    "PREV_REFUSED_COUNT",
    "PREV_REFUSED_RATE",
    "PREV_APP_D090",
    "PREV_REFUSED_D090",
    "PREV_REFUSED_RATE_D090",
    "PREV_REFUSED_RATE_DELTA_90V90"
]

print(
    prev_customer[cols]
    .describe(percentiles=[0.5, 0.9, 0.99])
    .round(4)
    .to_string()
)

print("\n===== 近期观察覆盖 =====")
print("近90天有申请的客户：", (prev_customer["PREV_APP_D090"] > 0).sum())

print("双窗口都有申请的客户：", eligible.sum())
# %%
# %% Feature Engineering 02：历史申请金额特征

import pandas as pd
import numpy as np

# 1. 只对正申请金额统计均值
prev["APP_AMT_POSITIVE"] = (
    prev["AMT_APPLICATION"]
    .where(prev["AMT_APPLICATION"] > 0)
)

# 2. 区分获批和拒绝申请金额
prev["APPROVED_APP_AMT"] = (
    prev["AMT_APPLICATION"].where(
        (prev["NAME_CONTRACT_STATUS"] == "Approved")
        & (prev["AMT_APPLICATION"] > 0)
    )
)

prev["REFUSED_APP_AMT"] = (
    prev["AMT_APPLICATION"].where(
        (prev["NAME_CONTRACT_STATUS"] == "Refused")
        & (prev["AMT_APPLICATION"] > 0)
    )
)

# 3. 有效获批申请的金额比
valid_approved = (
    (prev["NAME_CONTRACT_STATUS"] == "Approved")
    & (prev["AMT_APPLICATION"] > 0)
    & (prev["AMT_CREDIT"] > 0)
)

prev["APPROVED_CREDIT_RATIO"] = (
    prev["AMT_CREDIT"]
    .div(prev["AMT_APPLICATION"].replace(0, np.nan))
    .where(valid_approved)
)

# 4. 金额低于申请金额的标记
# 对无效申请保持NaN，不错误计入比例分母
prev["APPROVED_CREDIT_LT_APP"] = (
    (prev["APPROVED_CREDIT_RATIO"] < 0.99)
    .astype(float)
    .where(valid_approved)
)

# 5. 客户级金额聚合
amount_features = (
    prev.groupby("SK_ID_CURR")
    .agg(
        PREV_APP_AMT_MEAN=("APP_AMT_POSITIVE", "mean"),
        PREV_APP_AMT_MAX=("APP_AMT_POSITIVE", "max"),

        PREV_APPROVED_AMT_MEAN=("APPROVED_APP_AMT", "mean"),
        PREV_REFUSED_AMT_MEAN=("REFUSED_APP_AMT", "mean"),

        PREV_APPROVED_CREDIT_RATIO_MEAN=(
            "APPROVED_CREDIT_RATIO", "mean"
        ),
        PREV_APPROVED_CREDIT_LT_APP_RATE=(
            "APPROVED_CREDIT_LT_APP", "mean"
        )
    )
    .reset_index()
)

# 6. 合并到已有客户级特征
prev_customer = prev_customer.merge(
    amount_features,
    on="SK_ID_CURR",
    how="left",
    validate="one_to_one"
)

# 7. 验收
print("===== previous_application金额特征 =====")
print("数据维度：", prev_customer.shape)
print("重复客户ID：", prev_customer["SK_ID_CURR"].duplicated().sum())

print("\n===== 金额特征分布 =====")

print(
    prev_customer[
        [
            "PREV_APP_AMT_MEAN",
            "PREV_APPROVED_AMT_MEAN",
            "PREV_REFUSED_AMT_MEAN",
            "PREV_APPROVED_CREDIT_RATIO_MEAN",
            "PREV_APPROVED_CREDIT_LT_APP_RATE"
        ]
    ]
    .describe(percentiles=[0.5, 0.9, 0.99])
    .round(4)
    .to_string()
)
# %%
# %% EDA 05：历史拒绝率、近期申请次数与TARGET

import pandas as pd
import numpy as np

path = r"D:\Home Credit Default Risk"

# 1. 读取当前申请标签
app = pd.read_csv(
    path + r"\data\application_train.csv",
    usecols=["SK_ID_CURR", "TARGET"]
)

# 2. 左连接历史申请特征
analysis = app.merge(
    prev_customer[
        [
            "SK_ID_CURR",
            "PREV_APP_COUNT",
            "PREV_REFUSED_RATE",
            "PREV_APP_D090"
        ]
    ],
    on="SK_ID_CURR",
    how="left",
    validate="one_to_one"
)

# 3. 历史拒绝率分组
r = analysis["PREV_REFUSED_RATE"]

analysis["REFUSED_GROUP"] = np.select(
    [
        r == 0,
        (r > 0) & (r <= 0.25),
        (r > 0.25) & (r <= 0.5),
        r > 0.5
    ],
    [
        "0%",
        "0%-25%",
        "25%-50%",
        ">50%"
    ],
    default="无历史申请"
)

# 4. 近期申请次数分组
n = analysis["PREV_APP_D090"]

analysis["RECENT_APP_GROUP"] = np.select(
    [
        n == 0,
        n == 1,
        n == 2,
        n >= 3
    ],
    [
        "0次",
        "1次",
        "2次",
        "3次及以上"
    ],
    default="无历史申请"
)

# 5. 通用分组统计函数
def summarize_risk(data, group_col):
    result = (
        data.groupby(group_col)
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

print("===== 历史拒绝率与TARGET =====")
print(summarize_risk(analysis, "REFUSED_GROUP").to_string())

print("\n===== 近90天历史申请次数与TARGET =====")
print(summarize_risk(analysis, "RECENT_APP_GROUP").to_string())

print("\n===== 数据验收 =====")
print("合并后客户数：", len(analysis))
print("重复客户ID：", analysis["SK_ID_CURR"].duplicated().sum())
print("总体违约率：", f'{analysis["TARGET"].mean():.2%}')
# %%
# %% Feature Engineering 03：最终验收与保存

import pandas as pd

print("===== previous_application最终特征表 =====")

print("数据维度：", prev_customer.shape)

print("重复客户ID：", prev_customer["SK_ID_CURR"].duplicated().sum())

print("重复字段名：", prev_customer.columns.duplicated().sum())

print("是否包含TARGET：", "TARGET" in prev_customer.columns)

print("object类型字段数：", len(prev_customer.select_dtypes(include="object").columns))

print("\n===== 缺失率最高的10个特征 =====")

print(
    prev_customer.isna()
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .round(4)
    .to_string()
)

# 基础验收
assert prev_customer.shape == (338857, 21)
assert prev_customer["SK_ID_CURR"].is_unique
assert not prev_customer.columns.duplicated().any()
assert "TARGET" not in prev_customer.columns
assert len(prev_customer.select_dtypes(include="object").columns) == 0

# 保存
output_path = path + r"\features\previous_application_features.parquet"

prev_customer.to_parquet(
    output_path,
    index=False
)

# 重新读取检查
check = pd.read_parquet(output_path)

assert check.shape == prev_customer.shape
assert check["SK_ID_CURR"].is_unique

print("\n特征表保存并读取验证成功")
print("保存路径：", output_path)