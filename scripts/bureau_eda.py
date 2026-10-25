# %% EDA 01：bureau基础结构检查

import pandas as pd
import numpy as np

path = r"D:\Home Credit Default Risk"

# 1. 读取外部征信主表
bureau = pd.read_csv(
    path + r"\data\bureau.csv"
)

# 2. 数据维度
print("===== 数据规模 =====")
print("数据维度：", bureau.shape)

# 3. 字段名称与数据类型
print("\n===== 字段及类型 =====")
print(bureau.dtypes.to_string())

# 4. 前5条记录
print("\n===== 前5行 =====")
print(bureau.head().to_string())

# 5. 客户及外部信用记录数量
print("\n===== ID唯一性 =====")
print("客户数量：", bureau["SK_ID_CURR"].nunique())

print("外部信用记录数量：", bureau["SK_ID_BUREAU"].nunique())

print("重复SK_ID_BUREAU：", bureau["SK_ID_BUREAU"].duplicated().sum())

# 6. 缺失率
missing = (
    bureau.isna()
    .mean()
    .sort_values(ascending=False)
)

print("\n===== 缺失率最高的15列 =====")
print((missing.head(15) * 100).round(2).to_string())

print("\n缺失率超过50%的列数：", (missing > 0.5).sum())

print("完全没有缺失的列数：", (missing == 0).sum())
# %%

print("客户数量：", bureau["SK_ID_CURR"].nunique())
print("外部信用记录数量：", bureau["SK_ID_BUREAU"].nunique())
print("重复信用记录ID：", bureau["SK_ID_BUREAU"].duplicated().sum())
# %%
# %% EDA 02：外部信用状态和贷款类型

print("===== 信用记录状态 =====")
print(bureau["CREDIT_ACTIVE"].value_counts(dropna=False).to_string())

print("\n===== 信贷类型（前10种）=====")
print(bureau["CREDIT_TYPE"].value_counts(dropna=False).head(10).to_string())

print("\n===== 信用状态与贷款类型交叉表 =====")
print(pd.crosstab(bureau["CREDIT_TYPE"], bureau["CREDIT_ACTIVE"]).head(10).to_string())
# %%
# %% EDA 03：外部负债与逾期分布

cols = [
    "AMT_CREDIT_SUM",
    "AMT_CREDIT_SUM_DEBT",
    "CREDIT_DAY_OVERDUE",
    "AMT_CREDIT_SUM_OVERDUE"
]

print("===== 金额和逾期分布 =====")

print(bureau[cols].describe(percentiles=[0.5, 0.9, 0.99]).round(2).to_string())

print("\n===== 逾期与债务检查 =====")

print("债务余额缺失：", bureau["AMT_CREDIT_SUM_DEBT"].isna().sum())

print("债务余额小于0：", (bureau["AMT_CREDIT_SUM_DEBT"] < 0).sum())

print("逾期天数大于0：", (bureau["CREDIT_DAY_OVERDUE"] > 0).sum())

print("逾期金额大于0：", (bureau["AMT_CREDIT_SUM_OVERDUE"] > 0).sum())
# %%
# %% EDA 04：信用状态与债务质量分析

# 1. 按信用状态检查债务余额
bureau["DEBT_MISSING"] = (
    bureau["AMT_CREDIT_SUM_DEBT"].isna()
)

bureau["DEBT_NEGATIVE"] = (
    bureau["AMT_CREDIT_SUM_DEBT"] < 0
)

bureau["DEBT_POSITIVE"] = (
    bureau["AMT_CREDIT_SUM_DEBT"] > 0
)

status_debt = (
    bureau.groupby("CREDIT_ACTIVE")
    .agg(
        RECORD_COUNT=("SK_ID_BUREAU", "size"),
        DEBT_MISSING_RATE=("DEBT_MISSING", "mean"),
        DEBT_NEGATIVE_RATE=("DEBT_NEGATIVE", "mean"),
        DEBT_POSITIVE_RATE=("DEBT_POSITIVE", "mean"),
        DEBT_MEDIAN=("AMT_CREDIT_SUM_DEBT", "median")
    )
)

# 百分比格式
rate_cols = [
    "DEBT_MISSING_RATE",
    "DEBT_NEGATIVE_RATE",
    "DEBT_POSITIVE_RATE"
]

status_debt[rate_cols] = (
    status_debt[rate_cols] * 100
).round(2)

print("===== 信用状态与债务质量 =====")
print(status_debt.round(2).to_string())
# %%
# %% Feature Engineering 01：外部信用记录与负债特征

import pandas as pd
import numpy as np

# 1. 账户状态标记
bureau["IS_ACTIVE"] = (
    bureau["CREDIT_ACTIVE"] == "Active"
).astype("int8")

bureau["IS_CLOSED"] = (
    bureau["CREDIT_ACTIVE"] == "Closed"
).astype("int8")

# 2. 债务异常标记
bureau["DEBT_NEGATIVE"] = (
    bureau["AMT_CREDIT_SUM_DEBT"] < 0
).astype("int8")

# 3. 构造活跃账户负债
# 负值不直接参与正向负债合计
# 非Active账户和缺失债务暂时设为NaN
bureau["ACTIVE_DEBT"] = np.where(
    (bureau["CREDIT_ACTIVE"] == "Active")
    & bureau["AMT_CREDIT_SUM_DEBT"].notna(),
    bureau["AMT_CREDIT_SUM_DEBT"].clip(lower=0),
    np.nan
)

# 4. 活跃账户债务缺失标记
bureau["ACTIVE_DEBT_MISSING"] = (
    (bureau["CREDIT_ACTIVE"] == "Active")
    & bureau["AMT_CREDIT_SUM_DEBT"].isna()
).astype("int8")

# 5. 按客户聚合
bur_customer = (
    bureau.groupby("SK_ID_CURR")
    .agg(
        BUR_CREDIT_COUNT=("SK_ID_BUREAU", "size"),
        BUR_ACTIVE_COUNT=("IS_ACTIVE", "sum"),
        BUR_CLOSED_COUNT=("IS_CLOSED", "sum"),
        BUR_ACTIVE_DEBT_SUM=("ACTIVE_DEBT", "sum"),
        BUR_ACTIVE_DEBT_MAX=("ACTIVE_DEBT", "max"),
        BUR_ACTIVE_DEBT_MISSING_COUNT=(
            "ACTIVE_DEBT_MISSING", "sum"
        ),
        BUR_NEGATIVE_DEBT_COUNT=("DEBT_NEGATIVE", "sum")
    )
    .reset_index()
)

# 6. 活跃账户占比
bur_customer["BUR_ACTIVE_RATIO"] = (
    bur_customer["BUR_ACTIVE_COUNT"]
    / bur_customer["BUR_CREDIT_COUNT"]
)

# 7. 活跃账户债务缺失率
bur_customer["BUR_ACTIVE_DEBT_MISSING_RATE"] = (
    bur_customer["BUR_ACTIVE_DEBT_MISSING_COUNT"]
    / bur_customer["BUR_ACTIVE_COUNT"].replace(0, np.nan)
)

# 8. 没有任何已知Active余额的客户，
# 不应该简单认为其外部债务为0
known_active_count = (
    bureau.groupby("SK_ID_CURR")["ACTIVE_DEBT"]
    .count()
)

bur_customer = bur_customer.merge(
    known_active_count.rename("BUR_ACTIVE_DEBT_KNOWN_COUNT"),
    on="SK_ID_CURR",
    how="left",
    validate="one_to_one"
)

bur_customer.loc[
    bur_customer["BUR_ACTIVE_DEBT_KNOWN_COUNT"] == 0,
    ["BUR_ACTIVE_DEBT_SUM", "BUR_ACTIVE_DEBT_MAX"]
] = np.nan

# 9. 结果检查
print("===== bureau客户级第一批特征 =====")

print("数据维度：", bur_customer.shape)
print("重复客户ID：", bur_customer["SK_ID_CURR"].duplicated().sum())

print("\n===== 核心特征分布 =====")
print(
    bur_customer[
        [
            "BUR_CREDIT_COUNT",
            "BUR_ACTIVE_COUNT",
            "BUR_ACTIVE_RATIO",
            "BUR_ACTIVE_DEBT_SUM",
            "BUR_ACTIVE_DEBT_MISSING_RATE"
        ]
    ]
    .describe(percentiles=[0.5, 0.9, 0.99])
    .round(2)
    .to_string()
)
# %%
# %% EDA 05：外部信用记录时间分布

cols = [
    "DAYS_CREDIT",
    "DAYS_CREDIT_ENDDATE",
    "DAYS_ENDDATE_FACT",
    "DAYS_CREDIT_UPDATE"
]

print("===== 信用记录时间分布 =====")

print(bureau[cols].describe(percentiles=[0.5, 0.9, 0.99]).round(2).to_string())

print("\n===== 近期信用记录数量 =====")

print("最近90天：", bureau["DAYS_CREDIT"].between(-90, -1).sum())

print("最近180天：", bureau["DAYS_CREDIT"].between(-180, -1).sum())

print("最近365天：", bureau["DAYS_CREDIT"].between(-365, -1).sum())

print("\n===== 时间异常检查 =====")

print("DAYS_CREDIT大于0：", (bureau["DAYS_CREDIT"] > 0).sum())

print("DAYS_CREDIT_ENDDATE大于0：", (bureau["DAYS_CREDIT_ENDDATE"] > 0).sum())

print("DAYS_CREDIT_UPDATE大于0：", (bureau["DAYS_CREDIT_UPDATE"] > 0).sum())
# %%
# %% Feature Engineering 02：近期外部信贷活动特征

# 1. 暂时排除更新时间在申请日之后的记录
bur_time = bureau.loc[
    bureau["DAYS_CREDIT_UPDATE"] <= 0
].copy()

# 2. 构造时间窗口标记
bur_time["NEW_D090"] = (
    bur_time["DAYS_CREDIT"].between(-90, -1)
).astype("int8")

bur_time["NEW_PREV090"] = (
    bur_time["DAYS_CREDIT"].between(-180, -91)
).astype("int8")

bur_time["NEW_D365"] = (
    bur_time["DAYS_CREDIT"].between(-365, -1)
).astype("int8")

# 3. 聚合到客户级
bur_recent = (
    bur_time.groupby("SK_ID_CURR")
    .agg(
        BUR_CREDIT_NEW_D090=("NEW_D090", "sum"),
        BUR_CREDIT_NEW_PREV090=("NEW_PREV090", "sum"),
        BUR_CREDIT_NEW_D365=("NEW_D365", "sum"),
        BUR_LAST_CREDIT_DAY=("DAYS_CREDIT", "max")
    )
    .reset_index()
)

# 4. 计算近期与前期信贷活动变化
bur_recent["BUR_CREDIT_NEW_DELTA_90V90"] = (
    bur_recent["BUR_CREDIT_NEW_D090"]
    - bur_recent["BUR_CREDIT_NEW_PREV090"]
)

# 5. 距离最近一次信用活动的天数
bur_recent["BUR_DAYS_SINCE_LAST_CREDIT"] = (
    -bur_recent["BUR_LAST_CREDIT_DAY"]
)

bur_recent = bur_recent.drop(
    columns="BUR_LAST_CREDIT_DAY"
)

# 6. 合并到之前的客户级特征表
bur_customer = bur_customer.merge(
    bur_recent,
    on="SK_ID_CURR",
    how="left",
    validate="one_to_one"
)

# 7. 验收
print("===== 近期信贷活动特征 =====")
print("客户级数据维度：", bur_customer.shape)
print("重复客户ID：", bur_customer["SK_ID_CURR"].duplicated().sum())

print("\n===== 最近信用活动分布 =====")
print(
    bur_customer[
        [
            "BUR_CREDIT_NEW_D090",
            "BUR_CREDIT_NEW_PREV090",
            "BUR_CREDIT_NEW_D365",
            "BUR_CREDIT_NEW_DELTA_90V90",
            "BUR_DAYS_SINCE_LAST_CREDIT"
        ]
    ]
    .describe(percentiles=[0.5, 0.9, 0.99])
    .round(2)
    .to_string()
)

print("\n===== 活动变化方向 =====")

delta = bur_customer["BUR_CREDIT_NEW_DELTA_90V90"]

print("近期增加：", (delta > 0).sum())
print("近期减少：", (delta < 0).sum())
print("近期不变：", (delta == 0).sum())
# %%
# %% EDA 06：近期外部信贷活动与TARGET

import pandas as pd
import numpy as np

# 1. 读取申请主表标签
app = pd.read_csv(
    path + r"\data\application_train.csv",
    usecols=["SK_ID_CURR", "TARGET"]
)

# 2. 仅关联需要研究的特征
analysis = app.merge(
    bur_customer[
        ["SK_ID_CURR", "BUR_CREDIT_NEW_D090"]
    ],
    on="SK_ID_CURR",
    how="left",
    validate="one_to_one"
)

# 3. 区分无征信记录与有征信但无近期活动
n = analysis["BUR_CREDIT_NEW_D090"]

analysis["RECENT_CREDIT_GROUP"] = np.select(
    [
        n == 0,
        n == 1,
        n == 2,
        n >= 3
    ],
    [
        "0条",
        "1条",
        "2条",
        "3条及以上"
    ],
    default="无可用记录"
)

# 4. 统计各组违约率
result = (
    analysis.groupby("RECENT_CREDIT_GROUP")
    .agg(
        CUSTOMER_COUNT=("SK_ID_CURR", "size"),
        BAD_COUNT=("TARGET", "sum"),
        BAD_RATE=("TARGET", "mean")
    )
)

result["BAD_RATE"] = (
    result["BAD_RATE"] * 100
).round(2)

print("===== 近期外部信用活动与TARGET =====")
print(result.reindex(["0条", "1条", "2条", "3条及以上", "无可用记录"]).to_string())

print("\n===== 数据检查 =====")
print("申请客户数：", len(app))
print("合并后客户数：", len(analysis))
print("重复客户ID：", analysis["SK_ID_CURR"].duplicated().sum())
# %%
# %% Feature Engineering 03：合并bureau与bureau_balance客户特征

import pandas as pd

path = r"D:\Home Credit Default Risk"

# 1. 读取已经保存的月度征信客户级特征
bb_customer = pd.read_parquet(
    path + r"\features\bureau_balance_features.parquet"
)

# 2. 检查两张表
print("bureau客户特征：", bur_customer.shape)
print("bureau_balance客户特征：", bb_customer.shape)

assert bur_customer["SK_ID_CURR"].is_unique
assert bb_customer["SK_ID_CURR"].is_unique

# 3. 左连接，保留全部bureau客户
# 防止重复运行时出现_x、_y后缀
new_cols = bb_customer.columns.drop("SK_ID_CURR")

bur_customer = bur_customer.drop(
    columns=new_cols,
    errors="ignore"
)

bur_customer = bur_customer.merge(
    bb_customer,
    on="SK_ID_CURR",
    how="left",
    validate="one_to_one"
)

# 4. 对没有月度历史的客户，计数型特征填0
count_cols = [
    "BUR_BB_RECORD_COUNT",
    "BUR_BB_EVER_OVERDUE_COUNT",
    "BUR_BB_TREND_VALID_COUNT",
    "BUR_BB_WORSEN_COUNT"
]

bur_customer[count_cols] = (
    bur_customer[count_cols]
    .fillna(0)
    .astype(int)
)

# 注意：最大逾期等级、逾期比例、趋势变化仍保留NaN

# 5. 验收
print("\n===== 完整外部征信客户特征表 =====")
print("数据维度：", bur_customer.shape)
print("重复客户ID：", bur_customer["SK_ID_CURR"].duplicated().sum())

print("无月度历史的客户：", (bur_customer["BUR_BB_RECORD_COUNT"] == 0).sum())

print("曾出现历史逾期的客户：", (bur_customer["BUR_BB_EVER_OVERDUE_COUNT"] > 0).sum())

print("存在近期逾期恶化记录的客户：", (bur_customer["BUR_BB_WORSEN_COUNT"] > 0).sum())
# %%
# %% EDA 07：外部征信历史逾期与TARGET

import pandas as pd
import numpy as np

path = r"D:\Home Credit Default Risk"

app = pd.read_csv(
    path + r"\data\application_train.csv",
    usecols=["SK_ID_CURR", "TARGET"]
)

# bur_customer 已经是305811行、23列
risk_cols = [
    "SK_ID_CURR",
    "BUR_BB_RECORD_COUNT",
    "BUR_BB_EVER_OVERDUE_COUNT",
    "BUR_BB_WORSEN_COUNT"
]

risk = app.merge(
    bur_customer[risk_cols],
    on="SK_ID_CURR",
    how="left",
    validate="one_to_one"
)

# 对缺少bureau或bureau_balance历史的客户单独分类
history = risk["BUR_BB_RECORD_COUNT"]
ever = risk["BUR_BB_EVER_OVERDUE_COUNT"]
worsen = risk["BUR_BB_WORSEN_COUNT"]

risk["BB_RISK_GROUP"] = np.select(
    [
        history.isna() | (history == 0),
        ever == 0,
        worsen > 0,
        ever > 0
    ],
    [
        "无月度历史",
        "有历史未观察到逾期",
        "曾逾期且近期恶化",
        "曾逾期但近期无恶化记录"
    ],
    default="其他"
)

result = (
    risk.groupby("BB_RISK_GROUP")
    .agg(
        CUSTOMER_COUNT=("SK_ID_CURR", "size"),
        BAD_COUNT=("TARGET", "sum"),
        BAD_RATE=("TARGET", "mean")
    )
)

result["BAD_RATE"] = (
    result["BAD_RATE"] * 100
).round(2)

print("===== bureau_balance历史逾期与TARGET =====")
print(result.to_string())

print("\n===== 数据检查 =====")
print("合并后客户数：", len(risk))
print("重复客户ID：", risk["SK_ID_CURR"].duplicated().sum())
print("分组总客户数：", result["CUSTOMER_COUNT"].sum())
print("总体违约率：", f'{risk["TARGET"].mean():.2%}')
# %%
# %% Feature Engineering 04：补充静态逾期与信贷类型特征

import pandas as pd

path = r"D:\Home Credit Default Risk"

cols = [
    "SK_ID_CURR",
    "SK_ID_BUREAU",
    "CREDIT_TYPE",
    "CREDIT_DAY_OVERDUE",
    "AMT_CREDIT_SUM_OVERDUE",
    "DAYS_CREDIT_UPDATE"
]

bur_static = pd.read_csv(
    path + r"\data\bureau.csv",
    usecols=cols
)

# 排除更新时间晚于当前申请日的记录
bur_static = bur_static.loc[
    bur_static["DAYS_CREDIT_UPDATE"] <= 0
].copy()

# 1. 信用类型标记
bur_static["IS_CONSUMER"] = (
    bur_static["CREDIT_TYPE"] == "Consumer credit"
).astype(int)

bur_static["IS_CARD"] = (
    bur_static["CREDIT_TYPE"] == "Credit card"
).astype(int)

# 2. 当前逾期标记
bur_static["IS_CURRENT_OVERDUE"] = (
    bur_static["CREDIT_DAY_OVERDUE"] > 0
).astype(int)

# 3. 聚合到客户级
static_extra = (
    bur_static.groupby("SK_ID_CURR")
    .agg(
        BUR_CURRENT_OVERDUE_COUNT=(
            "IS_CURRENT_OVERDUE", "sum"
        ),
        BUR_MAX_CURRENT_DPD=(
            "CREDIT_DAY_OVERDUE", "max"
        ),
        BUR_CURRENT_OVERDUE_AMT_SUM=(
            "AMT_CREDIT_SUM_OVERDUE", "sum"
        ),
        BUR_CONSUMER_COUNT=(
            "IS_CONSUMER", "sum"
        ),
        BUR_CREDIT_CARD_COUNT=(
            "IS_CARD", "sum"
        )
    )
    .reset_index()
)

# 4. 合并到现有客户级特征表
# 避免重复运行产生 _x、_y
extra_cols = static_extra.columns.drop("SK_ID_CURR")

bur_customer = bur_customer.drop(
    columns=extra_cols,
    errors="ignore"
)

bur_customer = bur_customer.merge(
    static_extra,
    on="SK_ID_CURR",
    how="left",
    validate="one_to_one"
)

print("===== bureau补充特征验收 =====")
print("数据维度：", bur_customer.shape)
print("重复客户ID：", bur_customer["SK_ID_CURR"].duplicated().sum())

print("\n===== 静态特征分布 =====")
print(bur_customer[extra_cols].describe().round(2).to_string())
# %%
# %% Feature Engineering 05：最终验收与保存

import pandas as pd

print("===== bureau最终客户级特征表 =====")

print("数据维度：", bur_customer.shape)

print("重复客户ID：", bur_customer["SK_ID_CURR"].duplicated().sum())

print("重复字段名：", bur_customer.columns.duplicated().sum())

print("是否包含TARGET：", "TARGET" in bur_customer.columns)

print("object类型字段数：", len(bur_customer.select_dtypes(include="object").columns))

print("\n===== 缺失率最高的10个特征 =====")

print(
    bur_customer.isna()
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .round(4)
    .to_string()
)

# 基础质量检查
assert bur_customer.shape == (305811, 28)
assert bur_customer["SK_ID_CURR"].is_unique
assert not bur_customer.columns.duplicated().any()
assert "TARGET" not in bur_customer.columns

# 保存客户级候选特征
output_path = path + r"\features\bureau_features.parquet"

bur_customer.to_parquet(
    output_path,
    index=False
)

# 重新读取，验证文件可以正常使用
check = pd.read_parquet(output_path)

assert check.shape == bur_customer.shape
assert check["SK_ID_CURR"].is_unique

print("\n最终特征表保存并读取验证成功")
print("保存路径：", output_path)
# %%
