# %% EDA 01：bureau_balance基础结构检查

import pandas as pd

path = r"D:\Home Credit Default Risk"

bb = pd.read_csv(
    path + r"\data\bureau_balance.csv"
)

print("===== 数据规模 =====")
print("数据维度：", bb.shape)

print("\n===== 字段类型 =====")
print(bb.dtypes.to_string())

print("\n===== 信用记录数量 =====")
print("SK_ID_BUREAU数量：", bb["SK_ID_BUREAU"].nunique())

print("\n===== 信用记录-月份唯一性 =====")
print("重复组合数：", bb.duplicated(["SK_ID_BUREAU", "MONTHS_BALANCE"]).sum())

print("\n===== 月份范围 =====")
print("最早月份：", bb["MONTHS_BALANCE"].min(), "最近月份：", bb["MONTHS_BALANCE"].max())

print("\n===== STATUS分布 =====")
print(bb["STATUS"].value_counts(dropna=False).to_string())

print("\n===== 字段缺失率 =====")
print((bb.isna().mean() * 100).round(2).to_string())
# %%
# %% EDA 02：STATUS风险分布

print("===== 完整STATUS分布 =====")
print(
    bb["STATUS"]
    .value_counts(dropna=False)
    .reindex(["0", "1", "2", "3", "4", "5", "C", "X"])
    .to_string()
)

print("\n===== 历史逾期状态检查 =====")

print("STATUS为1-5的月份：", bb["STATUS"].isin(["1", "2", "3", "4", "5"]).sum())

print("STATUS为2-5的月份：", bb["STATUS"].isin(["2", "3", "4", "5"]).sum())

print("STATUS为5的月份：", (bb["STATUS"] == "5").sum())

print("状态未知X的月份：", (bb["STATUS"] == "X").sum())

print("状态关闭C的月份：", (bb["STATUS"] == "C").sum())

print("\n===== 有历史逾期的信用记录数 =====")

print(bb.loc[bb["STATUS"].isin(["1", "2", "3", "4", "5"]), "SK_ID_BUREAU"].nunique())
# %%
# %% Feature Engineering 01：信用记录级历史逾期特征

import pandas as pd
import numpy as np

# 1. 状态标记
bb["IS_KNOWN"] = bb["STATUS"].isin(
    ["0", "1", "2", "3", "4", "5"]
).astype("int8")

bb["IS_OVERDUE"] = bb["STATUS"].isin(
    ["1", "2", "3", "4", "5"]
).astype("int8")

# C和X不参与最大逾期等级计算
bb["STATUS_NUM"] = pd.to_numeric(
    bb["STATUS"], errors="coerce"
)

# 2. 按信用记录聚合
bb_features = (
    bb.groupby("SK_ID_BUREAU", sort=False)
    .agg(
        BB_OBS_MONTHS=("MONTHS_BALANCE", "size"),
        BB_KNOWN_MONTHS=("IS_KNOWN", "sum"),
        BB_OVERDUE_MONTHS=("IS_OVERDUE", "sum"),
        BB_MAX_STATUS=("STATUS_NUM", "max")
    )
    .reset_index()
)

# 3. 逾期月份比例
bb_features["BB_OVERDUE_RATE"] = (
    bb_features["BB_OVERDUE_MONTHS"]
    / bb_features["BB_KNOWN_MONTHS"].replace(0, np.nan)
)

# 4. 曾经逾期标记
bb_features["BB_EVER_OVERDUE"] = (
    bb_features["BB_OVERDUE_MONTHS"] > 0
).astype("int8")

# 5. 基础验收
print("===== 信用记录级特征 =====")
print("数据维度：", bb_features.shape)
print("重复信用记录ID：", bb_features["SK_ID_BUREAU"].duplicated().sum())

print("\n===== 核心特征分布 =====")
print(
    bb_features[
        [
            "BB_OBS_MONTHS",
            "BB_KNOWN_MONTHS",
            "BB_OVERDUE_MONTHS",
            "BB_MAX_STATUS",
            "BB_OVERDUE_RATE"
        ]
    ]
    .describe(percentiles=[0.5, 0.9, 0.99])
    .round(4)
    .to_string()
)

print("\n===== 历史逾期验收 =====")
print("曾出现逾期的信用记录：", bb_features["BB_EVER_OVERDUE"].sum())

print("逾期比例缺失的信用记录：", bb_features["BB_OVERDUE_RATE"].isna().sum())
# %%
# %% Feature Engineering 02：最近6个月 vs 前6个月逾期趋势

import numpy as np
import pandas as pd

# 1. 只使用申请月之前的最近12个月
bb12 = bb.loc[
    bb["MONTHS_BALANCE"].between(-12, -1),
    [
        "SK_ID_BUREAU",
        "MONTHS_BALANCE",
        "IS_KNOWN",
        "IS_OVERDUE"
    ]
].copy()

# 2. 划分时间窗口
bb12["WINDOW"] = np.where(
    bb12["MONTHS_BALANCE"] >= -6,
    "RECENT6",
    "PREV6"
)

# 3. 按信用记录与时间窗口聚合
window_stats = (
    bb12.groupby(["SK_ID_BUREAU", "WINDOW"])
    .agg(
        KNOWN=("IS_KNOWN", "sum"),
        OVERDUE=("IS_OVERDUE", "sum")
    )
    .unstack("WINDOW")
)

# 4. 展开列名
window_stats.columns = [
    f"BB_{metric}_{window}"
    for metric, window in window_stats.columns
]

window_stats = window_stats.reset_index()

# 5. 缺失窗口：观察月份数计为0
for col in [
    "BB_KNOWN_RECENT6",
    "BB_KNOWN_PREV6",
    "BB_OVERDUE_RECENT6",
    "BB_OVERDUE_PREV6"
]:
    window_stats[col] = (
        window_stats[col].fillna(0).astype(int)
    )

# 6. 计算各窗口逾期比例
window_stats["BB_OVERDUE_RATE_RECENT6"] = (
    window_stats["BB_OVERDUE_RECENT6"]
    / window_stats["BB_KNOWN_RECENT6"].replace(0, np.nan)
)

window_stats["BB_OVERDUE_RATE_PREV6"] = (
    window_stats["BB_OVERDUE_PREV6"]
    / window_stats["BB_KNOWN_PREV6"].replace(0, np.nan)
)

# 7. 两窗口都至少有2个明确状态月份
eligible = (
    (window_stats["BB_KNOWN_RECENT6"] >= 2)
    & (window_stats["BB_KNOWN_PREV6"] >= 2)
)

window_stats["BB_TREND_ELIGIBLE"] = (
    eligible.astype("int8")
)

# 8. 计算逾期比例变化
window_stats["BB_OVERDUE_DELTA_6V6"] = (
    window_stats["BB_OVERDUE_RATE_RECENT6"]
    - window_stats["BB_OVERDUE_RATE_PREV6"]
).where(eligible)

# 9. 合并到信用记录级特征表
bb_features = bb_features.merge(
    window_stats,
    on="SK_ID_BUREAU",
    how="left",
    validate="one_to_one"
)

# 10. 检查
print("===== bureau_balance近期逾期趋势 =====")
print("特征表维度：", bb_features.shape)
print("重复信用记录ID：", bb_features["SK_ID_BUREAU"].duplicated().sum())

delta = bb_features["BB_OVERDUE_DELTA_6V6"]

print("\n可以计算趋势的信用记录：", delta.notna().sum())

print("\n===== 趋势方向 =====")
print("近期恶化：", (delta > 0).sum())
print("近期改善：", (delta < 0).sum())
print("保持不变：", (delta == 0).sum())
print("无法计算：", delta.isna().sum())

print("\n===== 变化量描述统计 =====")
print(delta.describe(percentiles=[0.1, 0.5, 0.9]).round(4).to_string())


# %% Feature Engineering 03：聚合、验收并保存客户级特征

import pandas as pd

path = r"D:\Home Credit Default Risk"

# 1. 读取bureau主表中的两个ID
bureau_ids = pd.read_csv(
    path + r"\data\bureau.csv",
    usecols=["SK_ID_CURR", "SK_ID_BUREAU"]
)

# 2. 通过SK_ID_BUREAU关联客户
bur_bb = bureau_ids.merge(
    bb_features,
    on="SK_ID_BUREAU",
    how="left",
    validate="one_to_one"
)

print("===== 第一步：合并检查 =====")
print("合并后行数：", len(bur_bb))
print("无月度历史的信用记录：", bur_bb["BB_OBS_MONTHS"].isna().sum())

# 3. 仅使用真正有月度历史的信用记录
has_history = bur_bb.loc[
    bur_bb["BB_OBS_MONTHS"].notna()
].copy()

# 4. 构造信用记录级近期恶化标记
has_history["BB_WORSEN_FLAG"] = (
    has_history["BB_OVERDUE_DELTA_6V6"] > 0
).astype("int8")

# 5. 按客户聚合
bb_customer = (
    has_history.groupby("SK_ID_CURR")
    .agg(
        BUR_BB_RECORD_COUNT=(
            "SK_ID_BUREAU", "size"
        ),
        BUR_BB_EVER_OVERDUE_COUNT=(
            "BB_EVER_OVERDUE", "sum"
        ),
        BUR_BB_MAX_STATUS=(
            "BB_MAX_STATUS", "max"
        ),
        BUR_BB_OVERDUE_RATE_MEAN=(
            "BB_OVERDUE_RATE", "mean"
        ),
        BUR_BB_TREND_VALID_COUNT=(
            "BB_OVERDUE_DELTA_6V6", "count"
        ),
        BUR_BB_WORSEN_COUNT=(
            "BB_WORSEN_FLAG", "sum"
        ),
        BUR_BB_DELTA_MEAN=(
            "BB_OVERDUE_DELTA_6V6", "mean"
        )
    )
    .reset_index()
)

# 6. 客户级验收
print("\n===== 第二步：客户级验收 =====")

print("数据维度：", bb_customer.shape)

print("重复客户ID：", bb_customer["SK_ID_CURR"].duplicated().sum())

print("有月度历史的客户：", bb_customer["SK_ID_CURR"].nunique())

print("曾出现逾期的客户：", (bb_customer["BUR_BB_EVER_OVERDUE_COUNT"] > 0).sum())

print("近期存在恶化记录的客户：", (bb_customer["BUR_BB_WORSEN_COUNT"] > 0).sum())

# 7. 保存
bb_customer.to_parquet(
    path + r"\features\bureau_balance_features.parquet",
    index=False
)

print("\n保存成功：bureau_balance_features.parquet")
# %%
