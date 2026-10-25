import pandas as pd

# 读取当前贷款申请数据
df = pd.read_csv(r"D:\Home Credit Default Risk\data\application_train.csv")

# 查看数据规模
print("数据维度：", df.shape)

# 查看前五行
print(df.head())

# 查看标签分布
print(df["TARGET"].value_counts())
print(df["TARGET"].value_counts(normalize=True))

# 检查申请 ID 是否唯一
print("重复申请ID：", df["SK_ID_CURR"].duplicated().sum())

# 1. 计算每个字段的缺失数量和缺失比例
missing = pd.DataFrame({
    "缺失数量": df.isnull().sum(),
    "缺失比例": df.isnull().mean() * 100
})

# 2. 按缺失比例降序排列
missing = missing.sort_values(
    by="缺失比例",
    ascending=False
)

# 3. 查看缺失最严重的前20个字段
print("缺失率最高的20个字段：")
print(missing.head(20).round(2).to_string())

# 4. 统计不同缺失程度的字段数量
print("\n缺失率超过50%的字段数：", (missing["缺失比例"] > 50).sum())

print("缺失率超过30%的字段数：", (missing["缺失比例"] > 30).sum())

print("完全没有缺失的字段数：", (missing["缺失比例"] == 0).sum())


# 分析缺失值与坏样本率的关系
cols = [
    "EXT_SOURCE_1",
    "OWN_CAR_AGE",
    "COMMONAREA_AVG"
]

base_bad_rate = df["TARGET"].mean()

for col in cols:

    missing_flag = df[col].isna()

    result = (
        df.groupby(missing_flag)["TARGET"]
        .agg(["count", "sum", "mean"])
    )

    result.columns = [
        "样本数",
        "坏样本数",
        "坏样本率"
    ]

    result["Lift"] = (
        result["坏样本率"] / base_bad_rate
    )

    print(f"\n========== {col} ==========")
    print(result.round(4).to_string())

# 分析评分字段与坏样本率的关系
score_cols = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3"
]

for col in score_cols:
    print(f"\n========== {col} ==========")

    # 1. 查看基本统计信息
    print(df[col].describe())

    # 2. 只选择非缺失样本
    temp = df[[col, "TARGET"]].dropna().copy()

    # 3. 按评分数值分成10组
    temp["score_bin"] = pd.qcut(
        temp[col],
        q=10,
        duplicates="drop"
    )

    # 4. 计算每一组的坏样本率
    result = (
        temp.groupby("score_bin", observed=True)["TARGET"]
        .agg(["count", "mean"])
    )

    result.columns = ["样本数", "坏样本率"]

    # 5. 转换成百分比
    result["坏样本率"] = (
        result["坏样本率"] * 100
    )

    print("\n十分位分箱结果：")
    print(result.round(2).to_string())


from sklearn.metrics import roc_auc_score

# 计算评分字段的AUC和覆盖率
score_cols = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3"
]

for col in score_cols:
    temp = df[[col, "TARGET"]].dropna()

    auc = roc_auc_score(
        temp["TARGET"],
        -temp[col]
    )

    coverage = len(temp) / len(df)

    print(f"{col}: " f"AUC={auc:.4f}, " f"覆盖率={coverage:.2%}")

