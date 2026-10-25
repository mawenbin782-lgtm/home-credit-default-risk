#%%
import pandas as pd

# ==========================================
# EDA 01：分期还款数据结构检查
# ==========================================

# 1. 读取数据
ins = pd.read_csv(
    r"D:\Home Credit Default Risk\data\installments_payments.csv"
)

# 2. 基础信息
print("===== 数据基本信息 =====")
print("数据维度：", ins.shape)

print("\n字段名称：")
print(ins.columns.tolist())

print("\n前5行数据：")
print(ins.head().to_string())

# 3. 客户和历史贷款数量
print("\n===== 客户与贷款数量 =====")
print("客户数量：", ins["SK_ID_CURR"].nunique())
print("历史贷款数量：", ins["SK_ID_PREV"].nunique())

# 4. 字段缺失率
print("\n===== 字段缺失率 =====")
missing = (
    ins.isna().mean() * 100
).sort_values(ascending=False)

print(missing.round(2).to_string())

# 5. 检查同一期是否存在多条记录
print("\n===== 分期期次重复检查 =====")

installment_key = [
    "SK_ID_PREV",
    "NUM_INSTALMENT_VERSION",
    "NUM_INSTALMENT_NUMBER"
]

duplicate_count = ins.duplicated(
    subset=installment_key
).sum()

unique_installments = (
    ins[installment_key]
    .drop_duplicates()
    .shape[0]
)

print("不同分期期次组合数：", unique_installments)
print("同一分期期次的额外记录数：", duplicate_count)

# 6. 检查日期范围
print("\n===== 日期范围 =====")
print("计划还款日范围：", ins["DAYS_INSTALMENT"].min(), ins["DAYS_INSTALMENT"].max())

print("实际付款日范围：", ins["DAYS_ENTRY_PAYMENT"].min(), ins["DAYS_ENTRY_PAYMENT"].max())
# %%
installment_key = [
    "SK_ID_PREV",
    "NUM_INSTALMENT_VERSION",
    "NUM_INSTALMENT_NUMBER"
]

# 不同分期组合数量
unique_count = (
    ins[installment_key]
    .drop_duplicates()
    .shape[0]
)

# 同一分期组合的额外记录数量
duplicate_count = ins.duplicated(
    subset=installment_key
).sum()

# 每个分期组合对应多少条记录
payment_counts = (
    ins.groupby(installment_key)
    .size()
)

print("不同分期组合数量：", unique_count)
print("同一分期的额外记录数：", duplicate_count)

print("\n每期付款记录数量分布：")
print(payment_counts.value_counts().head(10).to_string())

print("\n对应多条记录的分期组合数：", (payment_counts > 1).sum())

# %% EDA 02：同一期多次付款分析

installment_key = [
    "SK_ID_PREV",
    "NUM_INSTALMENT_VERSION",
    "NUM_INSTALMENT_NUMBER"
]

# 1. 按分期组合统计
installment_check = (
    ins.groupby(installment_key)
    .agg(
        记录数=("SK_ID_CURR", "size"),
        应还日期种类数=("DAYS_INSTALMENT", "nunique"),
        应还金额种类数=("AMT_INSTALMENT", "nunique"),
        应还金额=("AMT_INSTALMENT", "first"),
        实际付款总额=("AMT_PAYMENT", "sum")
    )
)

# 2. 只保留对应多条付款记录的分期组合
multi_payment = installment_check[
    installment_check["记录数"] > 1
].copy()

print("===== 多记录分期组合检查 =====")

print("多记录分期组合数：", len(multi_payment))

print("应还日期不一致的组合数：", (multi_payment["应还日期种类数"] > 1).sum())

print("应还金额不一致的组合数：", (multi_payment["应还金额种类数"] > 1).sum())

# 3. 检查实际付款总额与应还金额的关系
multi_payment["付款覆盖率"] = (
    multi_payment["实际付款总额"] /
    multi_payment["应还金额"].replace(0, float("nan"))
)

print("\n===== 付款覆盖率描述统计 =====")

print(multi_payment["付款覆盖率"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99]))

# 4. 查看前5个多记录分期组合
print("\n===== 多记录分期组合示例 =====")

print(multi_payment.head().to_string())

# %% EDA 03：付款覆盖率异常分析

ratio = multi_payment["付款覆盖率"]

print("===== 付款覆盖率分组 =====")

print("覆盖率缺失：", ratio.isna().sum())
print("覆盖率低于90%：", (ratio < 0.9).sum())
print("覆盖率在90%-110%之间：", ratio.between(0.9, 1.1).sum())
print("覆盖率超过110%：", (ratio > 1.1).sum())
print("覆盖率超过200%：", (ratio > 2).sum())

# 检查应还金额为0的组合
print("\n应还金额为0：", (multi_payment["应还金额"] == 0).sum())

# 查看覆盖率最高的10个组合
print("\n===== 覆盖率最高的10个组合 =====")

print(
    multi_payment.nlargest(
        10, "付款覆盖率"
    )[
        ["记录数", "应还金额", "实际付款总额", "付款覆盖率"]
    ].to_string()
)

# 查看覆盖率最低的10个有效组合
print("\n===== 覆盖率最低的10个组合 =====")

print(
    multi_payment.nsmallest(
        10, "付款覆盖率"
    )[
        ["记录数", "应还金额", "实际付款总额", "付款覆盖率"]
    ].to_string()
)

# %% EDA 04：查看单期分次付款过程

example = ins.loc[
    (ins["SK_ID_PREV"] == 1000005) &
    (ins["NUM_INSTALMENT_VERSION"] == 1.0) &
    (ins["NUM_INSTALMENT_NUMBER"] == 9)
].copy()

# 按实际付款日期排序
example = example.sort_values("DAYS_ENTRY_PAYMENT")

# 计算累计支付金额
example["累计付款金额"] = example["AMT_PAYMENT"].cumsum()

# 查看结果
print(example[
    [
        "SK_ID_PREV",
        "NUM_INSTALMENT_NUMBER",
        "DAYS_INSTALMENT",
        "DAYS_ENTRY_PAYMENT",
        "AMT_INSTALMENT",
        "AMT_PAYMENT",
        "累计付款金额"
    ]
].to_string(index=False))
# %%
