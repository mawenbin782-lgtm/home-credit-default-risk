# -*- coding: utf-8 -*-
"""
Home Credit 申请评分卡（Credit Scorecard）
========================================
用「申请时点可得」的特征（application_train.csv 主表），
走标准评分卡流程：分箱 → WOE/IV → 逻辑回归 → 分数刻度化 → KS/AUC + 通过率-坏账率权衡。

关键防泄漏点：分箱切点、WOE、IV 全部只在训练集上学习，再套用到验证集。
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

RANDOM = 42
# 项目根目录：本文件在 scorecard/ 下，往上一级就是项目根
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "application_train.csv")

# ----------------------------------------------------------------------
# 0. 读数据 + 选「申请时点」特征
# ----------------------------------------------------------------------
df = pd.read_csv(DATA)

continuous_feats = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "CNT_CHILDREN", "CNT_FAM_MEMBERS",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "REGION_POPULATION_RELATIVE", "AMT_REQ_CREDIT_BUREAU_YEAR",
]
categorical_feats = [
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE",
]

target = "TARGET"
X = df[continuous_feats + categorical_feats].copy()
y = df[target]

# ----------------------------------------------------------------------
# 1. 数据清洗：天数转成年，处理缺失与异常
# ----------------------------------------------------------------------
X["age"] = -X["DAYS_BIRTH"] / 365.25
X["employment_years"] = -X["DAYS_EMPLOYED"] / 365.25
X.loc[X["DAYS_EMPLOYED"] >= 0, "employment_years"] = np.nan  # 365243=无业/退休，视为缺失
X["registration_years"] = -X["DAYS_REGISTRATION"] / 365.25
X["id_publish_years"] = -X["DAYS_ID_PUBLISH"] / 365.25

X = X.drop(columns=["DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH"])

continuous = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "CNT_CHILDREN", "CNT_FAM_MEMBERS", "age", "employment_years",
    "registration_years", "id_publish_years",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "REGION_POPULATION_RELATIVE", "AMT_REQ_CREDIT_BUREAU_YEAR",
]
cont_set = set(continuous)

# ----------------------------------------------------------------------
# 2. 划分训练/验证（分层抽样）
# ----------------------------------------------------------------------
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM, stratify=y
)
print(f"训练集 {len(X_tr)}，验证集 {len(X_va)}，坏客户率 train={y_tr.mean():.4f} valid={y_va.mean():.4f}")


# ----------------------------------------------------------------------
# 3. WOE / IV 工具
# ----------------------------------------------------------------------
def woe_iv(binned: pd.Series, target: pd.Series):
    tab = pd.DataFrame({"bin": binned, "y": target})
    total_good = (tab["y"] == 0).sum()
    total_bad = (tab["y"] == 1).sum()
    agg = tab.groupby("bin")["y"].agg(["count", "sum"])
    agg["good"] = agg["count"] - agg["sum"]
    agg["bad"] = agg["sum"]
    agg["good_dist"] = (agg["good"] + 0.5) / (total_good + 0.5)
    agg["bad_dist"] = (agg["bad"] + 0.5) / (total_bad + 0.5)
    agg["woe"] = np.log(agg["good_dist"] / agg["bad_dist"])
    agg["iv_part"] = (agg["good_dist"] - agg["bad_dist"]) * agg["woe"]
    iv = agg["iv_part"].sum()
    return agg[["count", "good", "bad", "woe", "iv_part"]], iv


def bin_continuous(s, bins):
    """用给定切点 bins 分箱，缺失值单独一箱 MISSING。"""
    b = pd.cut(s, bins=bins, include_lowest=True).astype(str)
    b = b.mask(s.isna(), "MISSING").astype(str)
    return b


# ----------------------------------------------------------------------
# 4. 分箱 + IV（切点/WOE 只在训练集学习）
# ----------------------------------------------------------------------
iv_table = []
woe_maps = {}          # 特征名 -> {bin: woe}
bin_edges = {}         # 连续特征 -> 训练集切点数组

for c in continuous:
    q_edges = np.unique(np.quantile(X_tr[c].dropna(), np.linspace(0, 1, 6)))
    bins = np.concatenate(([-np.inf], q_edges[1:-1], [np.inf]))
    bin_edges[c] = bins
    binned = bin_continuous(X_tr[c], bins)
    _, iv = woe_iv(binned, y_tr)
    iv_table.append((c, "连续", iv))
    agg, _ = woe_iv(binned, y_tr)
    woe_maps[c] = agg["woe"].to_dict()

for c in categorical_feats:
    s = X_tr[c].fillna("MISSING").astype(str)
    vc = s.value_counts()
    rare = vc[vc / len(s) < 0.005].index
    s = s.replace(rare, "OTHER")
    _, iv = woe_iv(s, y_tr)
    iv_table.append((c, "类别", iv))
    agg, _ = woe_iv(s, y_tr)
    woe_maps[c] = agg["woe"].to_dict()

iv_df = pd.DataFrame(iv_table, columns=["feature", "type", "IV"]).sort_values("IV", ascending=False)
print("\n===== IV 表（预测力从强到弱）=====")
print(iv_df.to_string(index=False))

# ----------------------------------------------------------------------
# 5. 特征筛选：IV >= 0.02 保留
# ----------------------------------------------------------------------
keep = iv_df[iv_df["IV"] >= 0.02]["feature"].tolist()
print(f"\n保留 {len(keep)} 个特征（IV>=0.02）：{keep}")


def encode_woe(frame):
    """把原始特征转成 WOE 值（连续用训练切点切，类别映射训练 WOE）。"""
    out = pd.DataFrame(index=frame.index)
    for c in keep:
        if c in cont_set:
            b = bin_continuous(frame[c], bin_edges[c])
        else:
            b = frame[c].fillna("MISSING").astype(str)
            b = b.where(b.isin(woe_maps[c].keys()), "OTHER")
        out[c] = b.map(woe_maps[c]).astype(float)
    return out


W_tr = encode_woe(X_tr)
W_va = encode_woe(X_va)

# ----------------------------------------------------------------------
# 6. 逻辑回归 + 分数刻度化
# ----------------------------------------------------------------------
lr = LogisticRegression(max_iter=2000, random_state=RANDOM)
lr.fit(W_tr, y_tr)

p_tr = lr.predict_proba(W_tr)[:, 1]
p_va = lr.predict_proba(W_va)[:, 1]

auc_tr = roc_auc_score(y_tr, p_tr)
auc_va = roc_auc_score(y_va, p_va)


def ks(y_true, y_score):
    d = pd.DataFrame({"y": y_true, "p": y_score}).sort_values("p", ascending=False)
    n_good = (d["y"] == 0).sum()
    n_bad = (d["y"] == 1).sum()
    cum_good = np.cumsum(d["y"] == 0) / n_good
    cum_bad = np.cumsum(d["y"] == 1) / n_bad
    return np.max(np.abs(cum_good - cum_bad))


ks_tr = ks(y_tr, p_tr)
ks_va = ks(y_va, p_va)
print("\n===== 模型表现 =====")
print(f"AUC  train={auc_tr:.4f}  valid={auc_va:.4f}")
print(f"KS   train={ks_tr:.4f}  valid={ks_va:.4f}")

# 分数刻度化：Score = A + B * ln(good/bad)，PDO=20，600 分对应 good:bad=50:1
PDO = 20
B = PDO / np.log(2)
base_odds = 50.0  # good:bad = 50:1（坏账率约 1.96%）时对应 600 分
A = 600 - B * np.log(base_odds)


def to_score(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    odds = (1 - p) / p  # good/bad 比
    return A + B * np.log(odds)  # good:bad 越高 → 分数越高


score_va = to_score(p_va)
print(f"\n分数公式：Score = {A:.2f} + {B:.2f} * ln(good/bad)")
print(f"验证集分数范围：[{score_va.min():.0f}, {score_va.max():.0f}]")

# ----------------------------------------------------------------------
# 7. 分数段「通过率 vs 坏账率」trade-off（策略岗最看重）
# ----------------------------------------------------------------------
print("\n===== 分数段 trade-off（验证集，从高到低累计）=====")
q = pd.qcut(score_va, q=10, duplicates="drop")
bands = pd.DataFrame({"band": q, "y": y_va.values})
band_summary = bands.groupby("band", observed=True).agg(
    人数=("y", "size"), 坏账率=("y", "mean")
).sort_index(ascending=False)

cum_n, cum_bad, rows = 0, 0, []
for band, row in band_summary.iterrows():
    cum_n += row["人数"]
    cum_bad += row["人数"] * row["坏账率"]
    rows.append({
        "分数段": str(band),
        "段内人数": int(row["人数"]),
        "段内坏账率": round(row["坏账率"], 4),
        "累计通过率": round(cum_n / len(y_va), 4),
        "累计坏账率": round(cum_bad / cum_n, 4),
    })
tradeoff = pd.DataFrame(rows)
print(tradeoff.to_string(index=False))

# ----------------------------------------------------------------------
# 8. 落盘
# ----------------------------------------------------------------------
out_dir = os.path.dirname(os.path.abspath(__file__))
iv_df.to_csv(f"{out_dir}/iv_table.csv", index=False)
pd.DataFrame({"feature": keep, "coef": lr.coef_[0]}).to_csv(f"{out_dir}/lr_coefficients.csv", index=False)
tradeoff.to_csv(f"{out_dir}/score_tradeoff.csv", index=False)
with open(f"{out_dir}/summary.txt", "w", encoding="utf-8") as f:
    f.write(f"AUC_valid={auc_va:.4f}\nKS_valid={ks_va:.4f}\nAUC_train={auc_tr:.4f}\nKS_train={ks_tr:.4f}\n")
    f.write(f"n_features={len(keep)}\nfeatures={keep}\n")
    f.write(f"score_formula=Score = {A:.2f} + {B:.2f} * ln(good/bad)\n")
print("\n[DONE] 结果已写入 scorecard/ 目录")
