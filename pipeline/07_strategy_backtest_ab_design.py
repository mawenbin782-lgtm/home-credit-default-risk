# -*- coding: utf-8 -*-
"""
阶段 9：离线审批阈值回测与 A/B 测试设计
==========================================
本脚本做两件事：
1. 用验证集预测概率，对“拒绝最高风险的前 x% 申请人”做离线策略回测；
2. 为最接近的两档策略生成一份线上 A/B 测试设计说明。

重要边界：
- 这里不是线上 A/B 测试，不能据此声称策略已经上线或获得因果结论。
- 验证集 TARGET 只作为历史违约结果的离线代理标签，用于筛选候选策略。
- 真实线上实验必须在目标市场随机分流，并等待约定的账龄窗口成熟后再判断。

运行方式：
    cd pipeline
    python 07_strategy_backtest_ab_design.py
"""
import json
import math
import os
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


DATA_DIR = os.path.join(config.OUTPUTS_DIR, "data")
OUT_DIR = os.path.join(config.OUTPUTS_DIR, "strategy_backtest")
FIG_DIR = os.path.join(OUT_DIR, "figures")
METRIC_DIR = os.path.join(OUT_DIR, "metrics")
DOC_DIR = os.path.join(config.BASE, "docs")

# 策略含义：按预测违约概率从高到低，拒绝最高风险的该比例申请。
REJECT_TOP_RATES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# A/B 设计示例：保守策略（A）与更严格策略（B）只相差 5% 的最高风险边际客群。
CONTROL_REJECT_TOP = 0.05
TREATMENT_REJECT_TOP = 0.10


def evaluate_threshold_policy(y_true: np.ndarray, risk_prob: np.ndarray, reject_top_rate: float) -> dict:
    """离线评估一个简单的审批阈值策略。

    策略规则：风险概率越高越可能被拒绝；reject_top_rate=0 表示全量通过。
    所有指标均基于同一验证集的已知历史标签，便于横向比较，不代表线上因果效果。
    """
    if not 0 <= reject_top_rate < 1:
        raise ValueError("reject_top_rate must be in [0, 1).")

    # q=0.90 代表拒绝风险概率最高的约 10% 申请。
    cutoff = float(np.quantile(risk_prob, 1 - reject_top_rate)) if reject_top_rate else float("inf")
    approved = risk_prob < cutoff
    rejected = ~approved

    # 在概率并列时，quantile 会让实际拒绝比例略有偏差；报告真实比例而非理论比例。
    n_total = len(y_true)
    n_approved = int(approved.sum())
    n_rejected = int(rejected.sum())
    total_bad = int(y_true.sum())
    approved_bad = int(y_true[approved].sum())
    rejected_bad = int(y_true[rejected].sum())
    total_good = n_total - total_bad
    rejected_good = n_rejected - rejected_bad

    return {
        "reject_top_rate_config": reject_top_rate,
        "risk_cutoff": cutoff if np.isfinite(cutoff) else None,
        "n_total": n_total,
        "n_approved": n_approved,
        "n_rejected": n_rejected,
        "approval_rate": n_approved / n_total,
        "observed_bad_rate_approved": approved_bad / n_approved if n_approved else np.nan,
        "bad_accounts_approved": approved_bad,
        "bad_capture_rate": rejected_bad / total_bad if total_bad else np.nan,
        "good_reject_rate": rejected_good / total_good if total_good else np.nan,
        "actual_reject_rate": n_rejected / n_total,
    }


def two_proportion_sample_size(p_control: float, p_treatment: float,
                               alpha: float = 0.05, power: float = 0.80) -> int:
    """估算两组坏账率比较所需的“每组已成熟放款数”。

    这是正态近似下的规划值，正式上线前应由数据/统计团队按真实分流、成熟率和损失口径复核。
    """
    if not (0 < p_control < 1 and 0 < p_treatment < 1):
        raise ValueError("Rates must lie strictly between 0 and 1.")
    delta = abs(p_control - p_treatment)
    if delta == 0:
        raise ValueError("Control and treatment rates must differ.")
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    z_power = NormalDist().inv_cdf(power)
    p_bar = (p_control + p_treatment) / 2
    numerator = (
        z_alpha * math.sqrt(2 * p_bar * (1 - p_bar))
        + z_power * math.sqrt(p_control * (1 - p_control) + p_treatment * (1 - p_treatment))
    ) ** 2
    return int(math.ceil(numerator / delta ** 2))


def plot_tradeoff(table: pd.DataFrame, output_path: str) -> None:
    """展示审批率与已通过客群坏账率之间的策略权衡。"""
    fig, ax1 = plt.subplots(figsize=(8.4, 4.8))
    x = table["actual_reject_rate"] * 100
    ax1.plot(x, table["observed_bad_rate_approved"] * 100, marker="o", color="#C44E52",
             label="Observed bad rate among approved")
    ax1.set_xlabel("Actual reject rate (%)")
    ax1.set_ylabel("Observed bad rate among approved (%)", color="#C44E52")
    ax1.tick_params(axis="y", labelcolor="#C44E52")
    ax1.grid(axis="both", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, table["approval_rate"] * 100, marker="s", color="#4C72B0",
             label="Approval rate")
    ax2.set_ylabel("Approval rate (%)", color="#4C72B0")
    ax2.tick_params(axis="y", labelcolor="#4C72B0")

    for _, row in table.iterrows():
        ax1.annotate(f"reject top {row['reject_top_rate_config']:.0%}",
                     (row["actual_reject_rate"] * 100,
                      row["observed_bad_rate_approved"] * 100),
                     xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax1.set_title("Offline approval-policy backtest: risk and volume trade-off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_ab_design(control: pd.Series, treatment: pd.Series, sample_size: int, output_path: str) -> None:
    """输出可学习、可面试讲解的 A/B 测试设计文档。"""
    marginal_band = treatment["actual_reject_rate"] - control["actual_reject_rate"]
    text = f"""# 审批阈值策略 A/B 测试设计（方案，不是已上线实验）

## 1. 实验背景与假设

离线回测显示，拒绝更高风险申请人可能降低已通过客群的历史违约率，但也会牺牲通过量。
拟在线上比较两档审批阈值：

- **对照策略 A**：拒绝预测风险最高的约 {control['actual_reject_rate']:.1%} 申请。
- **实验策略 B**：拒绝预测风险最高的约 {treatment['actual_reject_rate']:.1%} 申请。
- **策略差异客群**：约 {abs(marginal_band):.1%} 的边际高风险申请人；策略 B 比策略 A 更严格。

业务假设：在不超过预设通过量损失的前提下，策略 B 可降低已放款客群在约定账龄上的逾期/坏账风险。

## 2. 离线回测证据（仅用于筛选候选策略）

| 指标 | 策略 A | 策略 B |
| --- | ---: | ---: |
| 通过率 | {control['approval_rate']:.2%} | {treatment['approval_rate']:.2%} |
| 通过客群历史坏账率 | {control['observed_bad_rate_approved']:.2%} | {treatment['observed_bad_rate_approved']:.2%} |
| 拦截历史坏样本占比 | {control['bad_capture_rate']:.2%} | {treatment['bad_capture_rate']:.2%} |
| 误拒好样本占比 | {control['good_reject_rate']:.2%} | {treatment['good_reject_rate']:.2%} |

> 注：上述结果来自同一离线验证集的已知历史标签，不能代替线上因果结论，更不能称为已完成 A/B 测试。

## 3. 实验设计

- **实验单元**：符合当前授信基本准入条件的一笔申请；同一客户多次申请需按客户 ID 固定分桶，防止跨组污染。
- **分流方式**：在进入审批策略前，以稳定哈希随机 1:1 分流至 A/B 两组；流量比例、入组/剔除规则和策略版本需留痕。
- **观察窗口**：以放款月份为 cohort，至少等到预先约定的账龄窗口成熟后结论。例如以 MOB1 DPD30+ 作为早期质量指标，并在更长期窗口复核最终违约/损失。
- **主指标**：已放款客群的 MOB1 DPD30+（或业务统一定义的早期逾期率）。
- **业务指标**：申请通过率、放款量、放款金额、每千申请人的坏账账户数/预期损失。
- **护栏指标**：欺诈命中率、人工审核率、投诉率、策略拒绝率、转化率，以及按新老客/渠道/国家切分后的风险差异。
- **统计检验**：对主指标比较两组比例差，报告效应量、95% 置信区间和 p 值；每天监控样本比例失衡（SRM），不要因为短期波动提前下结论。

## 4. 样本量与决策规则

若把离线策略 A/B 的通过客群历史坏账率作为规划参考，双侧显著性水平 5%、统计功效 80% 下，
比较这两个比例大约需要 **{sample_size:,} 笔成熟放款/组**。这是粗略规划值；正式上线前需依据真实基线逾期率、最小可接受效果（MDE）、放款率和账龄成熟率重新计算。

建议预先写清停止/上线规则：

1. 实验运行达到计划样本量和账龄成熟窗口后再做主结论；
2. 若策略 B 的主风险指标显著改善，且通过率/放款量及其他护栏未突破阈值，可扩大流量；
3. 若风险恶化或关键护栏超限，应暂停实验并回退至当前策略。

## 5. 面试表达边界

我在项目中完成的是“离线阈值回测和线上 A/B 测试方案设计”，而不是线上策略部署。离线回测用于提出候选策略；只有线上随机分流并等待账龄成熟，才能评估策略的因果效果。
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(METRIC_DIR, exist_ok=True)
    os.makedirs(DOC_DIR, exist_ok=True)

    y_valid = np.load(os.path.join(DATA_DIR, "y_valid.npy"))
    risk_prob = np.load(os.path.join(DATA_DIR, "pred_lgb_final_valid.npy"))
    if len(y_valid) != len(risk_prob):
        raise ValueError("y_valid and pred_lgb_final_valid have different lengths.")

    table = pd.DataFrame([
        evaluate_threshold_policy(y_valid, risk_prob, rate)
        for rate in REJECT_TOP_RATES
    ])
    baseline_bad_rate = float(table.loc[table["reject_top_rate_config"] == 0,
                                        "observed_bad_rate_approved"].iloc[0])
    table["bad_rate_reduction_vs_approve_all"] = (
        1 - table["observed_bad_rate_approved"] / baseline_bad_rate
    )

    csv_path = os.path.join(OUT_DIR, "approval_policy_tradeoff.csv")
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    plot_path = os.path.join(FIG_DIR, "approval_policy_tradeoff.png")
    plot_tradeoff(table, plot_path)

    control = table.loc[table["reject_top_rate_config"] == CONTROL_REJECT_TOP].iloc[0]
    treatment = table.loc[table["reject_top_rate_config"] == TREATMENT_REJECT_TOP].iloc[0]
    sample_size = two_proportion_sample_size(
        float(control["observed_bad_rate_approved"]),
        float(treatment["observed_bad_rate_approved"]),
    )

    comparison = pd.DataFrame([control, treatment])
    comparison.insert(0, "policy", ["A_control", "B_treatment"])
    comparison.to_csv(os.path.join(OUT_DIR, "ab_candidate_policy_comparison.csv"),
                      index=False, encoding="utf-8-sig")

    summary = {
        "scope": "offline historical-label backtest and online A/B test design; not a deployed experiment",
        "validation_n": int(len(y_valid)),
        "historical_bad_rate_all_applicants": baseline_bad_rate,
        "control_reject_top_rate": CONTROL_REJECT_TOP,
        "treatment_reject_top_rate": TREATMENT_REJECT_TOP,
        "sample_size_per_group_matured_loans_illustrative": sample_size,
        "primary_online_metric": "MOB1 DPD30+ among approved loans, subject to local business definition",
    }
    with open(os.path.join(METRIC_DIR, "strategy_backtest_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    design_path = os.path.join(DOC_DIR, "审批阈值策略AB测试设计.md")
    write_ab_design(control, treatment, sample_size, design_path)

    print("Done: offline strategy backtest and A/B test design")
    print(f"- Trade-off table: {csv_path}")
    print(f"- Trade-off chart: {plot_path}")
    print(f"- Candidate comparison: {os.path.join(OUT_DIR, 'ab_candidate_policy_comparison.csv')}")
    print(f"- A/B design: {design_path}")
    print(f"- Illustrative per-group matured-loan sample size: {sample_size:,}")


if __name__ == "__main__":
    main()
