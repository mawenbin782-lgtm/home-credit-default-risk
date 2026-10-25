# -*- coding: utf-8 -*-
"""
统一风格的 matplotlib 绘图工具（ROC / KS / 特征重要性 / 风险分层）。

遵循 dataviz 原则：
  - 颜色跟随"实体/含义"，不跟随排名；
  - 顺序量用单一色相的深浅，不做彩虹色；
  - 细线、弱化网格与坐标轴；>= 2 条曲线必有图例；
  - 文字用墨色，不用系列颜色。
"""
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager, pyplot as plt
import numpy as np
import config

for f in (config.FONT_SIMHEI, config.FONT_MSYH):
    try:
        font_manager.fontManager.addfont(f)
    except Exception:
        pass
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.edgecolor'] = '#C8C8C8'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.color'] = '#ECECEC'
plt.rcParams['grid.linewidth'] = 0.6
plt.rcParams['grid.alpha'] = 1.0
plt.rcParams['xtick.color'] = '#666666'
plt.rcParams['ytick.color'] = '#666666'
plt.rcParams['axes.labelcolor'] = '#333333'
plt.rcParams['text.color'] = '#333333'
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['axes.labelsize'] = 10

# 项目统一的固定色板（与 PDF/思维导图一致）
C_PRIMARY = '#1F3864'   # 主蓝
C_ACCENT  = '#2E7D6B'   # 青绿
C_WARN    = '#B3541E'   # 橙
C_GREY    = '#999999'   # 参考线/中性
C_LIGHT   = '#7A93B8'   # 主蓝的浅一档

CATEGORY_COLORS = [C_PRIMARY, C_ACCENT, C_WARN, '#5B7FB8', '#C07B3A']


def _style_ax(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    return ax


def roc_curve_compare(y_true, models, out_path, title='ROC 曲线'):
    """models: list of (label, prob_array)。同一条 y_true。"""
    from sklearn.metrics import roc_curve, auc
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    _style_ax(ax)
    ax.plot([0, 1], [0, 1], '--', color=C_GREY, lw=1.2, label='随机猜测 (AUC=0.5)')
    for (label, prob), color in zip(models, CATEGORY_COLORS):
        fpr, tpr, _ = roc_curve(y_true, prob)
        a = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC={a:.4f})')
    ax.set_xlabel('假正率 FPR')
    ax.set_ylabel('真正率 TPR')
    ax.set_title(title, fontsize=12, color=C_PRIMARY, pad=10)
    ax.legend(loc='lower right', frameon=False, fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path


def ks_curve(y_true, prob, out_path, title='KS 曲线'):
    """画好坏样本的累计分布曲线，并标出最大 KS 差值。"""
    from sklearn.metrics import roc_curve
    bad = prob[y_true == 1]
    good = prob[y_true == 0]
    ks = 0.0
    ks_thresh = 0.0
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    _style_ax(ax)
    for arr, label, color in [(good, '好样本(未违约)', C_ACCENT), (bad, '坏样本(违约)', C_WARN)]:
        arr = np.sort(arr)
        cdf = np.arange(1, len(arr) + 1) / len(arr)
        ax.plot(arr, cdf, color=color, lw=2, label=label)
    # 用统一阈值网格算 KS
    grid = np.linspace(0, 1, 1000)
    cdf_good = np.searchsorted(np.sort(good), grid, side='right') / len(good)
    cdf_bad = np.searchsorted(np.sort(bad), grid, side='right') / len(bad)
    diff = cdf_bad - cdf_good
    i = int(np.argmax(np.abs(diff)))
    ks = abs(diff[i])
    ks_thresh = grid[i]
    ax.plot([ks_thresh, ks_thresh], [cdf_good[i], cdf_bad[i]], color=C_PRIMARY, lw=2)
    ax.scatter([ks_thresh], [(cdf_good[i] + cdf_bad[i]) / 2], color=C_PRIMARY, s=40, zorder=5)
    ax.annotate(f'KS = {ks:.4f}', xy=(ks_thresh, (cdf_good[i] + cdf_bad[i]) / 2),
                xytext=(ks_thresh + 0.04, (cdf_good[i] + cdf_bad[i]) / 2 - 0.12),
                color=C_PRIMARY, fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_PRIMARY, lw=1))
    ax.set_xlabel('预测违约概率')
    ax.set_ylabel('累计占比 CDF')
    ax.set_title(title, fontsize=12, color=C_PRIMARY, pad=10)
    ax.legend(loc='upper left', frameon=False, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path, ks


def feature_importance_hbar(names, gains, out_path, top_n=20, title='特征重要性 Top20'):
    """横向条形图，单一主蓝色相（顺序量）。"""
    idx = np.argsort(gains)[::-1][:top_n]
    names = [names[i] for i in idx][::-1]
    gains = [gains[i] for i in idx][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(names) + 1.2))
    _style_ax(ax)
    ax.barh(names, gains, color=C_PRIMARY, height=0.62)
    for i, v in enumerate(gains):
        ax.text(v, i, f' {v:.0f}', va='center', ha='left', fontsize=8, color='#555555')
    ax.set_xlabel('重要性（分裂增益 gain）')
    ax.set_title(title, fontsize=12, color=C_PRIMARY, pad=10)
    ax.grid(axis='x')
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path


def risk_decile_bar(decile_stats, out_path, title='风险十分位分层'):
    """decile_stats: DataFrame，含列 decile(1..10)、actual_bad_rate、pred_mean。"""
    fig, ax = plt.subplots(figsize=(7, 4.6))
    _style_ax(ax)
    x = np.arange(len(decile_stats)) + 1
    ax.bar(x, decile_stats['actual_bad_rate'], color=C_PRIMARY, width=0.66,
           label='实际坏样本率')
    ax.plot(x, decile_stats['pred_mean'], color=C_WARN, lw=2, marker='o', ms=4,
            label='预测概率均值')
    ax.set_xlabel('风险分位（1=低风险 … 10=高风险）')
    ax.set_ylabel('违约率')
    ax.set_xticks(x)
    ax.set_title(title, fontsize=12, color=C_PRIMARY, pad=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis='y')
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path
