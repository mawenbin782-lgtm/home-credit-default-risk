# -*- coding: utf-8 -*-
"""
用 matplotlib 绘制横向树状思维导图（markmap / XMind 风格：根节点在左，分支向右展开）。

节点结构：一个 dict，键 'text' 为文字，'children' 为子节点列表（可空）。

用法：
    tree = {
        'text': 'Home Credit 风控项目',
        'children': [
            {'text': '阶段1 业务理解', 'children': [...]},
            {'text': '阶段2 特征工程', 'children': [...]},
        ],
    }
    render_mindmap(tree, 'mindmaps/总图.png', title='项目总览', size=(16, 9))
"""
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager, pyplot as plt
from matplotlib.patches import FancyBboxPatch
import config

# 注册中文字体，matplotlib 才能正常显示中文
for f in (config.FONT_SIMHEI, config.FONT_MSYH):
    try:
        font_manager.fontManager.addfont(f)
    except Exception:
        pass
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 根/各级分支配色（一个分支一种颜色，视觉上区分不同模块）
BRANCH_COLORS = ['#2E7D6B', '#B3541E', '#5B7FB8', '#8A6FA8', '#C07B3A',
                 '#4A8AA8', '#B04A6B', '#6B7A2E', '#7A5A3A', '#3A6B8A']
ROOT_COLOR = '#1F3864'


def _text_w(text, fontsize):
    """估算文字像素宽度：中文全角算 1，英文/数字算 0.58。"""
    w = 0.0
    for c in text:
        w += 1.0 if ord(c) > 127 else 0.58
    return w * fontsize


def _assign(node, depth, counter):
    """递归给每个节点分配深度与纵向槽位（叶子按顺序占一格，内部节点取子节点均值）。"""
    node['depth'] = depth
    if not node.get('children'):
        node['y'] = counter[0]
        node['leaf'] = True
        counter[0] += 1
    else:
        ys = []
        for c in node['children']:
            _assign(c, depth + 1, counter)
            ys.append(c['y'])
        node['y'] = sum(ys) / len(ys)
        node['leaf'] = False


def render_mindmap(tree, out_png, title=None, size=(16, 9), fontsize=11,
                   out_pdf=None, dpi=150):
    """把树结构画成横向思维导图。out_png 为图片路径；out_pdf 可选，输出矢量版。"""
    counter = [0]
    _assign(tree, 0, counter)
    n_leaves = counter[0]

    # 每个节点的文字宽度与盒子尺寸
    pad_x, pad_y = 10, 6
    max_depth = 0
    def box(node):
        nonlocal max_depth
        max_depth = max(max_depth, node['depth'])
        node['tw'] = _text_w(node['text'], fontsize)
        node['bw'] = node['tw'] + 2 * pad_x
        node['bh'] = fontsize + 2 * pad_y

    def walk(n):
        box(n)
        for c in n.get('children', []):
            walk(c)
    walk(tree)

    # 每个层级取最宽盒子，作为该层累计 x 位置
    level_max_w = {}
    def lw(n):
        level_max_w[n['depth']] = max(level_max_w.get(n['depth'], 0), n['bw'])
        for c in n.get('children', []):
            lw(c)
    lw(tree)

    gap_x = 40          # 层与层之间的水平间距
    gap_y = 26          # 兄弟节点纵向间距
    xs = [0.0]
    for d in range(1, max_depth + 1):
        xs.append(xs[-1] + level_max_w[d - 1] / 2 + gap_x + level_max_w[d] / 2)

    # 纵向坐标：叶子槽位 * (盒子高 + gap_y)，根在中部
    dy = max(n['bh'] for n in [tree]) + gap_y
    # 重新算全局最大盒子高做纵向间距
    all_bh = []
    def bh(n):
        all_bh.append(n['bh'])
        for c in n.get('children', []):
            bh(c)
    bh(tree)
    dy = max(all_bh) + gap_y

    def x_of(n):
        return xs[n['depth']]
    def y_of(n):
        return -n['y'] * dy

    fig_w, fig_h = size
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, xs[-1] + level_max_w[max_depth] / 2 + 30)
    ymin = -(n_leaves - 1) * dy - 30
    ymax = 30
    ax.set_ylim(ymin, ymax)
    ax.axis('off')

    # 颜色规则：根深蓝；第一层按分支顺序取色；更深层白底 + 所属分支彩边
    def _mark_branch(n, idx):
        n['_branch'] = idx
        for c in n.get('children', []):
            _mark_branch(c, idx)

    for i, c in enumerate(tree.get('children', [])):
        _mark_branch(c, i)

    def branch_color(n):
        return BRANCH_COLORS[n['_branch'] % len(BRANCH_COLORS)]

    def fill_color(n):
        if n['depth'] == 0:
            return ROOT_COLOR
        if n['depth'] == 1:
            return branch_color(n)
        return 'white'

    def edge_color(n):
        return branch_color(n) if n.get('_branch') is not None else ROOT_COLOR

    # 画线 + 节点
    def draw(n):
        x, y = x_of(n), y_of(n)
        for c in n.get('children', []):
            cx, cy = x_of(c), y_of(c)
            ax.plot([x + n['bw'] / 2, cx - c['bw'] / 2], [y, cy],
                    color=edge_color(c), lw=1.2, alpha=0.55, zorder=1)
            draw(c)

        fc = fill_color(n)
        if n['depth'] >= 2:
            ec = branch_color(n)
        else:
            ec = fc
        boxpatch = FancyBboxPatch((x - n['bw'] / 2, y - n['bh'] / 2),
                                  n['bw'], n['bh'],
                                  boxstyle='round,pad=0.02,rounding_size=3',
                                  fc=fc, ec=ec, lw=1.3, zorder=2)
        ax.add_patch(boxpatch)
        tc = 'white' if n['depth'] <= 1 else '#333333'
        ax.text(x, y, n['text'], ha='center', va='center',
                fontsize=fontsize, color=tc, zorder=3)
    draw(tree)

    if title:
        ax.set_title(title, fontsize=fontsize + 6, color=ROOT_COLOR, pad=12, fontweight='bold')

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches='tight', facecolor='white')
    if out_pdf:
        fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_png
