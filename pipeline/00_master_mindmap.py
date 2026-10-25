# -*- coding: utf-8 -*-
"""
生成项目总思维导图（一张总览全流程的横向树状图）。

这是"一份总的思维导图"：把 8 个阶段、每个阶段的关键动作与里程碑，
以及最终交付物全部串起来，方便面试时快速回忆整个项目的骨架。
"""
import os
import config
import mindmap_utils

tree = {
    'text': 'Home Credit\n信贷风控项目',
    'children': [
        {
            'text': '01 业务理解与数据探索\n(已完成)',
            'children': [
                {'text': '违约目标 TARGET\n(坏样本率 8.07%)'},
                {'text': '七张表业务含义'},
                {'text': '数据粒度与分布'},
            ],
        },
        {
            'text': '02 多表特征工程\n(已完成初版)',
            'children': [
                {'text': '外部征信 bureau\n+ 月度状态'},
                {'text': '分期付款 installments'},
                {'text': 'POS / CASH 月度逾期'},
                {'text': '历史申请 previous'},
                {'text': '信用卡 credit_card'},
            ],
        },
        {
            'text': '03 多表融合与质量审计\n(本次核心)',
            'children': [
                {'text': '按 SK_ID_CURR 左连接\n保留 307,511 条'},
                {'text': 'T0 防泄漏审计'},
                {'text': '缺失值口径\n(无历史 vs 无分母)'},
                {'text': '唯一性 / 无穷大\n/ 类型检查'},
            ],
        },
        {
            'text': '04 数据集划分与预处理',
            'children': [
                {'text': 'Stratified 分层划分\n80% 训练 / 20% 验证'},
                {'text': '类别变量编码'},
                {'text': '缺失值填补'},
                {'text': '只在训练集拟合\n再应用到验证集'},
            ],
        },
        {
            'text': '05 建立 LR + LightGBM 基线',
            'children': [
                {'text': 'LightGBM 先跑\n快速拿到首个 AUC'},
                {'text': 'LR 逻辑回归对照\n(可解释基线)'},
                {'text': '输出验证集风险概率'},
            ],
        },
        {
            'text': '06 模型优化与特征筛选',
            'children': [
                {'text': 'StratifiedKFold\n交叉验证'},
                {'text': '特征重要性聚合'},
                {'text': '去除冗余 / 低贡献变量\n(交叉验证剔除不明显变量)'},
                {'text': '对比删减前后 AUC'},
            ],
        },
        {
            'text': '07 模型评估与消融实验',
            'children': [
                {'text': 'AUC / KS / PR-AUC'},
                {'text': 'ROC、KS 曲线\n风险十分位分层图'},
                {'text': '模块消融\n量化每张表增量价值'},
            ],
        },
        {
            'text': '08 项目交付与简历总结',
            'children': [
                {'text': '可复现代码'},
                {'text': '特征设计与模型报告'},
                {'text': '局限 / 改进方向'},
                {'text': '简历与面试材料'},
            ],
        },
    ],
}

out_png = os.path.join(config.MINDMAPS_DIR, '00_总思维导图.png')
out_pdf = os.path.join(config.MINDMAPS_DIR, '00_总思维导图.pdf')
mindmap_utils.render_mindmap(tree, out_png, out_pdf=out_pdf,
                             title='Home Credit 信贷风控项目 · 总思维导图',
                             size=(20, 14), fontsize=10)
print('已生成总思维导图:', out_png)
