# -*- coding: utf-8 -*-
"""
中文学习 PDF 生成器（基于 reportlab）。

设计目标：每个阶段产出一份"讲清楚为什么"的 PDF，结构统一：
   标题 -> 本阶段目标 -> 为什么做这一步(重点高亮) -> 关键决策与原理
   -> 输入/输出 -> 实际结果 -> 踩坑点 -> 下一步

用法示例：
    from pdf_utils import Doc
    d = Doc('阶段3：多表融合与质量审计')
    d.h1('一、本阶段要解决什么')
    d.body('……')
    d.why('为什么必须先做质量审计：……')
    d.bullets(['点1', '点2'])
    d.table([['列A','列B'], ['1','2']], col_widths=[60, 100])
    d.image('figures/xxx.png', width=150)
    d.save('learnings/03_多表融合.pdf')
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image, PageBreak, HRFlowable)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER

import config

_font_registered = False


def _register_fonts():
    global _font_registered
    if _font_registered:
        return
    # 同一个 SimHei 同时注册为 regular 与 bold，保证 <b> 标签不报错（黑体本身已够粗）
    pdfmetrics.registerFont(TTFont('CN', config.FONT_SIMHEI))
    pdfmetrics.registerFont(TTFont('CN-Bold', config.FONT_SIMHEI))
    _font_registered = True


# 配色：统一、克制，主色深蓝 + 强调色青绿 + 警示橙
C_PRIMARY = colors.HexColor('#1F3864')   # 深蓝（标题/主色）
C_ACCENT  = colors.HexColor('#2E7D6B')   # 青绿（二级标题/强调）
C_WARN    = colors.HexColor('#B3541E')   # 橙（踩坑/警示）
C_WHY_BG  = colors.HexColor('#EAF1F8')   # "为什么"高亮框底色（浅蓝）
C_CODE_BG = colors.HexColor('#F4F4F4')   # 代码块底色
C_GREY    = colors.HexColor('#666666')
C_ZEBRA   = colors.HexColor('#F7F9FC')   # 表格斑马纹底色
C_LINE    = colors.HexColor('#E3E8EF')   # 表格横向分隔线


def _styles():
    def mk(name, **kw):
        base = dict(fontName='CN', fontSize=10.5, leading=17, textColor=colors.black,
                    wordWrap='CJK', alignment=TA_LEFT)
        base.update(kw)
        return ParagraphStyle(name, **base)
    s = {
        'title': mk('title', fontName='CN-Bold', fontSize=21, leading=28,
                    textColor=C_PRIMARY, spaceAfter=6),
        'subtitle': mk('subtitle', fontSize=11, leading=16, textColor=C_GREY, spaceAfter=14),
        'h1': mk('h1', fontName='CN-Bold', fontSize=15, leading=21,
                 textColor=C_PRIMARY, spaceBefore=14, spaceAfter=6),
        'h2': mk('h2', fontName='CN-Bold', fontSize=12.5, leading=18,
                 textColor=C_ACCENT, spaceBefore=10, spaceAfter=4),
        'body': mk('body', firstLineIndent=0),
        'bullet': mk('bullet', leftIndent=14, bulletIndent=4, spaceAfter=2),
        'why': mk('why', fontSize=10.5, leading=18, backColor=C_WHY_BG,
                  borderPadding=7, spaceBefore=6, spaceAfter=6),
        'code': mk('code', fontSize=8.5, leading=12.5, backColor=C_CODE_BG,
                   borderPadding=6, spaceBefore=4, spaceAfter=8),
        'caption': mk('caption', fontSize=8.5, leading=12, textColor=C_GREY,
                      alignment=TA_CENTER, spaceBefore=2, spaceAfter=10),
        'warn': mk('warn', fontSize=10.5, leading=17, textColor=C_WARN, spaceBefore=4),
    }
    return s


class Doc:
    """一个阶段一份 PDF 的构建器。"""

    def __init__(self, title, subtitle=''):
        _register_fonts()
        self.title = title
        self.subtitle = subtitle
        self.st = _styles()
        self.story = []

    # ---- 基础元素 ----
    def h1(self, text):
        self.story.append(Paragraph(text, self.st['h1']))
        self.story.append(HRFlowable(width='100%', thickness=1.1, color=C_PRIMARY,
                                     spaceBefore=2, spaceAfter=10))

    def h2(self, text):
        self.story.append(Paragraph(text, self.st['h2']))

    def body(self, text):
        self.story.append(Paragraph(text, self.st['body']))

    def bullets(self, items):
        for it in items:
            self.story.append(Paragraph('●&nbsp;&nbsp;' + it, self.st['bullet']))

    def numbered(self, items):
        for i, it in enumerate(items, 1):
            self.story.append(Paragraph(f'<b>{i}.</b>&nbsp;{it}', self.st['bullet']))

    def why(self, text):
        """高亮的"为什么"框：这是每份 PDF 的学习核心。"""
        self.story.append(Paragraph('<b>◆ 为什么这一步？</b><br/>' + text, self.st['why']))

    def warn(self, text):
        self.story.append(Paragraph('▲ ' + text, self.st['warn']))

    def warnlist(self, items):
        """橙色警示列表：用于"中间的问题"板块，与普通项目符号区分。"""
        st = ParagraphStyle('warnitem', fontName='CN', fontSize=10.5, leading=17,
                            textColor=C_WARN, leftIndent=14, bulletIndent=4,
                            spaceAfter=2, wordWrap='CJK', alignment=TA_LEFT)
        for it in items:
            self.story.append(Paragraph('▲&nbsp;&nbsp;' + it, st))

    def label(self, text, color=None):
        """分节标签：左侧色条 + 浅底文字，做成一眼可辨的"标签条"。"""
        color = color or C_ACCENT
        tint = colors.HexColor('#F2F6FB')
        inner = ParagraphStyle('label-inner', fontName='CN-Bold', fontSize=11,
                               leading=15, textColor=color, wordWrap='CJK',
                               alignment=TA_LEFT)
        bar = Table([[Paragraph(text, inner)]], colWidths=[174*mm])
        bar.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), tint),
            ('LINEBEFORE', (0, 0), (0, -1), 3, color),
            ('LINEAFTER', (0, 0), (0, -1), 0, colors.white),
            ('LINEABOVE', (0, 0), (0, -1), 0, colors.white),
            ('LINEBELOW', (0, 0), (0, -1), 0, colors.white),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 9),
            ('RIGHTPADDING', (0, 0), (-1, -1), 9),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        self.story.append(Spacer(1, 8))
        self.story.append(bar)
        self.story.append(Spacer(1, 4))

    def code(self, text):
        # 转义 XML 特殊字符，保留换行
        import html
        text = html.escape(text)
        text = text.replace('\n', '<br/>')
        self.story.append(Paragraph(text, self.st['code']))

    def table(self, data, col_widths=None, header_bg=C_PRIMARY):
        import html
        # col_widths 单位是毫米，reportlab 的 colWidths 单位是点（pt）。先把毫米转点，
        # 再按比例归一化到正文可用宽度(174mm)，这样脚本里列宽之和不管是多少都能正好铺满、不溢出。
        if col_widths is not None:
            total = float(sum(col_widths))
            col_widths = [w * mm * (174.0 / total) for w in col_widths]
        body_style = ParagraphStyle('cell', fontName='CN', fontSize=9, leading=13,
                                    textColor=colors.black, wordWrap='CJK',
                                    alignment=TA_LEFT)
        head_style = ParagraphStyle('cellhead', fontName='CN-Bold', fontSize=9,
                                    leading=13, textColor=colors.white,
                                    wordWrap='CJK', alignment=TA_LEFT)
        n = len(data)
        # 每个单元格包成 Paragraph，中英文 / 斜杠 / 空格都能正确换行；转义 < > & 避免被当成标签
        rows = []
        for ri, row in enumerate(data):
            st = head_style if (ri == 0 and header_bg is not None) else body_style
            rows.append([Paragraph(html.escape(str(cell)), st) for cell in row])
        t = Table(rows, colWidths=col_widths, repeatRows=1)
        style = [
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 7),
            ('RIGHTPADDING', (0, 0), (-1, -1), 7),
            # 只保留横向分隔线，去掉竖线，观感更清爽
            ('LINEBELOW', (0, 0), (-1, -2), 0.4, C_LINE),
            ('LINEBELOW', (0, -1), (-1, -1), 0, colors.white),
        ]
        # 斑马纹：数据第 2、4、6… 行加浅底
        for i in range(2, n, 2):
            style.append(('BACKGROUND', (0, i), (-1, i), C_ZEBRA))
        if header_bg is not None and n > 1:
            style += [
                ('BACKGROUND', (0, 0), (-1, 0), header_bg),
                ('LINEBELOW', (0, 0), (-1, 0), 1.0, header_bg),
            ]
        t.setStyle(TableStyle(style))
        self.story.append(t)
        self.story.append(Spacer(1, 9))

    def image(self, path, width=160*mm, caption=''):
        from reportlab.lib.utils import ImageReader
        from PIL import Image as PILImage
        img = PILImage.open(path)
        w, h = img.size
        # 按宽度缩放，并限制单图高度不超过可用高度的 0.6
        draw_w = width
        draw_h = draw_w * h / w
        max_h = A4[1] * 0.6
        if draw_h > max_h:
            draw_h = max_h
            draw_w = draw_h * w / h
        self.story.append(Image(path, width=draw_w, height=draw_h))
        self.story.append(Spacer(1, 2))
        if caption:
            self.story.append(Paragraph(caption, self.st['caption']))

    def hr(self):
        self.story.append(HRFlowable(width='100%', thickness=0.7,
                                     color=C_ACCENT, spaceBefore=6, spaceAfter=6))

    def pagebreak(self):
        self.story.append(PageBreak())

    def save(self, path):
        doc = SimpleDocTemplate(path, pagesize=A4,
                                leftMargin=18*mm, rightMargin=18*mm,
                                topMargin=16*mm, bottomMargin=16*mm,
                                title=self.title)
        # 封面头
        story = [Paragraph(self.title, self.st['title'])]
        if self.subtitle:
            story.append(Paragraph(self.subtitle, self.st['subtitle']))
        story.append(HRFlowable(width='100%', thickness=1.2, color=C_PRIMARY, spaceAfter=12))
        story += self.story
        doc.build(story)
        return path
