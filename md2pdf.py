#!/usr/bin/env python3
"""md2pdf.py — PRISM 技术报告 Markdown → 学术 PDF（ChinaXiv 提交用）
ReportLab 路线：注册微软雅黑，解析标题/段落/表格/列表/引用，A4 学术版式。"""
import re
import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, HRFlowable, KeepTogether)

FONT_DIR = 'C:/Windows/Fonts'
pdfmetrics.registerFont(TTFont('MSYH', f'{FONT_DIR}/msyh.ttc', subfontIndex=0))
pdfmetrics.registerFont(TTFont('MSYHBD', f'{FONT_DIR}/msyhbd.ttc', subfontIndex=0))
pdfmetrics.registerFontFamily('MSYH', normal='MSYH', bold='MSYHBD', italic='MSYH', boldItalic='MSYHBD')

INK = colors.HexColor('#1a1a1a')
ACC = colors.HexColor('#2c3e50')
MUT = colors.HexColor('#6b7280')
LINE = colors.HexColor('#d5d9e0')
BG = colors.HexColor('#f4f6f8')

S = {
    'h1': ParagraphStyle('h1', fontName='MSYHBD', fontSize=16, leading=24, textColor=ACC,
                         spaceBefore=18, spaceAfter=8, wordWrap='CJK'),
    'h2': ParagraphStyle('h2', fontName='MSYHBD', fontSize=13, leading=19, textColor=ACC,
                         spaceBefore=14, spaceAfter=6, wordWrap='CJK'),
    'h3': ParagraphStyle('h3', fontName='MSYHBD', fontSize=11, leading=16, textColor=INK,
                         spaceBefore=10, spaceAfter=4, wordWrap='CJK'),
    'body': ParagraphStyle('body', fontName='MSYH', fontSize=10, leading=16.5, textColor=INK,
                           spaceAfter=5, alignment=TA_JUSTIFY, wordWrap='CJK'),
    'quote': ParagraphStyle('quote', fontName='MSYH', fontSize=9.5, leading=15, textColor=MUT,
                            leftIndent=10, spaceAfter=5, wordWrap='CJK'),
    'li': ParagraphStyle('li', fontName='MSYH', fontSize=10, leading=16, textColor=INK,
                         leftIndent=14, bulletIndent=4, spaceAfter=3, wordWrap='CJK'),
    'title': ParagraphStyle('title', fontName='MSYHBD', fontSize=20, leading=30,
                            textColor=ACC, alignment=TA_CENTER, spaceAfter=10, wordWrap='CJK'),
    'subtitle': ParagraphStyle('subtitle', fontName='MSYH', fontSize=11, leading=17,
                               textColor=MUT, alignment=TA_CENTER, spaceAfter=6, wordWrap='CJK'),
    'meta': ParagraphStyle('meta', fontName='MSYH', fontSize=9, leading=14, textColor=MUT,
                           alignment=TA_CENTER, spaceAfter=4, wordWrap='CJK'),
    'cell': ParagraphStyle('cell', fontName='MSYH', fontSize=8.5, leading=12.5,
                           textColor=INK, wordWrap='CJK'),
    'cellh': ParagraphStyle('cellh', fontName='MSYHBD', fontSize=8.5, leading=12.5,
                            textColor=ACC, wordWrap='CJK'),
}


def esc(t):
    return (t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


def inline(t):
    t = esc(t)
    t = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', t)
    t = re.sub(r'`(.+?)`', r'<font face="MSYH">\1</font>', t)
    return t


def make_table(rows, avail_w):
    ncol = max(len(r) for r in rows)
    rows = [r + [''] * (ncol - len(r)) for r in rows]
    # 权重：首列略窄
    weights = [1.1] + [1.0] * (ncol - 1)
    tot = sum(weights)
    cw = [avail_w * w / tot for w in weights]
    data = []
    for i, r in enumerate(rows):
        st = S['cellh'] if i == 0 else S['cell']
        data.append([Paragraph(inline(c), st) for c in r])
    t = Table(data, colWidths=cw, hAlign='CENTER', repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BG),
        ('GRID', (0, 0), (-1, -1), 0.4, LINE),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fafbfc')]),
    ]))
    return t


def parse(md, story, avail_w):
    lines = md.split('\n')
    i, first_h1 = 0, True
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s.startswith('|') and i + 1 < len(lines) and re.match(r'^\|[\s\-:|]+\|$', lines[i + 1].strip()):
            rows = [[c.strip() for c in s.strip('|').split('|')]]
            i += 2
            while i < len(lines) and lines[i].strip().startswith('|'):
                rows.append([c.strip() for c in lines[i].strip().strip('|').split('|')])
                i += 1
            story.append(Spacer(1, 3))
            story.append(make_table(rows, avail_w))
            story.append(Spacer(1, 6))
            continue
        if s.startswith('# '):
            if first_h1:
                story.append(Spacer(1, 60))
                story.append(Paragraph(inline(s[2:]), S['title']))
                first_h1 = False
            else:
                story.append(Paragraph(inline(s[2:]), S['h1']))
                story.append(HRFlowable(width='100%', thickness=0.7, color=LINE, spaceAfter=6))
        elif s.startswith('## '):
            story.append(Paragraph(inline(s[3:]), S['h2']))
        elif s.startswith('### '):
            story.append(Paragraph(inline(s[4:]), S['h3']))
        elif s.startswith('> '):
            story.append(Paragraph(inline(s[2:]), S['quote']))
        elif re.match(r'^[-*] ', s):
            story.append(Paragraph(inline(s[2:]), S['li'], bulletText='•'))
        elif re.match(r'^\d+\. ', s):
            m = re.match(r'^(\d+)\. (.*)', s)
            story.append(Paragraph(inline(m.group(2)), S['li'], bulletText=m.group(1) + '.'))
        elif s == '---':
            story.append(HRFlowable(width='30%', thickness=0.5, color=LINE,
                                    spaceBefore=8, spaceAfter=8, hAlign='CENTER'))
        elif s == '':
            pass
        else:
            # 合并后续非格式行为同段落
            buf = [s]
            j = i + 1
            while j < len(lines):
                nx = lines[j].strip()
                if (nx == '' or nx.startswith(('#', '|', '>', '- ', '* ', '---')) or re.match(r'^\d+\. ', nx)):
                    break
                buf.append(nx)
                j += 1
            txt = inline(' '.join(buf))
            if s.startswith('**') and s.endswith('**') and len(buf) == 1:
                story.append(Paragraph(txt, S['h3']))
            else:
                story.append(Paragraph(txt, S['body']))
            i = j - 1
        i += 1


def footer(canv, doc):
    canv.saveState()
    canv.setFont('MSYH', 8)
    canv.setFillColor(MUT)
    canv.drawCentredString(A4[0] / 2, 12 * mm, f'— {doc.page} —')
    canv.restoreState()


def main():
    src = Path(sys.argv[1])
    out = sys.argv[2]
    md = src.read_text(encoding='utf-8')
    # 元信息行（> 开头的头部）转为 subtitle 样式区
    doc = SimpleDocTemplate(out, pagesize=A4,
                            leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=20 * mm,
                            title='预测编码网络在语言建模中的样本效率、机制不可达性与测量纪律',
                            author='Independent Researcher')
    avail_w = A4[0] - 40 * mm
    story = []
    parse(md, story, avail_w)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f'OK -> {out}')


if __name__ == '__main__':
    main()
