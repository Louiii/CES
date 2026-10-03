"""Figure 1, drawn at one scale with shared typography and vector artwork.

Run from the project root:
  nanochat/.venv/bin/python inputs/paper/figures/src/build_ces_figure.py

Requires ReportLab, latexmk and Poppler. All geometry, examples, colors and text
sizes live here. LaTeX supplies only the concise mathematical streams; it does
not shrink them to fit. The architecture is vector artwork, not an embedded PNG.
Output: output/pdf/ces_architecture/ces_figure1_readable.{pdf,png}
Use --install to also update the paper's figure and provenance entry.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tempfile

from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'inputs/paper'
OUT = ROOT / 'output/pdf/ces_architecture'
WIDTH, HEIGHT = 1650, 1020
LEFT, LEFT_END, DIVIDER, ARCH = 10, 1185, 1210, 1260
LW = 2.0
FONT = dict(title=26, group=23, standard=24, special=23,
            meta=21, concise=24, apply=23, result=25, arch=23, detail=18,
            container=18)
COLOR = dict(ink='#263449', muted='#69788C', border='#7B8795', faint='#DCE3EB',
             blue='#326EA8', purple='#8874B3', green='#1A7F37', red='#CF222E',
             ces='#AD5963', light='#F3F5F8', bluefill='#EDF4FB',
             purplefill='#F2EEF8', greenfill='#DAFBE1', redfill='#FFEBE9',
             cesfill='#F8ECEE', white='#FFFFFF', norm='#FFF0DD',
             normtext='#A65316', attention='#FFF3C4', attentiontext='#806313',
             transition='#EBDDD1', transitiontext='#79513A', faded='#909AA7')

# The illustrated uniform grammar includes explicit INSERT_EOS. As disclosed in
# edit-serialization section, evaluated checkpoints terminate insertion implicitly.
DATA = [
    ('A', 'Replace one word', ['She', 'walk', 'to', 'school', '.'], {1}, [
        ('REPLACE_OPEN', None), ('REPLACE_START', 1), ('REPLACE_END', 1),
        ('walks', None), ('REPLACE_EOS', None), ('EDIT_END', None)],
     [('She ', 'ink'), ('walks', 'green'), (' to school.', 'ink')]),
    ('B', 'Delete a repeated word + insert a new word',
     ['We', 'really', 'really', 'like', 'tea', '.'], {2}, [
        ('DELETE_OPEN', None), ('DELETE_START', 2), ('DELETE_END', 2),
        ('INSERT_OPEN', None), ('INSERT_LOC', 3), ('green', None),
        ('INSERT_EOS', None), ('EDIT_END', None)],
     [('We really like ', 'ink'), ('green', 'green'), (' tea.', 'ink')]),
    ('C', 'Replace a two-token span', ['She', 'did', 'go', 'home', '.'], {1, 2}, [
        ('REPLACE_OPEN', None), ('REPLACE_START', 1), ('REPLACE_END', 2),
        ('went', None), ('REPLACE_EOS', None), ('EDIT_END', None)],
     [('She ', 'ink'), ('went', 'green'), (' home.', 'ink')]),
    ('D', 'Insert a phrase, then continue', ['She', 'walked', 'store', '.'], set(), [
        ('INSERT_OPEN', None), ('INSERT_LOC', 1), ('to', None), ('the', None),
        ('INSERT_EOS', None), ('EDIT_END', None)],
     [('She walked ', 'ink'), ('to the', 'green'), (' store. She smiled.', 'ink')]),
]
CONTINUATION = ['She', 'smiled', '.']


def register_fonts():
    for name, filename in [('R', 'Lato-Regular.ttf'), ('B', 'Lato-Bold.ttf')]:
        pdfmetrics.registerFont(TTFont(name, '/usr/share/fonts/truetype/lato/' + filename))


def width(s, size, font='R'):
    return pdfmetrics.stringWidth(s, font, size)


def col(s):
    return HexColor(COLOR.get(s, s))


class Drawing:
    def __init__(self, path):
        self.c = canvas.Canvas(str(path), pagesize=(WIDTH, HEIGHT))
        self.c.setTitle('Causal edit programs and CES architecture')
        self.c.setAuthor('Louis Robinson')
        self.formulas = []
        self.y_scale = 1.0
        self.x_scale = 1.0
        self.x_origin = ARCH
        self.top_gap = 0
        self.y_offset = 0
        self.arch_strokes = False

    def stroke_color(self, name):
        if self.arch_strokes and name in ('muted', 'border', 'faint'):
            return '#000000'
        return name

    def mapped_y(self, y):
        # Extra clearance above the transformer, applied to its downstream graph.
        gap = self.top_gap * min(1.0, max(0.0, (y-260)/16))
        return (y+gap)*self.y_scale+self.y_offset

    def mapped_x(self, x):
        return self.x_origin+(x-self.x_origin)*self.x_scale

    def text(self, x, y, s, size, font='R', color='ink', align='left'):
        self.c.setFillColor(col(color))
        self.c.setFont(font, size)
        draw = {'left': self.c.drawString, 'center': self.c.drawCentredString,
                'right': self.c.drawRightString}[align]
        draw(self.mapped_x(x), HEIGHT - self.mapped_y(y) - size * .82, s)

    def box(self, x, y, w, h, fill='white', stroke='border', radius=7, dotted=False):
        self.c.saveState()
        self.c.setFillColor(col(fill))
        self.c.setStrokeColor(col(self.stroke_color(stroke) if stroke else fill))
        self.c.setLineWidth(LW)
        self.c.setLineCap(1)
        if dotted:
            self.c.setDash(.1, 4)
        self.c.roundRect(self.mapped_x(x), HEIGHT-self.mapped_y(y+h), w*self.x_scale, self.mapped_y(y+h)-self.mapped_y(y), radius, stroke=int(stroke is not None), fill=1)
        self.c.restoreState()

    def line(self, x, y, xx, yy, color='faint', lw=LW):
        self.c.setLineWidth(lw)
        self.c.setStrokeColor(col(self.stroke_color(color)))
        self.c.line(self.mapped_x(x), HEIGHT-self.mapped_y(y), self.mapped_x(xx), HEIGHT-self.mapped_y(yy))

    def path(self, points, color='muted', head=True, radius=7):
        """Orthogonal connectors with genuinely curved corners and one arrowhead."""
        points = [(self.mapped_x(float(x)), self.mapped_y(float(y))) for x, y in points]
        c = self.c
        c.saveState()
        c.setStrokeColor(col(self.stroke_color(color))); c.setFillColor(col(self.stroke_color(color)))
        c.setLineWidth(LW); c.setLineCap(1); c.setLineJoin(1)
        p = c.beginPath(); p.moveTo(points[0][0], HEIGHT-points[0][1])
        for prev, point, nxt in zip(points, points[1:], points[2:]):
            a = math.dist(prev, point); b = math.dist(point, nxt)
            r = min(radius, a/2, b/2)
            entry = tuple(point[i] + (prev[i]-point[i])*r/a for i in (0, 1))
            leave = tuple(point[i] + (nxt[i]-point[i])*r/b for i in (0, 1))
            p.lineTo(entry[0], HEIGHT-entry[1])
            # Quadratic corner, expressed as a cubic Bezier.
            c1 = tuple(entry[i] + 2/3*(point[i]-entry[i]) for i in (0, 1))
            c2 = tuple(leave[i] + 2/3*(point[i]-leave[i]) for i in (0, 1))
            p.curveTo(c1[0], HEIGHT-c1[1], c2[0], HEIGHT-c2[1], leave[0], HEIGHT-leave[1])
        x, y = points[-1]; p.lineTo(x, HEIGHT-y); c.drawPath(p)
        if head:
            px, py = points[-2]; angle = math.atan2(y-py, x-px); length = 10
            p = c.beginPath(); p.moveTo(x, HEIGHT-y)
            p.lineTo(x-length*math.cos(angle-.45), HEIGHT-y+length*math.sin(angle-.45))
            p.lineTo(x-length*math.cos(angle+.45), HEIGHT-y+length*math.sin(angle+.45))
            p.close(); c.drawPath(p, stroke=0, fill=1)
        c.restoreState()

    def circle(self, x, y, color='muted', router=False):
        self.c.setStrokeColor(col(self.stroke_color(color))); self.c.setFillColor(col('cesfill' if router else 'white'))
        self.c.setLineWidth(LW)
        self.c.circle(self.mapped_x(x), HEIGHT-self.mapped_y(y), 11, stroke=1, fill=1)
        if router:
            self.line(x, y-6, x, y, 'ces')
            self.line(x-6, y, x+6, y, 'ces')
        else:
            self.text(x, y-10, '+', 22, color=self.stroke_color('muted'), align='center')

    def block(self, x, y, w, h, labels, fill='light', color='ink', ces=False, size=None, data=False):
        self.box(x, y, w, h, fill, None if data else 'ces' if ces else 'border')
        size = size or FONT['arch']
        labels = labels.split('\n')
        top = y + (h - len(labels)*size*1.10)/2 + 2
        for j, label in enumerate(labels):
            assert width(label, size, 'R') <= w*self.x_scale-7, (label, w, size)
            self.text(x+w/2, top+j*size*1.10, label, size, 'R', color, 'center')

    def math(self, x, y, formula, size=22, anchor='west', max_width=None, shaded=False):
        self.formulas.append(dict(x=self.mapped_x(x), y=HEIGHT-self.mapped_y(y), formula=formula, size=size,
                                  anchor=anchor, max_width=max_width*self.x_scale if max_width else None, shaded=shaded))


def masked_indices(row):
    """Vocabulary target coordinates; matches the audited serializer policy."""
    source, program = row[2], row[4]
    masked, start = set(), None
    for j, (label, address) in enumerate(program):
        if address is not None:
            masked.add(len(source)+1+j)
        if label in ('DELETE_START', 'REPLACE_START'):
            start = address
        elif label in ('DELETE_END', 'REPLACE_END'):
            masked.update(range(start, address+2))
        elif label == 'INSERT_LOC':
            masked.add(address+1)
    return masked


def token_width(label, idx, pointer=False):
    standard = '_' not in label
    label_width = max(width(s, FONT['standard'] if standard else FONT['special'],
                            'R') for s in label.split('_'))
    return max(label_width+9, width(str(idx), FONT['meta'])+9,
               width(str(idx), FONT['meta'])+width('@ 3', FONT['meta'], 'B')+18 if pointer else 0)


def token(d, x, y, w, label, idx, address=None, changed=False, masked=False, payload=False):
    special = '_' in label
    fill, fg = ('redfill', 'red') if changed else ('purplefill', 'purple') if address is not None else (
        ('bluefill', 'blue') if special else ('light', 'ink'))
    if payload:
        fill, fg = 'greenfill', 'green'
    d.box(x, y, w, 90, fill, 'border' if masked else None, dotted=masked)
    if special:
        for j, part in enumerate(label.split('_')):
            d.text(x+w/2, y+8+j*27, part, FONT['special'], 'R', fg, 'center')
    else:
        d.text(x+w/2, y+22, label, FONT['standard'], 'R', fg, 'center')
    d.line(x+6, y+62, x+w-6, y+62, 'white', 1.5)
    if address is None:
        d.text(x+w/2, y+69, str(idx), FONT['meta'], color='muted', align='center')
    else:
        d.text(x+6, y+69, str(idx), FONT['meta'], color='muted')
        d.text(x+w-6, y+69, f'@ {address}', FONT['meta'], 'B', 'purple', 'right')
    return x+w/2


def concise_stream(row):
    words = [r'\standard{'+s+'}' for s in row[2]] + [r'\mathcal E_{\mathrm S}']
    operations = {'DELETE': 'D', 'INSERT': 'I', 'REPLACE': 'R', 'EDIT': 'E'}
    roles = {'START': 'S', 'END': 'E', 'LOC': 'L', 'EOS': 'EOS'}
    for label, address in row[4]:
        if '_' not in label:
            words.append(r'\payload{'+label+'}')
            continue
        op, role = label.split('_')
        symbol = r'\mathcal '+operations[op]
        if role != 'OPEN':
            symbol += r'_{\mathrm{'+roles[role]+'}}'
        if address is not None:
            symbol = r'{\color{Pointer}'+symbol+'^{'+str(address)+'}}'
        words.append(symbol)
    if row[0] == 'D':
        words += [r'\standard{'+s+'}' for s in CONTINUATION]
    return r'\sep '.join(words)


def example(d, y, row):
    letter, title, source, changed, program, result = row
    d.text(LEFT, y+4, title, FONT['title'], 'B')
    top = y + (66 if letter == 'A' else 42)
    labels = [(s, None) for s in source] + [('EDIT_START', None)] + program
    if letter == 'D':
        labels += [(s, None) for s in CONTINUATION]
    widths = [token_width(s, idx, p is not None) for idx, (s, p) in enumerate(labels)]
    gaps = [4]*(len(labels)-1)
    gaps[len(source)-1] = 10
    if letter == 'D':
        gaps[len(source)+len(program)] = 10
    total = sum(widths)+sum(gaps)
    assert total <= LEFT_END-LEFT, (letter, total)
    # Tight standard cells: surplus width is not spread across ordinary words.
    x, centres = LEFT, []
    masked = masked_indices(row)
    for idx, ((label, address), w) in enumerate(zip(labels, widths)):
        is_payload = len(source) < idx < len(source)+1+len(program) and '_' not in label
        centre = token(d, x, top, w, label, idx, address, idx in changed, idx in masked, is_payload)
        centres.append(centre)
        x += w+(gaps[idx] if idx < len(gaps) else 0)
    if letter == 'A':
        first_end = LEFT+sum(widths[:len(source)])+sum(gaps[:len(source)-1])
        prog_start = first_end+gaps[len(source)-1]
        for l, r, title_, fg in [(LEFT, first_end, 'STANDARD TOKENS', 'muted'),
                                (prog_start, x, 'EDIT PROGRAM', 'blue')]:
            d.path([(l, top-8), (l, top-16), (r, top-16), (r, top-8)],
                   'faint', head=False)
            tw = width(title_, FONT['group'], 'R')+14
            d.c.setFillColor(col('white'))
            d.c.rect((l+r-tw)/2, HEIGHT-(top-12), tw, 26, stroke=0, fill=1)
            d.text((l+r)/2, top-29, title_, FONT['group'], 'R', fg, 'center')
    pointers = [(len(source)+1+j, address) for j, (_, address) in enumerate(program) if address is not None]
    for j, (idx, address) in enumerate(pointers):
        endx = centres[address]+(-5 if j == 0 else 5)
        yy = top+91 + 1.5*11 + j*(0.5*12)
        d.path([(centres[idx], top+91), (centres[idx], yy),
                (endx, yy), (endx, top+91)], 'purple', radius=5)
    # The concise stream and compact APPLY result share one horizontal line.
    apply_y = top+136
    label_w = width('APPLY', FONT['apply'], 'B')+20
    result_w = sum(width(s, FONT['result'], 'B' if fg == 'green' else 'R') for s, fg in result)
    card_w = label_w+38+result_w+18
    card_x = LEFT_END-card_w
    d.math(LEFT, apply_y+20, concise_stream(row), FONT['concise'], max_width=card_x-LEFT-34, shaded=True)
    d.box(card_x, apply_y, card_w, 40, 'white', None)
    d.box(card_x, apply_y, label_w, 40, 'bluefill', None)
    d.text(card_x+label_w/2, apply_y+11, 'APPLY', FONT['apply'], 'B', 'blue', 'center')
    d.path([(card_x+label_w+10, apply_y+20), (card_x+label_w+28, apply_y+20)], 'muted')
    xx = card_x+label_w+38
    for s, fg in result:
        font = 'B' if fg == 'green' else 'R'
        d.text(xx, apply_y+8, s, FONT['result'], font, fg)
        xx += width(s, FONT['result'], font)
    return top+178


def architecture(d):
    # Shared typography; geometry is wider without stretching text or circles.
    x = ARCH
    d.arch_strokes = True
    d.y_scale = .98
    d.x_scale = 1.20
    d.top_gap = 30
    d.y_offset = -44
    fs = FONT['arch']
    # Three input lanes; rotations are shared across transformer layers.
    d.block(x, 54, 100, 42, 'Tokens x', data=True)
    d.block(x+106, 54, 94, 42, 'Indices i', data=True)
    d.block(x+206, 54, 104, 42, 'Pointers p', 'cesfill', 'ces', True, data=True)
    d.block(x, 124, 100, 61, 'Token\nembedding', 'transition', 'transitiontext')
    d.block(x+106, 124, 94, 61, 'RoPE\n(indices)')
    d.block(x+206, 124, 104, 61, 'RoPE\n(pointers)', 'cesfill', 'ces', True)
    for cc, fg in [(x+50, 'muted'), (x+153, 'muted'), (x+258, 'ces')]:
        d.path([(cc, 96), (cc, 124)], fg)
    d.box(x+105, 218, 205, 42, 'cesfill', 'ces')
    d.math(x+207.5, 239, r'R=\operatorname{diag}(R(i),R(p))', 25, 'center', 195)
    d.path([(x+153, 185), (x+153, 218)])
    d.path([(x+258, 185), (x+258, 202), (x+258, 218)], 'ces')

    # Two equally spaced residual sublayers. The bypass lanes and bottom add
    # node have explicit clearance from the enclosing frame.
    bx, bw, cx = x+8, 266, x+151
    d.box(bx, 276, bw, 313, 'light')
    d.text(bx+bw+8, 595, 'Transformer block', FONT['container'], color='faded', align='right')
    d.text(bx+bw+9, 563, '× L', FONT['container'], color='muted')
    d.path([(x+50, 185), (x+50, 270), (cx, 270), (cx, 307)])
    for norm_y, op_y, add_y, tap_y, label, fill, fg in [
        (307, 359, 417, 296, 'Causal Attention', 'attention', 'attentiontext'),
        (448, 500, 558, 437, 'Transition', 'transition', 'transitiontext'),
    ]:
        d.block(x+62, norm_y, 178, 35, 'RMSNorm', 'norm', 'normtext')
        d.block(x+62, op_y, 178, 38, label, fill, fg)
        d.path([(cx, norm_y+35), (cx, op_y)], head=False)
        d.path([(cx, op_y+38), (cx, add_y-11/d.y_scale)], head=False)
        d.path([(cx, tap_y), (x+30, tap_y), (x+30, add_y), (cx-10, add_y)])
        d.circle(cx, add_y)
    d.path([(cx, 429), (cx, 448)], head=False)
    d.path([(x+310, 239), (x+316, 239), (x+316, 378), (x+240, 378)])
    d.block(x+62, 620, 178, 35, 'RMSNorm', 'norm', 'normtext')
    d.path([(cx, 570), (cx, 620)])
    d.path([(cx, 655), (cx, 674)], head=False)
    d.circle(cx, 685, 'ces', True)

    # Vocabulary readout on the left, position head on the right. Expanded
    # states and score cells keep both mathematical ascenders and descenders clear.
    hx, hw, split = x+90, 226, x+203
    qx, kx = x+146, x+254
    d.box(hx, 714, hw, 277, 'cesfill', 'ces')
    d.text(hx+hw, 994, 'Position head', FONT['container'], color='faded', align='right')
    d.path([(cx+10, 685), (split, 685), (split, 719)], 'ces', head=False)
    d.path([(cx-10, 685), (x+38, 685), (x+38, 835)])
    # Shadow states convey a stack; both arrows terminate on the FRONT cells.
    d.box(x+210, 739, 100, 64, '#DCE3EB', None)
    d.box(x+207, 742, 100, 64, '#E8ECF1', None)
    for cell_x, cc, label, symbol in [
        (x+96, qx, 'Current', r'h_t'),
        (x+204, kx, 'Source', r'h_i'),
    ]:
        d.box(cell_x, 745, 100, 64, 'white', None)
        d.text(cc, 754, label, fs, align='center')
        d.math(cc, 791, symbol, fs, 'center')
        d.path([(split, 719), (cc, 719), (cc, 745)], 'ces', radius=4)
        d.path([(cc, 809), (cc, 828)], 'ces')
        d.path([(cc, 860), (cc, 876)], 'ces', head=False)
    d.block(x+96, 828, 100, 32, 'Wq', 'transition', 'transitiontext', True)
    d.block(x+204, 828, 100, 32, 'Wk', 'transition', 'transitiontext', True)
    d.block(x+96, 876, 100, 32, 'RMSNorm', 'norm', 'normtext')
    d.block(x+204, 876, 100, 32, 'RMSNorm', 'norm', 'normtext')
    d.box(x+128, 933, 150, 48, 'white', 'ces')
    d.math(split, 957, r'q^{\mathsf T} k_i/\sqrt{C}', 24, 'center', 140)
    for cc in [qx, kx]:
        d.path([(cc, 908), (cc, 933)], 'ces')
    d.block(hx, 1020, hw, 32, 'Predicted address logits', 'cesfill', 'ces', True, data=True)
    d.path([(split, 981), (split, 1020)], 'ces')
    d.block(x, 835, 76, 67, 'Vocab\nhead', 'transition', 'transitiontext')
    d.block(x, 1003, 76, 49, 'Vocab\nlogits', data=True)
    d.path([(x+38, 902), (x+38, 1003)])


def latex_overlay(d, base, dest, tmp):
    tex = r'''\documentclass[border=0pt]{standalone}
\usepackage{fix-cm,lmodern,amsmath,xcolor,graphicx,tikz}
\usepackage[T1]{fontenc}
\usepackage{lato}
\definecolor{Pointer}{HTML}{8874B3}
\definecolor{Payload}{HTML}{1A7F37}
\definecolor{Standard}{HTML}{263449}
\definecolor{Separator}{HTML}{C6CCD5}
\definecolor{ConciseBackground}{HTML}{FAF6F1}
\newcommand{\standard}[1]{{\color{Standard}\text{\normalfont\lato #1}}}
\newcommand{\payload}[1]{{\color{Payload}\text{\normalfont\lato #1}}}
\newcommand{\sep}{\mkern2mu{\color{Separator}\raisebox{.08ex}{\scalebox{.65}{\ensuremath{\vert}}}}\mkern2mu}
\newsavebox{\measurebox}
\begin{document}
\begin{tikzpicture}[x=1bp,y=1bp]
'''
    definitions, nodes = [], []
    for idx, f in enumerate(d.formulas):
        name = 'formula' + chr(65+idx)
        content = rf'{{\normalfont\color{{Standard}}\fontsize{{{f["size"]}}}{{{1.1*f["size"]}}}\selectfont ${f["formula"]}$}}'
        definitions.append(rf'\newsavebox{{\{name}}}\sbox{{\{name}}}{{{content}}}')
        if f['max_width']:
            definitions.append(rf'\ifdim\wd\{name}>{f["max_width"]}bp\errmessage{{Formula too wide at {f["x"]},{f["y"]}}}\fi')
        node_style = 'inner xsep=8bp,inner ysep=7bp,rounded corners=5bp,fill=ConciseBackground' if f['shaded'] else 'inner sep=0pt'
        nodes.append(rf'\node[anchor={f["anchor"]},{node_style}] at ({f["x"]},{f["y"]}) {{\usebox{{\{name}}}}};')
    tex = tex.replace(r'\begin{tikzpicture}', '%\n'.join(definitions)+'%\n'+r'\noindent\begin{tikzpicture}')
    tex += rf'\node[anchor=south west,inner sep=0pt] at (0,0) {{\includegraphics[width={WIDTH}bp]{{{base}}}}};'+'\n'
    tex += '\n'.join(nodes)+'\n'
    tex += '\\end{tikzpicture}\n\\end{document}\n'
    (tmp/'figure.tex').write_text(tex)
    with (tmp/'build.log').open('w') as log:
        subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error', 'figure.tex'],
                       cwd=tmp, stdout=log, stderr=subprocess.STDOUT, check=True)
    log_text = (tmp/'figure.log').read_text()
    for problem in ('Missing character:', 'Overfull', 'undefined'):
        assert problem not in log_text, f'{problem} in {tmp}/figure.log'
    shutil.copy2(tmp/'figure.pdf', dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    args = parser.parse_args()
    register_fonts(); OUT.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='ces-figure1-'))
    d = Drawing(tmp/'base.pdf')
    y = 10
    for index, row in enumerate(DATA):
        bottom = example(d, y, row)
        if index != len(DATA)-1:
            d.line(LEFT, bottom+10, LEFT_END, bottom+10)
        y = bottom+21
    assert bottom <= HEIGHT, bottom
    d.line(DIVIDER, 10, DIVIDER, HEIGHT-3)
    architecture(d)
    d.c.showPage(); d.c.save()
    dest = OUT/'ces_figure1_readable.pdf'
    latex_overlay(d, tmp/'base.pdf', dest, tmp)
    subprocess.run(['pdftoppm', '-singlefile', '-scale-to', '3200', '-png', str(dest), str(dest.with_suffix(''))], check=True)
    metadata = dict(width=WIDTH, height=HEIGHT, fonts=FONT, line_width=LW,
                    fonts_at_396pt_width={k: round(v*396/WIDTH, 2) for k, v in FONT.items()},
                    masks={r[0]: sorted(masked_indices(r)) for r in DATA},
                    source=str(Path(__file__).resolve()), build_directory=str(tmp))
    (dest.with_suffix('.json')).write_text(json.dumps(metadata, indent=2)+'\n')
    if args.install:
        shutil.copy2(dest, PAPER/'figures/ces_combined_schematic.pdf')
        path = PAPER/'evidence/selected_panel_assets.json'
        assets = json.loads(path.read_text())
        assets['ces_combined_schematic.pdf'] = dict(source=str(dest),
            sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
            generator=str(Path(__file__).resolve()),
            caption_source=str(PAPER/'sections/methods.tex'),
            build_command='nanochat/.venv/bin/python inputs/paper/figures/src/build_ces_figure.py --install')
        path.write_text(json.dumps(assets, indent=2)+'\n')
    print(json.dumps(metadata, indent=2)); print(dest)


if __name__ == '__main__':
    main()
