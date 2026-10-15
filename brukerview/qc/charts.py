"""Small inline-SVG charts for the QC report.

Self-contained (no JS library): every mark carries a ``<title>`` so hovering
shows its exact value, and colours come from CSS custom properties defined by
the report, so light and dark themes are both styled. Series colours use the
categorical slots ``--s1..--s3`` in fixed order; at most three series per chart.
"""

from __future__ import annotations

import math
from html import escape
from typing import Callable, Dict, List, Optional, Sequence, Tuple

FONT = 11


def nice_ticks(lo: float, hi: float, n: int = 5) -> List[float]:
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return [0.0, 1.0]
    if hi == lo:
        hi = lo + (abs(lo) or 1.0)
    raw = (hi - lo) / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=10 * mag)
    start = math.floor(lo / step) * step
    ticks, t = [], start
    while not ticks or ticks[-1] < hi - 1e-12 * abs(hi):
        ticks.append(round(t, 10))
        t += step
    return ticks


def fmt_num(v: Optional[float], digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return '–'
    a = abs(v)
    if a == 0:
        return '0'
    if a >= 1000:
        return f'{v:,.0f}'
    if a >= 100:
        return f'{v:.0f}'
    if a >= 10:
        return f'{v:.1f}'
    if a >= 1:
        return f'{v:.2f}'
    return f'{v:.{digits}f}'


def _scale(d0, d1, r0, r1):
    span = (d1 - d0) or 1.0
    return lambda v: r0 + (v - d0) / span * (r1 - r0)


def line_chart(series: Sequence[Dict], xlabel: str, ylabel: str, width: int = 560, height: int = 300,
               y_zero: bool = True, x_ticks: Optional[Sequence[float]] = None,
               x_fmt: Callable[[float], str] = fmt_num, markers: bool = True,
               aria: str = '') -> str:
    """Lines with markers. ``series``: ``{'name', 'slot' (1-3), 'points': [(x, y, tip)]}``."""
    pts = [(x, y) for s in series for x, y, *_ in s['points'] if y is not None and math.isfinite(y)]
    if not pts:
        return ''
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    xt = list(x_ticks) if x_ticks else nice_ticks(min(xs), max(xs))
    yt = nice_ticks(0 if y_zero else min(ys), max(ys))
    longest = max((len(s['name']) for s in series), default=0)
    right = min(40 + 6.4 * longest, 260) if len(series) > 1 else 24
    L, R, T, B = 56, width - right, 14, height - 44
    sx = _scale(min(xt[0], min(xs)), max(xt[-1], max(xs)), L, R)
    sy = _scale(yt[0], yt[-1], B, T)
    o = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(aria or ylabel)}">']
    for t in yt:
        o.append(f'<line class="grid" x1="{L}" x2="{R}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>'
                 f'<text class="tick" x="{L - 6}" y="{sy(t) + 4:.1f}" text-anchor="end">{fmt_num(t)}</text>')
    for t in xt:
        o.append(f'<text class="tick" x="{sx(t):.1f}" y="{B + 16}" text-anchor="middle">{escape(x_fmt(t))}</text>')
    o.append(f'<line class="axis" x1="{L}" x2="{R}" y1="{B}" y2="{B}"/>')
    o.append(f'<text class="label" x="{(L + R) / 2:.0f}" y="{height - 6}" text-anchor="middle">{escape(xlabel)}</text>')
    o.append(f'<text class="label" transform="translate(13,{(T + B) / 2:.0f}) rotate(-90)" text-anchor="middle">{escape(ylabel)}</text>')
    ends = []
    for s in series:
        p = [(x, y, *rest) for x, y, *rest in s['points'] if y is not None and math.isfinite(y)]
        if not p:
            continue
        cls = f's{s.get("slot", 1)}'
        if len(p) > 1:
            d = ' '.join(f'{"M" if i == 0 else "L"}{sx(x):.1f},{sy(y):.1f}' for i, (x, y, *_) in enumerate(p))
            o.append(f'<path class="line {cls}" d="{d}"/>')
        if markers:
            for x, y, *rest in p:
                tip = rest[0] if rest else f'{s["name"]}: {fmt_num(y)}'
                o.append(f'<g class="pt"><circle class="hit" cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="10"/>'
                         f'<circle class="dot {cls}" cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4.5"/>'
                         f'<title>{escape(tip)}</title></g>')
        ends.append([sy(p[-1][1]), s['name'], cls, sx(p[-1][0])])
    if len(series) > 1:  # direct end labels, nudged apart so they never stack
        ends.sort()
        for i in range(1, len(ends)):
            if ends[i][0] - ends[i - 1][0] < 14:
                ends[i][0] = ends[i - 1][0] + 14
        for y, name, cls, x in ends:
            o.append(f'<line class="lead {cls}" x1="{R + 4}" x2="{R + 14}" y1="{y:.1f}" y2="{y:.1f}"/>'
                     f'<text class="endlabel" x="{R + 18}" y="{y + 4:.1f}">{escape(name)}</text>')
    o.append('</svg>')
    return ''.join(o)


def legend(series: Sequence[Dict]) -> str:
    if len(series) < 2:
        return ''
    items = ''.join(f'<span class="key"><i class="sw s{s.get("slot", 1)}"></i>{escape(s["name"])}</span>'
                    for s in series)
    return f'<div class="legend">{items}</div>'


def dot_rows(labels: Sequence[str], panels: Sequence[Dict], groups: Optional[Sequence[str]] = None,
             label_width: Optional[int] = None, panel_width: int = 170, row_h: int = 22) -> str:
    """Cleveland dot plots sharing one row axis, one panel (own x scale) per metric.

    ``panels``: ``{'title', 'values', 'lo'?, 'hi'?, 'better': 'high'|'low'|None,
    'fmt'?, 'zero'?: bool, 'ref'?: (value, label)}``. The best row per panel gets a
    ring and its value printed; every dot has a hover title. ``groups`` optionally
    names each row's group: a hairline and the group name separate the groups.
    """
    n = len(labels)
    if label_width is None:  # ~6.4 px per character at 11.5 px
        label_width = int(24 + 6.4 * max((len(x) for x in labels), default=10))
    head = 34
    gaps = [i for i in range(1, n) if groups and groups[i] != groups[i - 1]]
    extra = 18 * (len(gaps) + (1 if groups else 0))
    height = head + n * row_h + extra + 8
    width = label_width + len(panels) * (panel_width + 16)
    ys, y = [], head + (18 if groups else 0)
    for i in range(n):
        if i in gaps:
            y += 18
        ys.append(y + row_h / 2)
        y += row_h
    o = [f'<svg class="chart dots" viewBox="0 0 {width:.0f} {height:.0f}" width="{width:.0f}" height="{height:.0f}" '
         f'role="img" aria-label="metric dot plot">']
    if groups:
        starts = [0] + gaps
        for s in starts:
            gy = ys[s] - row_h / 2 - 5
            o.append(f'<text class="group" x="4" y="{gy:.1f}">{escape(groups[s])}</text>')
            if s:
                o.append(f'<line class="axis" x1="0" x2="{width}" y1="{gy - 13:.1f}" y2="{gy - 13:.1f}"/>')
    for i, lab in enumerate(labels):
        o.append(f'<text class="rowlabel" x="{label_width - 10}" y="{ys[i] + 4:.1f}" text-anchor="end">{escape(lab)}</text>')
    for j, p in enumerate(panels):
        x0 = label_width + j * (panel_width + 16)
        x1 = x0 + panel_width
        vals = [v if v is not None and isinstance(v, (int, float)) and math.isfinite(v) else None for v in p['values']]
        lo = [a if a is not None else v for a, v in zip(p.get('lo') or vals, vals)]
        hi = [a if a is not None else v for a, v in zip(p.get('hi') or vals, vals)]
        fin = [v for v in lo + hi + vals if v is not None]
        if p.get('ref'):
            fin.append(p['ref'][0])
        if not fin:
            continue
        dmin = 0 if p.get('zero', True) and min(fin) >= 0 else min(fin)
        ticks = nice_ticks(dmin, max(fin), 4)
        sx = _scale(ticks[0], ticks[-1], x0 + 6, x1 - 6)
        f = p.get('fmt', fmt_num)
        o.append(f'<text class="ptitle" x="{x0}" y="14">{escape(p["title"])}</text>')
        if p.get('better'):
            o.append(f'<text class="hint" x="{x0}" y="27">{"higher" if p["better"] == "high" else "lower"} is better</text>')
        bottom = ys[-1] + row_h / 2
        for t in ticks:
            o.append(f'<line class="grid" x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{head - 2}" y2="{bottom:.1f}"/>'
                     f'<text class="tick" x="{sx(t):.1f}" y="{bottom + 12:.1f}" text-anchor="middle">{fmt_num(t)}</text>')
        if p.get('ref'):
            rv, rl = p['ref']
            o.append(f'<line class="refline" x1="{sx(rv):.1f}" x2="{sx(rv):.1f}" y1="{head - 2}" y2="{bottom:.1f}">'
                     f'<title>{escape(rl)}</title></line>')
        best = None
        cand = [(v, i) for i, v in enumerate(vals) if v is not None]
        if p.get('better') and cand:
            best = (max if p['better'] == 'high' else min)(cand)[1]
        for i, v in enumerate(vals):
            if v is None:
                o.append(f'<text class="tick" x="{x0 + 6}" y="{ys[i] + 4:.1f}">n/a</text>')
                continue
            if lo[i] != hi[i]:
                o.append(f'<line class="whisker" x1="{sx(lo[i]):.1f}" x2="{sx(hi[i]):.1f}" y1="{ys[i]:.1f}" y2="{ys[i]:.1f}"/>')
            tip = f'{labels[i]} — {p["title"]}: {f(v)}'
            if lo[i] != hi[i]:
                tip += f' (range {f(lo[i])}–{f(hi[i])})'
            ring = ' best' if i == best else ''
            o.append(f'<g class="pt"><circle class="hit" cx="{sx(v):.1f}" cy="{ys[i]:.1f}" r="10"/>'
                     f'<circle class="dot s1{ring}" cx="{sx(v):.1f}" cy="{ys[i]:.1f}" r="4.5"/><title>{escape(tip)}</title></g>')
            if i == best:
                o.append(f'<text class="bestlabel" x="{min(sx(hi[i]) + 8, x1 - 2):.1f}" y="{ys[i] - 6:.1f}">{f(v)}</text>')
    o.append('</svg>')
    return ''.join(o)
