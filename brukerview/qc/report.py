"""Assemble the self-contained HTML QC report (+ TSV tables)."""

from __future__ import annotations

import csv
import datetime as _dt
import math
import re
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import charts, figures
from .scans import ROLES, Scan, fmt_param, parameter_table

# ---- metric dictionary: key -> (label, better, format) -------------------------------------------
METRICS: Dict[str, Tuple[str, Optional[str], str]] = {
    'snr': ('SNR', 'high', '{:.1f}'),
    'noise_method': ('Noise estimate', None, '{}'),
    'bg_ratio': ('Noise region mean/SD', None, '{:.2f}'),
    'cnr': ('CNR CSF–tissue', 'high', '{:.1f}'),
    'contrast_ratio': ('CSF/tissue ratio', None, '{:.2f}'),
    'snr_per_sqrt_min': ('SNR / √min', 'high', '{:.2f}'),
    'snr_per_nl_sqrt_min': ('SNR / nL / √min', 'high', '{:.3f}'),
    'cnr_per_nl_sqrt_min': ('CNR / nL / √min', 'high', '{:.3f}'),
    'snr_b0': ('SNR b0', 'high', '{:.1f}'),
    'snr_b500': ('SNR b500', 'high', '{:.1f}'),
    'snr_b1000': ('SNR b1000', 'high', '{:.1f}'),
    'snr_b2000': ('SNR b2000', 'high', '{:.1f}'),
    'snr_b0_bg': ('SNR b0 (background)', 'high', '{:.1f}'),
    'b0_snr_per_nl_sqrt_min': ('b0 SNR / nL / √min', 'high', '{:.3f}'),
    'fa_median': ('FA median', None, '{:.2f}'),
    'md_median_um2_ms': ('MD median (µm²/ms)', None, '{:.2f}'),
    'tsnr': ('tSNR (median)', 'high', '{:.1f}'),
    'tsnr_first10': ('tSNR first 10 vols', 'high', '{:.1f}'),
    'snr_diff': ('Image SNR (diff.)', 'high', '{:.1f}'),
    'tsnr_per_sqrt_s': ('tSNR / √TRvol(s)', 'high', '{:.2f}'),
    'tsnr_per_nl_sqrt_s': ('tSNR / nL / √s', 'high', '{:.3f}'),
    'drift_pct_per_min': ('Drift (%/min)', None, '{:+.2f}'),
    'first_vol_pct': ('Vol 1 vs mean (%)', None, '{:+.1f}'),
    'ghost_pct': ('Ghost (% signal)', 'low', '{:.1f}'),
    'coverage_slab_pct': ('Brain in slab (%)', 'high', '{:.0f}'),
    'coverage_sampled_pct': ('Brain sampled (%)', 'high', '{:.0f}'),
    'disp_median_mm': ('Shift median (mm)', 'low', '{:.2f}'),
    'disp_p95_mm': ('Shift p95 (mm)', 'low', '{:.2f}'),
    'disp_p95_px': ('Shift p95 (px)', 'low', '{:.1f}'),
    'mm_per_100hz': ('mm per 100 Hz', 'low', '{:.2f}'),
    'b0_sd_hz': ('ΔB0 SD (Hz)', 'low', '{:.0f}'),
    'b0_sd_ppm': ('ΔB0 SD (ppm)', 'low', '{:.3f}'),
    'b0_iqr_hz': ('ΔB0 IQR (Hz)', 'low', '{:.0f}'),
    'b0_p5_p95_hz': ('ΔB0 5–95% (Hz)', 'low', '{:.0f}'),
    'fwhm_hz': ('Water FWHM (Hz)', 'low', '{:.1f}'),
    'fwhm_ppm': ('Water FWHM (ppm)', 'low', '{:.3f}'),
    'fwhm_mag_hz': ('Magnitude FWHM (Hz)', 'low', '{:.1f}'),
}

ROLE_METRICS = {
    'anat': ['snr', 'cnr', 'contrast_ratio', 'snr_per_sqrt_min', 'snr_per_nl_sqrt_min', 'cnr_per_nl_sqrt_min',
             'noise_method', 'bg_ratio'],
    'dwi': ['snr_b0', 'snr_b500', 'snr_b1000', 'snr_b2000', 'snr_b0_bg', 'b0_snr_per_nl_sqrt_min',
            'fa_median', 'md_median_um2_ms', 'ghost_pct', 'coverage_slab_pct', 'coverage_sampled_pct',
            'disp_median_mm', 'disp_p95_mm', 'disp_p95_px'],
    'epi_ref': ['snr', 'snr_per_nl_sqrt_min', 'bg_ratio', 'ghost_pct', 'coverage_slab_pct',
                'coverage_sampled_pct', 'disp_p95_mm', 'disp_p95_px'],
    'bold': ['tsnr', 'tsnr_first10', 'snr_diff', 'tsnr_per_sqrt_s', 'tsnr_per_nl_sqrt_s', 'ghost_pct',
             'coverage_slab_pct', 'coverage_sampled_pct', 'disp_p95_mm', 'disp_p95_px', 'mm_per_100hz',
             'drift_pct_per_min', 'first_vol_pct'],
    'fieldmap': ['b0_sd_hz', 'b0_sd_ppm', 'b0_iqr_hz', 'b0_p5_p95_hz'],
    'mrs': ['fwhm_hz', 'fwhm_ppm', 'fwhm_mag_hz'],
}


def fmt_metric(key: str, v: Any) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return '–'
    f = METRICS.get(key, (key, None, '{}'))[2]
    try:
        return f.format(v)
    except (ValueError, TypeError):
        return str(v)


# ---- small HTML helpers ---------------------------------------------------------------------------
def table(header: Sequence[str], rows: Sequence[Sequence[str]], highlight: Sequence[int] = (),
          cls: str = '', raw: bool = False, num_from: int = 1) -> str:
    h = ''.join(f'<th class="{"vary" if i in highlight else ""}">{escape(c)}</th>' for i, c in enumerate(header))
    body = []
    for r in rows:
        cells = []
        for i, c in enumerate(r):
            klass = ' '.join(k for k in ('vary' if i in highlight else '', 'num' if i >= num_from else '') if k)
            cells.append(f'<td class="{klass}">{c if raw else escape(str(c))}</td>')
        body.append('<tr>' + ''.join(cells) + '</tr>')
    return f'<div class="tablewrap"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def param_table(scans: List[Scan]) -> str:
    cols, rows, varying = parameter_table(scans)
    header = ['Scan'] + [lab for _, lab in cols]
    hi = [i + 1 for i, (k, _) in enumerate(cols) if k in varying]
    rows = [[f'E{s.number}'] + r for s, r in zip(scans, rows)]
    return table(header, rows, highlight=hi, cls='params', num_from=2)


def metric_table(results, keys: Sequence[str]) -> str:
    keys = [k for k in keys if any(r.metrics.get(k) is not None for r in results)]
    header = ['Scan', 'Protocol'] + [METRICS.get(k, (k,))[0] for k in keys]
    rows = [[f'E{r.scan.number}', r.scan.params['name']] + [fmt_metric(k, r.metrics.get(k)) for k in keys]
            for r in results]
    return table(header, rows, cls='metrics', num_from=2)


def short_label(scan: Scan, keys: Sequence[str]) -> str:
    """'E21 · Cor · 90×90 · 0.50 mm · TR 750' from the parameters that vary in the family."""
    p = scan.params
    bits = []
    for k in keys:
        v = p.get(k)
        if v in (None, ''):
            continue
        if k == 'orientation':
            bits.append({'coronal': 'Cor', 'axial': 'Ax', 'sagittal': 'Sag'}.get(v, v))
        elif k == 'matrix':
            bits.append(v.replace('x', '×'))
        elif k == 'inplane_um':
            bits.append(v.split('x')[0] + ' µm')
        elif k == 'slice_thick_mm':
            g = p.get('slice_gap_mm')
            bits.append(f'{v:.2g}' + (f'+{g:.1g}' if g else '') + ' mm')
        elif k == 'tr_ms':
            bits.append(f'TR {v:g}')
        elif k == 'te_ms':
            bits.append(f'TE {v:.3g}')
        elif k == 'repetitions':
            bits.append(f'{v} vol')
        elif k == 'reversed_pe':
            bits.append('rev-PE' if v == 'Yes' else 'fwd-PE')
        elif k == 'n_slices':
            bits.append(f'{v} sl')
    return ' · '.join([f'E{scan.number}'] + bits)


LABEL_KEYS = ['orientation', 'matrix', 'inplane_um', 'slice_thick_mm', 'n_slices', 'tr_ms', 'te_ms',
              'repetitions', 'reversed_pe']


def labels_for(scans: List[Scan]) -> List[str]:
    _, _, varying = parameter_table(scans)
    if 'slice_gap_mm' in varying:
        varying.add('slice_thick_mm')
    keys = [k for k in LABEL_KEYS if k in varying] or ['orientation']
    if 'inplane_um' in keys and 'matrix' in keys:
        keys.remove('inplane_um')
    return [short_label(s, keys) for s in scans]


def md_to_html(md: str) -> str:
    """Tiny markdown subset: #/##/### headings, '-' bullets, paragraphs, **b**, *i*, `code`."""
    def inline(t):
        t = escape(t)
        t = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', t)
        t = re.sub(r'(?<!\*)\*(?!\s)(.+?)(?<!\s)\*', r'<em>\1</em>', t)
        return re.sub(r'`(.+?)`', r'<code>\1</code>', t)
    out, para, items = [], [], []

    def flush():
        if para:
            out.append('<p>' + inline(' '.join(para)) + '</p>')
            para.clear()
        if items:
            out.append('<ul>' + ''.join(f'<li>{inline(i)}</li>' for i in items) + '</ul>')
            items.clear()
    for line in md.splitlines():
        s = line.strip()
        if not s:
            flush()
        elif s.startswith('#'):
            flush()
            lvl = min(len(s) - len(s.lstrip('#')) + 1, 5)
            out.append(f'<h{lvl}>{inline(s.lstrip("#").strip())}</h{lvl}>')
        elif s.startswith('- '):
            if para:
                flush()
            items.append(s[2:])
        elif items and line.startswith('  '):
            items[-1] += ' ' + s
        else:
            if items:
                flush()
            para.append(s)
    flush()
    return '\n'.join(out)


# ---- sections -------------------------------------------------------------------------------------
def _section(id_: str, title: str, body: str, lead: str = '') -> str:
    lead = f'<p class="lead">{lead}</p>' if lead else ''
    return f'<section id="{id_}"><h2>{escape(title)}</h2>{lead}{body}</section>'


def _fig(uri: str, caption: str = '', cls: str = '') -> str:
    cap = f'<figcaption>{caption}</figcaption>' if caption else ''
    return f'<figure class="{cls}"><img src="{uri}" alt="{escape(re.sub("<.*?>", "", caption))}">{cap}</figure>'


def _chart(title: str, svg: str, note: str = '', series=None) -> str:
    if not svg:
        return ''
    lg = charts.legend(series) if series else ''
    note = f'<p class="note">{note}</p>' if note else ''
    return f'<div class="chartcard"><h4>{escape(title)}</h4>{lg}{svg}{note}</div>'


def sec_inventory(scans: List[Scan], analysed: set) -> str:
    rows = []
    for s in scans:
        p = s.params
        status = 'analysed' if s.number in analysed else 'not analysed'
        rows.append([f'E{s.number}', p['name'], p['method'], ROLES.get(s.role, s.role), p.get('matrix', ''),
                     fmt_param('scan_time_s', p.get('scan_time_s')), status])
    return table(['Scan', 'Protocol', 'Method', 'Role', 'Matrix', 'Time', 'QC'], rows, cls='inventory', num_from=99)


def sec_reference(ref) -> str:
    center = ref.center
    ext = np.array([6.5, 5.0, 8.0])
    uri = figures.ortho([{'data': ref.data, 'aff': ref.affine, 'label': f'E{ref.scan.number}',
                          'mask': ref.mask}], center, ext, res=0.04)
    return (f'<p>The brain mask is drawn once, on <strong>E{ref.scan.number} {escape(ref.scan.params["name"])}</strong>, '
            f'and carried into every other scan through the scanner geometry (no registration). '
            f'Volume <strong>{ref.volume_mm3:.0f} mm³</strong>. Method: {escape(ref.how)}. '
            f'It is saved as <code>brain_mask.nii.gz</code>; edit it and pass it back with '
            f'<code>--brain-mask</code> if it needs correcting.</p>'
            + _fig(uri, 'Brain mask (green outline) on the anatomical reference, world-axis planes through the brain centre. '
                   + PLANE_NOTE))


def sec_anat(results, ref) -> str:
    scans = [r.scan for r in results]
    body = '<h3>Protocol parameters</h3><p class="note">Shaded columns differ between these scans.</p>' + param_table(scans)
    body += '<h3>Metrics</h3>' + metric_table(results, ROLE_METRICS['anat'])
    methods = {r.metrics.get('noise_method') for r in results}
    if methods & {'air pockets', 'laplacian'}:
        body += ('<p class="note">The 3-D RARE field of view is filled with head and body, so there is no air '
                 'around the head to measure noise in (background mean/SD ≈ 1, where pure noise gives ≈ 2–2.75). '
                 'Noise was measured instead in the darkest 1% of 5-voxel neighbourhoods (airways, ear canals); '
                 'their mean/SD (shown) confirms they hold pure noise. This is thermal noise: respiratory ghosting '
                 'inside the brain is not included. If no such pockets exist the Laplacian residual inside the '
                 'brain is used (labelled "laplacian"; it over-estimates noise where anatomy is textured).</p>')
    ext = np.array([6.5, 5.0, 8.0])
    vols = [{'data': r.arrays['vol'], 'aff': r.arrays['aff'], 'label': short_label(r.scan, ['inplane_um']),
             'window': figures.window(r.arrays['vol'][r.arrays['brain']], 0.5, 99.5)} for r in results]
    body += '<h3>Representative images</h3>' + _fig(
        figures.ortho(vols, ref.center, ext, res=0.04),
        'Same world-axis planes through the brain centre for each acquisition, nearest-neighbour '
        'resampled so the native voxel size is visible. Window: 0.5–99.5th percentile within the brain. ' + PLANE_NOTE)
    return body


def sec_dwi(results) -> str:
    scans = [r.scan for r in results]
    labels = labels_for(scans) if len(scans) > 1 else [f'E{s.number}' for s in scans]
    body = '<h3>Protocol parameters</h3>' + param_table(scans)
    series = []
    for i, r in enumerate(results[:3]):
        rows = r.arrays.get('shell_snr') or []
        series.append({'name': labels[i], 'slot': i + 1,
                       'points': [(x['b'], x['snr'], f"{labels[i]} b={x['b']:g}: SNR {x['snr']:.1f} "
                                   f"(per direction {x['snr_dir_min']:.1f}–{x['snr_dir_max']:.1f}, n={x['n_vol']})")
                                  for x in rows]})
    cards = _chart('Single-image SNR per shell', charts.line_chart(
        series, 'b-value (s/mm²)', 'SNR (difference method)', aria='SNR per b-value'),
        'Signal = mean brain intensity of one diffusion-weighted image; noise from the difference of the two '
        'repetitions. Below SNR ≈ 3 the magnitude noise floor biases the signal upward (and FA with it).', series)
    fa_series = []
    for i, r in enumerate(results[:3]):
        FA, core = r.arrays.get('FA'), r.arrays.get('core')
        if FA is None:
            continue
        h, e = np.histogram(FA[core], bins=40, range=(0, 1), density=True)
        c = 0.5 * (e[1:] + e[:-1])
        fa_series.append({'name': labels[i], 'slot': i + 1,
                          'points': [(x, y, f'{labels[i]}: FA {x:.3f}, density {y:.2f}') for x, y in zip(c, h)]})
    cards += _chart('FA distribution in the brain', charts.line_chart(
        fa_series, 'FA', 'density', markers=False, aria='FA histogram'),
        'Tensor fitted to b ≤ 1000 s/mm². With low SNR the FA of grey matter is inflated, so a '
        'right-shifted histogram can mean more noise, not more anisotropy.', fa_series)
    body += '<h3>Charts</h3><div class="grid2">' + cards + '</div>'
    body += '<h3>Metrics</h3>' + metric_table(results, ROLE_METRICS['dwi'])
    body += '<h3>Representative images</h3>'
    for r, lab in zip(results, labels):
        core = r.arrays['core']
        ks = figures.pick_slices(core, 7)
        mid = ks[len(ks) // 2]
        body += _fig(figures.dwi_strip(r.arrays['shells'], r.arrays.get('FA'), r.arrays.get('MD'),
                                               r.arrays['brain'], mid, lab),
                     f'{escape(lab)}: one slice, mean of each shell, FA and MD.')
        b0 = r.arrays['b0']
        rows = [{'data': b0, 'label': 'b0', 'vmin': 0, 'vmax': figures.window(b0[core])[1]}]
        if r.arrays.get('FA') is not None:
            rows.append({'data': r.arrays['FA'], 'label': 'FA', 'vmin': 0, 'vmax': 0.8, 'outline': False})
        if r.arrays.get('disp_map') is not None:
            rows.append({'data': np.nan_to_num(r.arrays['disp_map']), 'label': 'pred. shift', 'cmap': 'RdBu_r',
                         'vmin': -0.6, 'vmax': 0.6, 'cbar': 'mm along PE', 'outline': True})
        body += _fig(figures.montage(rows, r.arrays['brain'], ks),
                     f'{escape(lab)}: b0 with the anatomical brain outline (mismatch = distortion or '
                     f'motion), FA, and the shift predicted from the B0 map.')
    return body


def sec_epi_ref(results) -> str:
    scans = [r.scan for r in results]
    labels = labels_for(scans)
    body = '<h3>Protocol parameters</h3>' + param_table(scans)
    body += '<h3>Metrics</h3>' + metric_table(results, ROLE_METRICS['epi_ref'])
    body += ('<p class="note">Single-volume references: SNR comes from the air background (validated by its '
             'mean/SD ratio). Forward/reversed phase-encode pairs are meant for susceptibility '
             'correction (e.g. FSL topup); their displacement is equal and opposite, so the predicted '
             'shift magnitude is identical within a pair.</p>')
    # pairs: same orientation, opposite PE
    body += '<h3>Representative images</h3>'
    by_orient: Dict[str, list] = {}
    for r in results:
        by_orient.setdefault(r.scan.params.get('orientation', ''), []).append(r)
    for orient, rs in by_orient.items():
        brain = rs[0].arrays['brain']
        ks = figures.pick_slices(rs[0].arrays['core'], 7)
        rows = []
        for r in rs:
            m = r.arrays['mean']
            rows.append({'data': m, 'label': f"E{r.scan.number} {'rev' if r.scan.params.get('reversed_pe') == 'Yes' else 'fwd'}",
                         'vmin': 0, 'vmax': figures.window(m[r.arrays['core']])[1]})
        if len(rs) == 2 and rs[0].arrays['mean'].shape == rs[1].arrays['mean'].shape:
            a, b = rs[0].arrays['mean'], rs[1].arrays['mean']
            s = figures.window(a[rs[0].arrays['core']])[1]
            rows.append({'data': (a - b) / s, 'label': 'difference', 'cmap': 'RdBu_r', 'vmin': -0.5, 'vmax': 0.5,
                         'cbar': 'fraction of brain signal'})
        body += _fig(figures.montage(rows, brain, ks), f'{escape(orient)} pair: forward and reversed phase '
                     'encoding, and their difference (edges that flip sign along PE = susceptibility shift).')
    return body


def _bold_sweeps(results) -> List[Tuple[str, List]]:
    """Groups of BOLD scans that differ only in slice thickness/gap/count (>=3 thicknesses)."""
    keyf = lambda r: tuple(r.scan.params.get(k) for k in ('orientation', 'matrix', 'tr_ms', 'te_ms', 'repetitions',
                                                          'fov_mm', 'slice_gap_mm'))
    groups: Dict[tuple, list] = {}
    for r in results:
        groups.setdefault(keyf(r), []).append(r)
    out = [(k, sorted(v, key=lambda r: (r.scan.params['slice_thick_mm'], r.scan.number)))
           for k, v in groups.items() if len({r.scan.params.get('slice_thick_mm') for r in v}) >= 3]
    return sorted(out, key=lambda kv: -len(kv[1]))[:3]


def sec_bold(results) -> str:
    scans = [r.scan for r in results]
    labels = labels_for(scans)
    body = '<h3>Protocol parameters</h3><p class="note">Shaded columns differ between protocols. "Vol. TR" is the time per volume: these are segmented EPI, one TR per shot.</p>' + param_table(scans)

    groups = [r.scan.params.get('orientation', '').capitalize() + ' slices' for r in results]
    order = sorted(range(len(results)), key=lambda i: (groups[i], i))
    R = [results[i] for i in order]
    L = [re.sub(r' · (Ax|Cor|Sag)(?= ·|$)', '', labels[i]) for i in order]  # group header names the plane
    G = [groups[i] for i in order]

    def P(key, lo=None, hi=None, **kw):
        lab, better, f = METRICS[key]
        return {'title': lab, 'values': [r.metrics.get(key) for r in R], 'better': better,
                'lo': [r.metrics.get(lo) for r in R] if lo else None,
                'hi': [r.metrics.get(hi) for r in R] if hi else None,
                'fmt': lambda v, f=f: f.format(v), **kw}
    dots = charts.dot_rows(L, [P('tsnr', 'tsnr_q1', 'tsnr_q3'), P('tsnr_per_sqrt_s'), P('tsnr_per_nl_sqrt_s'),
                               P('ghost_pct'), P('coverage_sampled_pct'), P('disp_p95_mm')], groups=G,
                           panel_width=140)
    body += '<h3>Protocol comparison</h3>' + _chart(
        'Temporal SNR, efficiency and geometry for every BOLD protocol', dots,
        'tSNR: median over brain voxels, whiskers = interquartile range across voxels. tSNR/√TRvol puts '
        'protocols with different volume TR on one scale (tSNR per unit √time); the /nL version also divides '
        'out voxel volume. Ring = best value in that column.')
    sweeps = _bold_sweeps(results)
    if sweeps:
        names = []
        for k, rs in sweeps:
            p = rs[0].scan.params
            names.append(f"TR {p['tr_ms']:g} ms (vol {p.get('vol_tr_s', 0):.1f} s)")
        if len(set(names)) < len(names):
            names = [f"{n} · {rs[0].scan.params.get('orientation')}" for n, (k, rs) in zip(names, sweeps)]
        s1, s2 = [], []
        for i, ((k, rs), nm) in enumerate(zip(sweeps, names)):
            s1.append({'name': nm, 'slot': i + 1, 'points': [
                (r.scan.params['slice_thick_mm'], r.metrics.get('tsnr'),
                 f"E{r.scan.number} {nm}, {r.scan.params['slice_thick_mm']:.2f} mm "
                 f"(+{r.scan.params.get('slice_gap_mm') or 0:.1f} gap, {r.scan.params.get('n_slices')} sl): "
                 f"tSNR {r.metrics.get('tsnr'):.1f}") for r in rs]})
            s2.append({'name': nm, 'slot': i + 1, 'points': [
                (r.scan.params['slice_thick_mm'], r.metrics.get('coverage_sampled_pct'),
                 f"E{r.scan.number} {nm}: {r.metrics.get('coverage_sampled_pct'):.0f}% of brain voxels inside a slice")
                for r in rs]})
        body += '<h3>Slice-thickness sweep</h3><div class="grid2">' + _chart(
            'tSNR vs slice thickness', charts.line_chart(s1, 'slice thickness (mm)', 'tSNR (median)', aria='tSNR vs thickness'),
            'Scans identical except for slice thickness/gap. Thermal SNR is proportional to thickness; '
            'departures from a straight line through zero show other noise sources or slice-profile effects.', s1) + _chart(
            'Brain sampled vs slice thickness', charts.line_chart(s2, 'slice thickness (mm)', '% brain voxels inside a slice', aria='coverage vs thickness'),
            'Thicker slices with the same gap sample more of the brain; the slab extent is set by slices × (thickness + gap).', s2) + '</div>'
    long = [r for r in results if r.metrics.get('n_vols', 0) >= 30][:3]
    if long:
        ll = labels_for([r.scan for r in long]) if len(long) > 1 else [labels[results.index(long[0])]]
        ts_series, sl_series = [], []
        for i, (r, lab) in enumerate(zip(long, ll)):
            s = r.arrays['series']
            pct = 100 * (s / s.mean() - 1)
            tr = r.arrays['vol_tr']
            ts_series.append({'name': lab, 'slot': i + 1, 'points': [
                (k * tr, float(v), f'{lab}: t={k * tr:.0f} s, {v:+.2f}%') for k, v in enumerate(pct)]})
            sl = r.arrays['slice_tsnr']
            sl_series.append({'name': lab, 'slot': i + 1, 'points': [
                (k + 1, v, f'{lab}: slice {k + 1}, tSNR {v:.1f}') for k, v in enumerate(sl) if np.isfinite(v)]})
        body += '<h3>Long runs</h3><div class="grid2">' + _chart(
            'Brain-mean signal over time', charts.line_chart(ts_series, 'time (s)', 'signal change (%)', y_zero=False,
                                                            markers=False, aria='time course'),
            'Mean over the brain core, % of its run mean. Linear drift (%/min) is in the metrics table.', ts_series) + _chart(
            'tSNR by slice', charts.line_chart(sl_series, 'slice (acquisition order)', 'tSNR (median)', aria='slice tSNR'),
            'Coil sensitivity and coverage along the slab: the volume coil should give a flat profile.', sl_series) + '</div>'
    body += '<h3>Metrics</h3>' + metric_table(results, ROLE_METRICS['bold'])
    # gallery with a shared tSNR scale
    vmax = np.nanpercentile([r.metrics.get('tsnr_q3') or np.nan for r in results], 90) * 1.5
    body += ('<h3>Representative images</h3><p class="note">Top row: mean EPI with the anatomical brain '
             'outline (mismatch along phase encoding = distortion). Bottom row: tSNR on one colour scale for all '
             f'protocols (0–{vmax:.0f}), so maps are directly comparable.</p>')
    for r, lab in zip(results, labels):
        ks = figures.pick_slices(r.arrays['core'], 9)
        m = r.arrays['mean']
        rows = [{'data': m, 'label': 'mean', 'vmin': 0, 'vmax': figures.window(m[r.arrays['core']])[1]},
                {'data': r.arrays['tsnr'], 'label': 'tSNR', 'cmap': 'inferno', 'vmin': 0, 'vmax': vmax,
                 'cbar': 'tSNR', 'outline': False}]
        body += _fig(figures.montage(rows, r.arrays['brain'], ks, panel_in=0.95), escape(lab) + ' — ' + escape(r.scan.params['name']))
    return body


def sec_geometry(all_epi: List[Tuple[str, Any]]) -> str:
    if not all_epi:
        return ''
    R = [r for _, r in all_epi]
    groups = [g for g, _ in all_epi]
    labs = []
    for g, r in all_epi:
        labs.append(f'E{r.scan.number} {r.scan.params["name"]}'[:44])

    def P(key):
        lab, better, f = METRICS[key]
        return {'title': lab, 'values': [r.metrics.get(key) for r in R], 'better': better,
                'fmt': lambda v, f=f: f.format(v)}
    svg = charts.dot_rows(labs, [P('ghost_pct'), P('coverage_slab_pct'), P('coverage_sampled_pct'),
                                 P('disp_p95_mm'), P('mm_per_100hz')], groups=groups, label_width=300)
    return _chart('Ghosting, coverage and predicted distortion — all EPI', svg,
                  'Ghost: mean signal where Nyquist (N/2) and interleaved-segment (N/shots) ghosts of the head land, '
                  'minus clean background, as % of brain signal. Coverage: share of the anatomical brain mask inside '
                  'the slab ("in slab") and inside a slice\'s nominal thickness ("sampled", gaps excluded). '
                  'Predicted shift: |ΔB0| × FOV<sub>PE</sub> / effective PE bandwidth, 95th percentile over the brain.')


def sec_fieldmap_mrs(fm_results, mrs_results, ref) -> str:
    body = ''
    if fm_results:
        body += '<h3>B0 field maps</h3>' + param_table([r.scan for r in fm_results])
        body += metric_table(fm_results, ROLE_METRICS['fieldmap'])
        series = []
        for i, r in enumerate(fm_results[:3]):
            v = r.arrays['values']
            h, e = np.histogram(v, bins=60, range=(-400, 400), density=True)
            c = 0.5 * (e[1:] + e[:-1])
            name = f'E{r.scan.number} {r.scan.params["name"]}'
            series.append({'name': name, 'slot': i + 1, 'points': [
                (x, y * 100, f'{name}: {x:+.0f} Hz, {y * 100:.2f} %/Hz') for x, y in zip(c, h)]})
        body += '<div class="grid2">' + _chart('Off-resonance in the brain', charts.line_chart(
            series, 'ΔB0 relative to brain median (Hz)', 'density (% of voxels per Hz)', markers=False,
            aria='B0 histogram'), 'Narrower = better shim. 100 Hz ≈ 0.33 ppm at 7 T.', series)
        vols = [{'data': np.where(r.arrays['hz'] != 0, r.arrays['hz'] - np.median(r.arrays['hz'][r.arrays['brain']]), np.nan),
                 'aff': r.arrays['aff'], 'label': f'E{r.scan.number}', 'mask': ref.mask, 'mask_aff': ref.affine,
                 'cmap': 'RdBu_r', 'window': (-300, 300)} for r in fm_results]
        body += _fig(figures.ortho(vols, ref.center, np.array([6.5, 5.0, 8.0]), res=0.1),
                     'Field maps (Hz, −300 blue to +300 red, centred on the brain median) with the brain outline.') + '</div>'
    if mrs_results:
        body += '<h3>Water linewidth (shim voxels)</h3>' + param_table([r.scan for r in mrs_results])
        body += metric_table(mrs_results, ROLE_METRICS['mrs'])
        series = []
        for i, r in enumerate(mrs_results[:3]):
            sp = r.arrays.get('spectrum')
            if sp is None:
                continue
            f, y = sp['freq_hz'], sp['real']
            m = np.abs(f) <= 300
            step = max(1, int(m.sum() / 300))
            name = f'E{r.scan.number} {r.scan.params["name"]}'
            series.append({'name': name, 'slot': i + 1, 'points': [
                (float(a), float(b / y.max()), f'{name}: {a:+.1f} Hz, {b / y.max():.3f}') for a, b in zip(f[m][::step], y[m][::step])]})
        body += _chart('Unsuppressed water, absorption mode', charts.line_chart(
            series, 'frequency relative to water (Hz)', 'normalised signal', markers=False, y_zero=False,
            aria='water spectra'), 'FWHM of the phased real spectrum (magnitude FWHM in the table is ~√3× wider for a Lorentzian line).', series)
    return body


PLANE_NOTE = ('Planes are named in scanner coordinates; for a rodent lying prone in a horizontal bore, scanner '
              '<em>axial</em> is the anatomical coronal plane and scanner <em>coronal</em> is anatomical horizontal. '
              'The same applies to the "Ax"/"Cor" protocol names.')

METHODS_HTML = """
<ul>
<li><strong>Geometry.</strong> Every scan is placed in the scanner's subject coordinates from <code>visu_pars</code>
(<code>VisuCoreOrientation</code>, <code>VisuCorePosition</code>, extent). Masks move between scans by resampling in
world space; nothing is registered, so residual mismatch is genuine (EPI distortion, motion between scans).</li>
<li><strong>Brain mask</strong> on the highest-resolution 3-D anatomical scan: FSL <code>bet2</code> hull (10× voxel scaling,
second pass seeded at the first pass's centre), refined by keeping voxels at 0.85–1.5× the median parenchymal intensity,
opening/closing and hole filling. In each scan the mask is eroded one voxel in-plane ("core") before measuring signal.</li>
<li><strong>Noise.</strong> (1) Difference method (NEMA): σ = SD(A − B)/√2 over the brain for two identical acquisitions
(DTI repetitions; consecutive BOLD volumes). (2) Air background: SD of background voxels outside the dilated head, away
from ghost locations, divided by the SD of pure magnitude noise for the number of receive channels
(0.655 for one channel, 0.683 for two); only used when the background mean/SD ratio is 1.6–3.2, as noise should give.
(3) Otherwise (tight 3-D FOV with no air around the head): the same, in the darkest 1% of 5-voxel neighbourhoods
inside the FOV (airways, ear canals), again only if their mean/SD looks like noise; (4) last resort: robust SD of the
Laplacian-filtered image inside the brain.</li>
<li><strong>SNR</strong> = mean brain-core signal / σ, for one image. <strong>Efficiency</strong> divides out √(scan time) and
voxel volume (1 nL = 0.001 mm³), the two things SNR is traded against.</li>
<li><strong>tSNR</strong>: voxelwise mean / SD of the linearly detrended time series (dof-corrected), median over the brain core.
Runs of 10 volumes give a tSNR dominated by thermal noise; 100-volume runs include physiological fluctuation and drift.</li>
<li><strong>Ghosting</strong>: Otsu head mask of the mean image, shifted by N/2 and by k·N/shots along phase encoding (wrapped).
Ghost % = (mean in ghost region − mean in clean background) / mean brain signal.</li>
<li><strong>Predicted distortion</strong>: the B0 map (Hz) acquired before the scan is resampled into the EPI grid and centred on its brain
median; displacement = Δf · FOV<sub>PE</sub> / <code>PVM_EpiEffBandwidth</code>. It assumes the shim did not change between the map and
the scan.</li>
<li><strong>DTI</strong>: weighted log-linear tensor fit on b ≤ 1000 s/mm² using the scanner's effective b-matrices
(<code>PVM_DwBMat</code>); FA/MD medians over the brain core (whole brain, not white matter).</li>
<li><strong>Field maps</strong>: SD/IQR of ΔB0 over the brain after removing the median. <strong>Spectroscopy</strong>: FWHM of
the unsuppressed water peak (group delay removed, zero-order phased).</li>
<li><strong>Not assessed</strong>: motion correction, slice-timing, registration accuracy to an atlas, physiological recordings.
Single subject, single session: differences of a few percent are within measurement noise.</li>
</ul>
"""

CSS = """
:root{color-scheme:light;--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--vary:#eaf2fc;--best:#0b0b0b}
@media (prefers-color-scheme: dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;
--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--vary:#1d2a3a;--best:#f0efec}}
:root[data-theme="dark"]{color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--vary:#1d2a3a;--best:#f0efec}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
.viz-root{background:var(--page);color:var(--ink);min-height:100vh}
section,table,td,th,p,li,h1,h2,h3,h4{color:inherit}
main{max-width:1280px;margin:0 auto;padding:24px 28px 80px}
header h1{font-size:24px;margin:0 0 4px} header .sub{color:var(--ink2)}
nav{position:sticky;top:0;background:var(--page);border-bottom:1px solid var(--grid);padding:8px 0;margin:16px 0 8px;z-index:2;display:flex;flex-wrap:wrap;gap:4px 16px;font-size:13px}
nav a{color:var(--ink2);text-decoration:none} nav a:hover{color:var(--ink)}
section{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:18px 22px;margin:18px 0}
h2{font-size:19px;margin:0 0 6px} h3{font-size:15px;margin:22px 0 8px} h4{font-size:13px;margin:0 0 6px;font-weight:600}
.lead{color:var(--ink2);margin-top:0} .note{color:var(--ink2);font-size:12.5px}
code{font-size:12.5px;background:var(--page);color:var(--ink);padding:0 3px;border-radius:3px}
.tablewrap{overflow-x:auto;margin:6px 0 10px}
table{border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}
th{text-align:left;font-weight:600;color:var(--ink2);border-bottom:1px solid var(--axis);padding:4px 8px;white-space:nowrap;vertical-align:bottom}
td{border-bottom:1px solid var(--grid);padding:3px 8px;white-space:nowrap} td.num{text-align:right}
th.vary,td.vary{background:var(--vary)}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px;margin:10px 0}
.tile{border:1px solid var(--ring);border-radius:8px;padding:10px 12px;background:var(--surface)}
.tile .k{font-size:12px;color:var(--ink2)} .tile .v{font-size:22px;font-weight:600} .tile .d{font-size:12px;color:var(--ink2)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:16px;align-items:start}
.chartcard{border:1px solid var(--ring);border-radius:8px;padding:12px 14px;margin:8px 0;background:var(--surface);overflow-x:auto}
figure{margin:10px 0 18px;background:#fff;color:#0b0b0b;border-radius:6px;padding:6px;overflow-x:auto} figure img{max-width:100%;display:block}
figcaption{color:#52514e;font-size:12.5px;padding:4px 2px 0}
svg.chart{width:100%;height:auto;max-width:760px;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;overflow:visible}
svg.chart.dots{width:auto;max-width:none}
svg .grid{stroke:var(--grid);stroke-width:1} svg .axis{stroke:var(--axis);stroke-width:1}
svg .tick{fill:var(--muted);font-size:10.5px} svg .label{fill:var(--ink2);font-size:11.5px}
svg .rowlabel{fill:var(--ink);font-size:11.5px} svg .ptitle{fill:var(--ink);font-size:11.5px;font-weight:600}
svg .hint{fill:var(--muted);font-size:10px} svg .group{fill:var(--ink2);font-size:11px;font-weight:600}
svg .endlabel{fill:var(--ink2);font-size:11px} svg .bestlabel{fill:var(--ink);font-size:10.5px;font-weight:600}
svg .line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
svg .line.s1,svg .lead.s1{stroke:var(--s1)} svg .line.s2,svg .lead.s2{stroke:var(--s2)} svg .line.s3,svg .lead.s3{stroke:var(--s3)}
svg .lead{stroke-width:2}
svg .dot{stroke:var(--surface);stroke-width:2} svg .dot.s1{fill:var(--s1)} svg .dot.s2{fill:var(--s2)} svg .dot.s3{fill:var(--s3)}
svg .dot.best{stroke:var(--best);stroke-width:2}
svg .whisker{stroke:var(--s1);stroke-width:2;stroke-linecap:round;opacity:.45}
svg .refline{stroke:var(--axis);stroke-width:1}
svg .hit{fill:transparent} svg .pt:hover .dot{r:6.5}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--ink2);margin:0 0 4px}
.key{display:inline-flex;align-items:center;gap:6px} .sw{width:12px;height:12px;border-radius:3px;display:inline-block}
.sw.s1{background:var(--s1)} .sw.s2{background:var(--s2)} .sw.s3{background:var(--s3)}
.notes h3,.notes h4{margin-top:14px} .notes li{margin:3px 0}
.themebtn{float:right;font-size:12px;background:none;border:1px solid var(--axis);color:var(--ink2);border-radius:6px;padding:3px 8px;cursor:pointer}
"""


def highlights(res: Dict[str, list]) -> str:
    tiles = []

    def best(role, key):
        rs = [r for r in res.get(role, []) if r.metrics.get(key) is not None and np.isfinite(r.metrics.get(key))]
        if len(rs) < 2:
            return
        lab, better, _ = METRICS[key]
        f = max if better == 'high' else min
        b = f(rs, key=lambda r: r.metrics[key])
        others = [r.metrics[key] for r in rs if r is not b]
        rng = f'{fmt_metric(key, min(others))}–{fmt_metric(key, max(others))}' if len(others) > 1 else fmt_metric(key, others[0])
        tiles.append(f'<div class="tile"><div class="k">{escape(ROLES[role].split(" (")[0])} · {escape(lab)} '
                     f'({"higher" if better == "high" else "lower"} is better)</div>'
                     f'<div class="v">{fmt_metric(key, b.metrics[key])}</div>'
                     f'<div class="d">E{b.scan.number} {escape(b.scan.params["name"])} · others {rng}</div></div>')
    for role, key in [('anat', 'snr_per_nl_sqrt_min'), ('dwi', 'snr_b0'), ('dwi', 'snr_b2000'),
                      ('bold', 'tsnr'), ('bold', 'tsnr_per_sqrt_s'), ('bold', 'ghost_pct'),
                      ('fieldmap', 'b0_sd_hz'), ('mrs', 'fwhm_hz')]:
        best(role, key)
    return '<div class="tiles">' + ''.join(tiles) + '</div>' if tiles else ''


def build_html(info: Dict[str, str], scans: List[Scan], out: Dict[str, Any], notes_md: Optional[str] = None,
               version: str = '') -> str:
    ref = out['reference']
    res = out['results']
    analysed = {r.scan.number for rs in res.values() for r in rs} | {ref.scan.number}
    nav, secs = [], []

    def add(id_, title, body, lead=''):
        if body:
            nav.append(f'<a href="#{id_}">{escape(title)}</a>')
            secs.append(_section(id_, title, body, lead))
    summary = highlights(res)
    if notes_md:
        summary += '<div class="notes">' + md_to_html(notes_md) + '</div>'
    add('summary', 'Summary', summary, 'Best protocol per key metric, among the scans that measure it.')
    add('inventory', 'Scans', sec_inventory(scans, analysed),
        f'{len(scans)} scans; {len(analysed)} analysed. Localizers and EPI ghost-correction adjustment scans '
        '(<code>ADJ_GHOST_*</code>, acquired automatically before each EPI) are listed but not analysed.')
    add('mask', 'Brain mask', sec_reference(ref))
    if res['anat']:
        add('anat', ROLES['anat'], sec_anat(res['anat'], ref))
    if res['dwi']:
        add('dwi', ROLES['dwi'], sec_dwi(res['dwi']))
    if res['epi_ref']:
        add('seepi', ROLES['epi_ref'], sec_epi_ref(res['epi_ref']))
    if res['bold']:
        add('bold', ROLES['bold'], sec_bold(res['bold']))
    epi = [(ROLES[r_][: ROLES[r_].find(' (') if ' (' in ROLES[r_] else None], r)
           for r_ in ('dwi', 'epi_ref', 'bold') for r in res[r_]]
    add('geometry', 'EPI geometry across protocols', sec_geometry(epi))
    add('shim', 'Shim: field maps and water linewidth', sec_fieldmap_mrs(res['fieldmap'], res['mrs'], ref))
    add('methods', 'Methods', METHODS_HTML)
    meta = ' · '.join(escape(v) for v in [info.get('subject_id', ''), info.get('date', ''), info.get('station', ''),
                                          info.get('software', '')] if v)
    gen = _dt.datetime.now().strftime('%Y-%m-%d %H:%M')
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>QC · {escape(info.get('name', ''))}</title><style>{CSS}</style></head>
<body><div class="viz-root"><main>
<header><button class="themebtn" onclick="var r=document.documentElement;var dark=r.dataset.theme?r.dataset.theme==='dark':matchMedia('(prefers-color-scheme: dark)').matches;r.dataset.theme=dark?'light':'dark'">light / dark</button>
<h1>Acquisition QA/QC report</h1>
<div class="sub">{meta}</div>
<div class="sub"><code>{escape(info.get('study_dir', ''))}</code></div>
<div class="sub">Generated {gen} by brukerview {escape(version)} <code>brukerview qc</code></div></header>
<nav>{''.join(nav)}</nav>
{''.join(secs)}
</main></div></body></html>"""


# ---- tables on disk --------------------------------------------------------------------------------
def write_tsv(path: Path, scans: List[Scan], out: Dict[str, Any]) -> None:
    rows = []
    for role, rs in out['results'].items():
        for r in rs:
            row = {'scan': r.scan.number, 'role': role, 'protocol': r.scan.params['name']}
            row.update({k: v for k, v in r.scan.params.items() if k not in ('scan', 'name')})
            row.update({k: v for k, v in r.metrics.items() if not k.startswith('_')})
            rows.append(row)
    keys: List[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(keys)
        for r in sorted(rows, key=lambda r: r['scan']):
            w.writerow(['' if r.get(k) is None else (f'{r[k]:.6g}' if isinstance(r.get(k), (float, np.floating)) else r.get(k))
                        for k in keys])


def write_params_tsv(path: Path, scans: List[Scan]) -> None:
    keys: List[str] = ['scan', 'role']
    for s in scans:
        keys += [k for k in s.params if k not in keys]
    with open(path, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(keys)
        for s in scans:
            d = dict(s.params, role=s.role)
            w.writerow(['' if d.get(k) is None else d.get(k) for k in keys])
