"""Representative images for the report (matplotlib -> base64 PNG)."""

from __future__ import annotations

import base64
import io
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .geometry import axis_aligned_slices  # noqa: E402

OUTLINE = '#1baf7a'   # brain outline (categorical slot 3, reads on gray)
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 8, 'axes.titlesize': 8,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})


def to_uri(fig, dpi: int = 100) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def _window(vals: np.ndarray, lo: float = 1, hi: float = 99.5):
    v = vals[np.isfinite(vals)]
    if v.size == 0:
        return 0, 1
    a, b = np.percentile(v, [lo, hi])
    return float(a), float(b if b > a else a + 1)


def _bbox(mask: np.ndarray, margin: int, axes=(0, 1)):
    idx = np.argwhere(mask)
    if idx.size == 0:
        return tuple(slice(None) for _ in range(mask.ndim))
    sl = [slice(None)] * mask.ndim
    for a in axes:
        lo, hi = idx[:, a].min() - margin, idx[:, a].max() + margin + 1
        sl[a] = slice(max(lo, 0), min(hi, mask.shape[a]))
    return tuple(sl)


def _off(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def pick_slices(mask: np.ndarray, n: int) -> List[int]:
    """Up to ``n`` evenly spaced slices that contain brain."""
    have = [k for k in range(mask.shape[2]) if mask[..., k].sum() > 20]
    if not have:
        have = list(range(mask.shape[2]))
    if len(have) <= n:
        return have
    return [have[int(round(i))] for i in np.linspace(0, len(have) - 1, n)]


def montage(rows: Sequence[Dict], brain: np.ndarray, slices: Sequence[int], margin: int = 6,
            title: Optional[str] = None, panel_in: float = 1.15) -> str:
    """Rows of native slices, all cropped to the brain bounding box.

    Each row: ``{'data', 'label', 'cmap', 'vmin', 'vmax', 'outline', 'cbar'}``.
    """
    crop = _bbox(brain, margin)
    ncol = len(slices)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(panel_in * ncol + 0.9, panel_in * len(rows) + 0.25),
                             squeeze=False)
    for r, row in enumerate(rows):
        im = None
        for c, k in enumerate(slices):
            ax = axes[r, c]
            img = row['data'][crop][..., k]
            im = ax.imshow(img.T, origin='lower', cmap=row.get('cmap', 'gray'),
                           vmin=row.get('vmin'), vmax=row.get('vmax'), interpolation='nearest')
            if row.get('outline', True):
                ax.contour(brain[crop][..., k].T.astype(float), levels=[0.5], colors=OUTLINE, linewidths=0.6)
            _off(ax)
            if r == 0:
                ax.set_title(f'slice {k + 1}', color='#52514e')
            if c == 0:
                ax.set_ylabel(row.get('label', ''), color='#0b0b0b')
        if row.get('cbar') and im is not None:
            cb = fig.colorbar(im, ax=axes[r, :].tolist(), fraction=0.012, pad=0.01)
            cb.set_label(row['cbar'], color='#52514e')
            cb.outline.set_visible(False)
    if title:
        fig.suptitle(title, x=0.01, ha='left', fontsize=9)
    return to_uri(fig)


def ortho(volumes: Sequence[Dict], center: np.ndarray, half_extent: np.ndarray,
          res: float = 0.05, plane_names=('x-plane · sagittal', 'y-plane · scanner coronal',
                                          'z-plane · scanner axial')) -> str:
    """Rows of three world-axis-aligned planes through ``center``.

    Each row ``{'data', 'aff', 'label', 'mask'?, 'mask_aff'?, 'cmap'?, 'window'?}``.
    Sampling is nearest-neighbour on a fine grid so native voxel size stays visible.
    """
    fig, axes = plt.subplots(len(volumes), 3, figsize=(9.6, 2.6 * len(volumes)), squeeze=False,
                             gridspec_kw={'width_ratios': [2 * half_extent[2], 2 * half_extent[0], 2 * half_extent[0]]})
    for r, v in enumerate(volumes):
        planes = axis_aligned_slices(v['data'], v['aff'], center, half_extent, res, order=0)
        mplanes = (axis_aligned_slices(v['mask'].astype(np.float32), v.get('mask_aff', v['aff']), center,
                                       half_extent, res, order=0) if v.get('mask') is not None else None)
        # sagittal: put the long (z) axis horizontally
        planes[0] = planes[0][::-1].T[::-1]
        if mplanes is not None:
            mplanes[0] = mplanes[0][::-1].T[::-1]
        vals = np.concatenate([p[np.isfinite(p)] for p in planes])
        lo, hi = v.get('window') or _window(vals)
        for c, pl in enumerate(planes):
            ax = axes[r, c]
            ax.imshow(pl, cmap=v.get('cmap', 'gray'), vmin=lo, vmax=hi, interpolation='nearest', aspect='equal')
            if mplanes is not None:
                ax.contour(np.nan_to_num(mplanes[c]), levels=[0.5], colors=OUTLINE, linewidths=0.7)
            _off(ax)
            if r == 0:
                ax.set_title(plane_names[c], color='#52514e')
            if c == 0:
                ax.set_ylabel(v.get('label', ''), color='#0b0b0b')
    return to_uri(fig)


def dwi_strip(shells: Dict[float, np.ndarray], FA: Optional[np.ndarray], MD: Optional[np.ndarray],
                      brain: np.ndarray, k: int, label: str) -> str:
    """One slice of a DWI: mean image of each shell, then FA and MD."""
    crop = _bbox(brain, 6)
    panels = [(f'b={int(b)}', img, 'gray', _window(img[brain]) if brain.any() else (None, None))
              for b, img in sorted(shells.items())]
    if FA is not None:
        panels.append(('FA', FA, 'gray', (0, 0.8)))
    if MD is not None:
        panels.append(('MD (µm²/ms)', MD * 1000, 'gray', (0, 1.2)))
    fig, axes = plt.subplots(1, len(panels), figsize=(1.5 * len(panels), 1.75), squeeze=False)
    for ax, (t, img, cmap, (lo, hi)) in zip(axes[0], panels):
        if t.startswith('b='):
            lo, hi = 0, hi
        ax.imshow(img[crop][..., k].T, origin='lower', cmap=cmap, vmin=lo, vmax=hi, interpolation='nearest')
        _off(ax)
        ax.set_title(t, color='#52514e')
    axes[0, 0].set_ylabel(label)
    return to_uri(fig)


def window(vals, lo=1, hi=99.5):
    return _window(np.asarray(vals), lo, hi)
