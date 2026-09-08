"""Interactive matplotlib slice viewer for :class:`BrukerImage`."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .reader import BrukerImage

# Every GUI backend is named "...Agg" (TkAgg, QtAgg, WXAgg), so a substring test
# for "agg" also matches working interactive backends.
_NON_GUI_BACKENDS = {'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template'}


def _is_headless(plt) -> bool:
    return plt.get_backend().lower() in _NON_GUI_BACKENDS


HELP = """\
brukerview keys
  up/down or scroll   slice
  left/right          next dimension (echo / diffusion / time)
  [ / ]               third dimension, if any
  home / end          first / last slice
  right-drag          window (horizontal) and level (vertical)
  a                   auto contrast (0.5 - 99.5 percentile)
  r                   reset contrast to full range
  i                   invert colormap
  m                   montage of all slices (new window)
  h                   print this help
  s                   save figure (matplotlib)
  q                   close
"""


def _robust_range(data: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> Tuple[float, float]:
    flat = data.reshape(-1)
    step = max(1, flat.size // 2_000_000)
    sample = flat[::step]
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(sample, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def montage_array(vol: np.ndarray, cols: Optional[int] = None) -> np.ndarray:
    """Tile a ``(nx, ny, nz)`` volume into a 2D image (rows of slices)."""
    nx, ny, nz = vol.shape
    cols = cols or math.ceil(math.sqrt(nz))
    rows = math.ceil(nz / cols)
    out = np.full((rows * ny, cols * nx), np.nan, dtype=np.float32)
    for k in range(nz):
        r, c = divmod(k, cols)
        out[r * ny:(r + 1) * ny, c * nx:(c + 1) * nx] = vol[:, :, k].T
    return out


class SliceViewer:
    def __init__(self, img: BrukerImage, scaled: bool = True, cmap: str = 'gray',
                 vrange: Optional[Tuple[float, float]] = None):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        self.plt = plt
        self.img = img
        self.data = img.load(scaled=scaled)
        if self.data.ndim < 3:
            self.data = self.data.reshape(self.data.shape + (1,) * (3 - self.data.ndim))
        self.cmap = cmap
        self.inverted = False
        self.idx: List[int] = [0] * (self.data.ndim - 2)

        nz = self.data.shape[2]
        self.dim_names = ['slice'] + [g.label for g in img.extra_groups]
        self.dim_labels = [[f'slice {i + 1}/{nz}' for i in range(nz)]]
        self.dim_labels += [img.group_labels(g) for g in img.extra_groups]
        # guard against label list / axis length mismatch
        for d in range(1, len(self.dim_labels)):
            n = self.data.shape[2 + d]
            if len(self.dim_labels[d]) != n:
                self.dim_labels[d] = [f'{self.dim_names[d]} {i + 1}/{n}' for i in range(n)]

        self.full_range = (float(np.nanmin(self.data)), float(np.nanmax(self.data)))
        self.lo, self.hi = vrange or _robust_range(self.data)

        nav = [d for d in range(self.data.ndim - 2) if self.data.shape[2 + d] > 1]
        n_sliders = len(nav)
        self.fig = plt.figure(figsize=(7, 7 + 0.35 * n_sliders))
        try:
            self.fig.canvas.manager.set_window_title(f'brukerview - {img.scan_name} {img.protocol}')
        except Exception:
            pass
        bottom = 0.04 + 0.05 * n_sliders
        self.ax = self.fig.add_axes([0.05, bottom + 0.02, 0.9, 0.93 - bottom])
        self.ax.set_axis_off()

        dx, dy, _ = img.spacing
        self.im = self.ax.imshow(self._plane(), cmap=cmap, vmin=self.lo, vmax=self.hi,
                                 aspect=dy / dx, interpolation='nearest', origin='upper')
        self.ax.format_coord = self._format_coord
        self.title = self.fig.text(0.5, 0.975, '', ha='center', va='top', fontsize=10)

        self.sliders = {}
        for i, d in enumerate(nav):
            sax = self.fig.add_axes([0.15, 0.03 + 0.05 * (n_sliders - 1 - i), 0.75, 0.03])
            s = Slider(sax, self.dim_names[d], 1, self.data.shape[2 + d],
                       valinit=1, valstep=1, valfmt='%d')
            s.on_changed(lambda val, d=d: self._set_index(d, int(val) - 1, from_slider=True))
            self.sliders[d] = s

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self._drag = None
        self._update()

    # ---- data access ---------------------------------------------------------
    def _plane(self) -> np.ndarray:
        sel = (slice(None), slice(None), *self.idx)
        return self.data[sel].T

    def _format_coord(self, x, y) -> str:
        xi, yi = int(round(x)), int(round(y))
        nx, ny = self.data.shape[:2]
        if 0 <= xi < nx and 0 <= yi < ny:
            val = self.data[(xi, yi, *self.idx)]
            return f'x={xi} y={yi}  value={val:.4g}'
        return ''

    # ---- navigation ------------------------------------------------------------
    def _set_index(self, d: int, value: int, from_slider: bool = False):
        n = self.data.shape[2 + d]
        value = max(0, min(n - 1, value))
        if value == self.idx[d]:
            return
        self.idx[d] = value
        if not from_slider and d in self.sliders:
            self.sliders[d].set_val(value + 1)   # triggers _set_index again -> no-op
        self._update()

    def _step(self, d: int, delta: int):
        if d < len(self.idx):
            self._set_index(d, self.idx[d] + delta)

    def _update(self):
        self.im.set_data(self._plane())
        self.im.set_clim(self.lo, self.hi)
        parts = [self.img.title]
        parts += [self.dim_labels[d][self.idx[d]] for d in range(len(self.idx)) if self.data.shape[2 + d] > 1]
        parts.append(f'[{self.lo:.4g}, {self.hi:.4g}]')
        self.title.set_text('   '.join(parts))
        self.fig.canvas.draw_idle()

    # ---- events -------------------------------------------------------------------
    def _on_key(self, event):
        k = event.key
        if k in ('up', 'down'):
            self._step(0, 1 if k == 'up' else -1)
        elif k in ('left', 'right'):
            self._step(1, 1 if k == 'right' else -1)
        elif k in ('[', ']'):
            self._step(2, 1 if k == ']' else -1)
        elif k == 'home':
            self._set_index(0, 0)
        elif k == 'end':
            self._set_index(0, self.data.shape[2] - 1)
        elif k == 'a':
            self.lo, self.hi = _robust_range(self.data)
            self._update()
        elif k == 'r':
            self.lo, self.hi = self.full_range
            self._update()
        elif k == 'i':
            self.inverted = not self.inverted
            self.im.set_cmap(self.cmap + ('_r' if self.inverted else ''))
            self.fig.canvas.draw_idle()
        elif k == 'm':
            self.show_montage()
        elif k == 'h':
            print(HELP)

    def _on_scroll(self, event):
        self._step(0, 1 if event.button == 'up' else -1)

    def _on_press(self, event):
        if event.button == 3 and event.inaxes is self.ax:
            self._drag = (event.x, event.y, self.lo, self.hi)

    def _on_release(self, event):
        self._drag = None

    def _on_motion(self, event):
        if self._drag is None:
            return
        x0, y0, lo0, hi0 = self._drag
        width0 = hi0 - lo0
        level0 = (hi0 + lo0) / 2
        span = self.full_range[1] - self.full_range[0] or 1.0
        width = width0 * math.exp((event.x - x0) / 150.0)
        level = level0 + (event.y - y0) / 300.0 * span
        self.lo, self.hi = level - width / 2, level + width / 2
        self._update()

    # ---- extras -------------------------------------------------------------------
    def show_montage(self):
        vol = self.data[(slice(None), slice(None), slice(None), *self.idx[1:])]
        tiles = montage_array(vol)
        dx, dy, _ = self.img.spacing
        fig, ax = self.plt.subplots(figsize=(9, 9))
        ax.imshow(tiles, cmap=self.im.get_cmap(), vmin=self.lo, vmax=self.hi,
                  aspect=dy / dx, interpolation='nearest')
        ax.set_axis_off()
        extra = '   '.join(self.dim_labels[d][self.idx[d]] for d in range(1, len(self.idx)))
        ax.set_title(f'{self.img.title}   {extra}', fontsize=10)
        fig.tight_layout()
        if not _is_headless(self.plt):
            fig.show()

    def savefig(self, path: str, dpi: int = 150):
        self.fig.savefig(path, dpi=dpi)

    def run(self):
        if _is_headless(self.plt):
            # No GUI backend: plt.show() would return silently and open nothing.
            raise SystemExit(
                'matplotlib has no interactive backend, so no window can open.\n'
                'Install a GUI toolkit (Debian/Ubuntu: apt install python3-tk; '
                'or pip install PyQt5), or use --save out.png for a snapshot.')
        print(HELP)
        self.plt.show()


def show(img: BrukerImage, **kwargs) -> SliceViewer:
    v = SliceViewer(img, **kwargs)
    v.run()
    return v
