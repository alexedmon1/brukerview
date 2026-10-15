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
  space               play / pause movie
  d                   movie dimension (slice / echo / diffusion / time)
  + / -               movie faster / slower
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


TIME_NAMES = ('cycle', 'movie', 'repetition', 'frame')


class SliceViewer:
    def __init__(self, img: BrukerImage, scaled: bool = True, cmap: str = 'gray',
                 vrange: Optional[Tuple[float, float]] = None, fps: float = 10.0,
                 movie_dim: Optional[str] = None):
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

        # movie playback: a timer steps one dimension and wraps around
        self.fps = float(fps)
        self.movie_dim = self.resolve_dim(movie_dim) if movie_dim else self.default_movie_dim()
        self.playing = False
        self._timer = self.fig.canvas.new_timer(interval=self._interval())
        self._timer.add_callback(self._tick)

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self._drag = None
        self._update()

    # ---- data access ---------------------------------------------------------
    def _plane(self, idx: Optional[Sequence[int]] = None) -> np.ndarray:
        sel = (slice(None), slice(None), *(self.idx if idx is None else idx))
        return self.data[sel].T

    def _title_text(self, idx: Sequence[int]) -> str:
        parts = [self.img.title]
        parts += [self.dim_labels[d][idx[d]] for d in range(len(idx)) if self.data.shape[2 + d] > 1]
        return '   '.join(parts)

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
        text = f'{self._title_text(self.idx)}   [{self.lo:.4g}, {self.hi:.4g}]'
        if self.playing:
            text += f'   \u25b6 {self.dim_names[self.movie_dim]} {self.fps:g} fps'
        self.title.set_text(text)
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
        elif k == ' ':
            self.toggle_play()
        elif k == 'd':
            self.cycle_movie_dim()
        elif k in ('+', '='):
            self.set_fps(self.fps * 1.5)
        elif k in ('-', '_'):
            self.set_fps(self.fps / 1.5)
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

    # ---- movie ----------------------------------------------------------------------
    def movie_dims(self) -> List[int]:
        """Dimensions that can be played (more than one position)."""
        return [d for d in range(self.data.ndim - 2) if self.data.shape[2 + d] > 1]

    def default_movie_dim(self) -> int:
        """The longest non-slice dimension (time, diffusion, echo), else slices."""
        extra = [d for d in self.movie_dims() if d > 0]
        if extra:
            return max(extra, key=lambda d: (self.data.shape[2 + d], -d))
        return 0

    def resolve_dim(self, name: str) -> int:
        """Dimension index from a name: 'slice', a frame-group label ('echo',
        'diffusion', 'cycle', ...) or 'time' for whichever time-like group exists."""
        key = name.strip().lower()
        if key == 'time':
            for d, nm in enumerate(self.dim_names):
                if nm in TIME_NAMES:
                    return d
        elif key in self.dim_names:
            return self.dim_names.index(key)
        raise ValueError(f'no {name!r} dimension; this image has: {", ".join(self.dim_names)}')

    def _interval(self) -> int:
        return max(10, int(round(1000.0 / self.fps)))

    def _tick(self):
        d = self.movie_dim
        self._set_index(d, (self.idx[d] + 1) % self.data.shape[2 + d])

    def toggle_play(self):
        if self.data.shape[2 + self.movie_dim] < 2:
            print(f'nothing to play: {self.dim_names[self.movie_dim]} has one position')
            return
        self.playing = not self.playing
        (self._timer.start if self.playing else self._timer.stop)()
        self._update()

    def set_fps(self, fps: float):
        self.fps = float(min(max(fps, 0.5), 60.0))
        self._timer.interval = self._interval()
        self._update()

    def cycle_movie_dim(self):
        dims = self.movie_dims()
        if not dims:
            return
        i = dims.index(self.movie_dim) if self.movie_dim in dims else -1
        self.movie_dim = dims[(i + 1) % len(dims)]
        print(f'movie dimension: {self.dim_names[self.movie_dim]}')
        self._update()

    def save_movie(self, path: str, dim: Optional[str] = None, fps: Optional[float] = None, dpi: int = 100) -> str:
        """Write the movie dimension (at the current position of the others) to
        ``.gif`` (Pillow) or ``.mp4``/``.mov``/``.avi``/``.mkv`` (needs ffmpeg)."""
        import matplotlib.animation as animation
        from pathlib import Path

        d = self.resolve_dim(dim) if dim else self.movie_dim
        fps = float(fps or self.fps)
        suffix = Path(path).suffix.lower()
        if suffix == '.gif':
            writer = animation.PillowWriter(fps=fps)
        elif suffix in ('.mp4', '.mov', '.avi', '.mkv'):
            if not animation.writers.is_available('ffmpeg'):
                raise SystemExit(f'{suffix} needs ffmpeg on PATH; use a .gif instead')
            writer = animation.FFMpegWriter(fps=fps)
        else:
            raise SystemExit(f'unsupported movie format {suffix!r}: use .gif or .mp4')

        idx = list(self.idx)
        dx, dy, _ = self.img.spacing
        nx, ny = self.data.shape[:2]
        height = 6.0 * (ny * dy) / (nx * dx)
        fig = self.plt.figure(figsize=(6.0, height + 0.45))
        ax = fig.add_axes([0, 0, 1, height / (height + 0.45)])
        ax.set_axis_off()
        im = ax.imshow(self._plane(idx), cmap=self.im.get_cmap(), vmin=self.lo, vmax=self.hi,
                       aspect=dy / dx, interpolation='nearest', origin='upper')
        title = fig.text(0.5, 0.99, '', ha='center', va='top', fontsize=8)

        def draw(i):
            idx[d] = i
            im.set_data(self._plane(idx))
            title.set_text(self._title_text(idx))
            return im, title

        anim = animation.FuncAnimation(fig, draw, frames=self.data.shape[2 + d], blit=False)
        anim.save(str(path), writer=writer, dpi=dpi)
        self.plt.close(fig)
        return str(path)

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
