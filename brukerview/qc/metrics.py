"""Image-quality metrics. Pure numpy/scipy; every function takes arrays + masks.

Conventions
-----------
* "brain" masks are the anatomical brain mask carried into the scan's grid and
  eroded by one voxel (``core``), so EPI distortion at the edge does not put
  non-brain voxels into the signal ROI.
* Noise is estimated, in order of preference, from (1) the difference of two
  repeated acquisitions (NEMA difference method), (2) the temporal residual of a
  time series, (3) air background with a magnitude-noise correction.
"""

from __future__ import annotations

from math import gamma, sqrt
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi


# ---- noise in magnitude images ---------------------------------------------------------
def magnitude_sd_factor(n_channels: int = 1) -> float:
    """SD of pure-noise sum-of-squares magnitude, in units of the per-channel sigma.

    The background of an n-channel SoS magnitude image follows a (non-central at
    zero signal = central) chi distribution with 2n degrees of freedom:
    SD/sigma = sqrt(2n - 2 (Gamma(n + 1/2) / Gamma(n))^2). n=1 gives the familiar
    Rayleigh 0.655.
    """
    n = max(int(n_channels), 1)
    return sqrt(2 * n - 2 * (gamma(n + 0.5) / gamma(n)) ** 2)


def background_mask(vol: np.ndarray, obj: np.ndarray, margin: int = 4,
                    exclude: Optional[np.ndarray] = None, border: int = 2) -> np.ndarray:
    """Air voxels: outside the dilated object, away from the image border, not
    in ``exclude`` (e.g. EPI ghost regions), and not exactly zero (masked recon)."""
    st = ndi.generate_binary_structure(obj.ndim, 1)
    far = ~ndi.binary_dilation(obj, structure=st, iterations=margin)
    inner = np.zeros_like(obj, dtype=bool)
    sl = tuple(slice(border, -border) if s > 2 * border + 1 and i < 2 else slice(None)
               for i, s in enumerate(obj.shape))
    inner[sl] = True
    bg = far & inner & (vol != 0)
    if exclude is not None:
        bg &= ~exclude
    return bg


def sigma_background(vol: np.ndarray, bg: np.ndarray, n_channels: int = 1) -> float:
    v = vol[bg]
    if v.size < 50:
        return float('nan')
    # trim the brightest 1% (stray ghosts, flow artefacts) before taking the SD
    v = v[v <= np.percentile(v, 99)]
    return float(np.std(v) / magnitude_sd_factor(n_channels))


def background_ratio(vol: np.ndarray, bg: np.ndarray) -> float:
    """mean/SD of the background. Pure magnitude noise gives 1.91 (one channel)
    to 2.75 (two-channel sum of squares); values near 1 mean the "background" is
    really low-signal tissue or artefact, and background SNR is not valid."""
    v = vol[bg]
    return float(v.mean() / v.std()) if v.size > 50 and v.std() > 0 else float('nan')


BG_RATIO_VALID = (1.6, 3.2)


def sigma_air_pockets(vol: np.ndarray, n_channels: int = 1, fraction: float = 0.01,
                      size: int = 5, two_d: bool = False) -> Dict[str, float]:
    """Noise from the darkest neighbourhoods of the image (airways, ear canals,
    gaps between head and bed): voxels whose local mean is in the lowest
    ``fraction``, eroded by one voxel. The fraction is halved (down to 0.1%) until
    those voxels look like pure magnitude noise (mean/SD inside ``BG_RATIO_VALID``),
    so a pocket smaller than ``fraction`` is not diluted by the tissue around it.
    Check ``ratio`` before using ``sigma``.
    """
    v = np.asarray(vol, np.float64)
    lm = ndi.uniform_filter(v, (size, size, 1) if two_d else size)
    out = {'sigma': float('nan'), 'ratio': float('nan'), 'n': 0}
    f = fraction
    while f >= 0.001:
        x = v[ndi.binary_erosion(lm <= np.percentile(lm, 100 * f))]
        if x.size < 100 or x.std() == 0:
            break
        out = {'sigma': float(x.std() / magnitude_sd_factor(n_channels)), 'ratio': float(x.mean() / x.std()),
               'n': int(x.size), 'fraction': f}
        if BG_RATIO_VALID[0] <= out['ratio'] <= BG_RATIO_VALID[1]:
            break
        f /= 2
    return out


def sigma_laplacian(vol: np.ndarray, mask: np.ndarray, two_d: bool = False) -> float:
    """Noise SD from the robust spread of a Laplacian-filtered image inside ``mask``.

    For white Gaussian noise, the 6- (3-D) or 4-neighbour (in-plane) Laplacian has
    SD = sigma * sqrt(sum K^2); the median absolute deviation (x1.4826) ignores
    most anatomical edges. Used when an image has no air background (tight FOV).
    Smooth structure leaks in, and k-space filtering/zero-filling correlates the
    noise, so treat it as approximate (~±20%).
    """
    k = np.zeros((3, 3, 3))
    k[1, 1, 1] = -4.0 if two_d else -6.0
    nb = [(0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1)] + ([] if two_d else [(1, 1, 0), (1, 1, 2)])
    for a in nb:
        k[a] = 1.0
    L = ndi.convolve(np.asarray(vol, np.float64), k, mode='nearest')
    st = np.zeros((3, 3, 3), bool)
    if two_d:
        st[:, :, 1] = ndi.generate_binary_structure(2, 1)
    else:
        st = ndi.generate_binary_structure(3, 1)
    m = ndi.binary_erosion(mask, structure=st)
    x = L[m]
    if x.size < 50:
        return float('nan')
    mad = np.median(np.abs(x - np.median(x)))
    return float(1.4826 * mad / np.sqrt((k ** 2).sum()))


def snr_background(vol: np.ndarray, brain: np.ndarray, bg: np.ndarray, n_channels: int = 1) -> Dict[str, float]:
    s = sigma_background(vol, bg, n_channels)
    mu = float(np.mean(vol[brain])) if brain.any() else float('nan')
    return {'signal': mu, 'sigma': s, 'snr': mu / s if s and np.isfinite(s) else float('nan'),
            'n_bg': int(bg.sum()), 'bg_ratio': background_ratio(vol, bg)}


def snr_consecutive(ts: np.ndarray, brain: np.ndarray) -> Dict[str, float]:
    """Single-image SNR of a time series by the difference method applied to every
    pair of consecutive volumes (sigma pooled as RMS)."""
    x = ts[brain]
    if x.shape[-1] < 2:
        return {}
    d = np.diff(x, axis=-1)
    sigma = float(np.sqrt(np.mean(np.var(d, axis=0))) / sqrt(2))
    mu = float(x.mean())
    return {'signal': mu, 'sigma': sigma, 'snr': mu / sigma if sigma > 0 else float('nan')}


def snr_difference(a: np.ndarray, b: np.ndarray, brain: np.ndarray) -> Dict[str, float]:
    """NEMA difference method: SNR of ONE image from two identical acquisitions.

    sigma = SD(a - b) / sqrt(2) over the ROI; signal = mean((a + b) / 2).
    """
    if not brain.any():
        return {'signal': float('nan'), 'sigma': float('nan'), 'snr': float('nan')}
    d = (a - b)[brain]
    sigma = float(np.std(d) / sqrt(2))
    mu = float(np.mean(((a + b) / 2)[brain]))
    return {'signal': mu, 'sigma': sigma, 'snr': mu / sigma if sigma > 0 else float('nan')}


# ---- time series ------------------------------------------------------------------------
def detrend(ts: np.ndarray, order: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Remove a polynomial trend along the last axis. Returns (residual, fit)."""
    t = np.linspace(-1, 1, ts.shape[-1])
    X = np.vander(t, order + 1)
    flat = ts.reshape(-1, ts.shape[-1]).T
    beta, *_ = np.linalg.lstsq(X, flat, rcond=None)
    fit = (X @ beta).T.reshape(ts.shape)
    return ts - fit, fit


def tsnr_map(ts: np.ndarray, order: int = 1) -> np.ndarray:
    """Voxelwise temporal SNR: mean / SD of the detrended residual (dof-corrected)."""
    res, _ = detrend(ts, order)
    dof = max(ts.shape[-1] - (order + 1), 1)
    sd = np.sqrt(np.sum(res ** 2, axis=-1) / dof)
    mu = ts.mean(axis=-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(sd > 0, mu / sd, 0.0)
    return out.astype(np.float32)


def drift_percent_per_min(series: np.ndarray, vol_tr_s: float) -> float:
    """Linear drift of a (brain-mean) time course, % of its mean per minute."""
    if series.size < 3:
        return float('nan')
    t = np.arange(series.size) * vol_tr_s / 60.0
    slope = np.polyfit(t, series, 1)[0]
    return float(100.0 * slope / np.mean(series))


def dvars(ts: np.ndarray, brain: np.ndarray) -> np.ndarray:
    """RMS over brain voxels of the frame-to-frame difference, % of mean signal."""
    x = ts[brain]
    d = np.diff(x, axis=-1)
    return 100.0 * np.sqrt(np.mean(d ** 2, axis=0)) / np.mean(x)


# ---- EPI ghosting -------------------------------------------------------------------------
def ghost_regions(obj: np.ndarray, pe_axis: int, n_shots: int = 1) -> np.ndarray:
    """Where Nyquist (N/2) and interleaved-segment (k*N/shots) ghosts of ``obj`` land,
    outside the object itself. Shifts wrap around, as aliased ghosts do."""
    n = obj.shape[pe_axis]
    shifts = {n // 2}
    for k in range(1, max(n_shots, 1)):
        shifts.add(int(round(k * n / n_shots)))
    shifts.update({(n - s) % n for s in list(shifts)})
    shifts.discard(0)
    g = np.zeros_like(obj, dtype=bool)
    for s in shifts:
        g |= np.roll(obj, s, axis=pe_axis)
    st = ndi.generate_binary_structure(obj.ndim, 1)
    near = ndi.binary_dilation(obj, structure=st, iterations=2)
    return g & ~near


def ghost_ratio(mean_img: np.ndarray, obj: np.ndarray, brain: np.ndarray, pe_axis: int,
                n_shots: int = 1, n_channels: int = 1) -> Dict[str, float]:
    """Ghost-to-signal ratio (%): (mean in ghost region - mean in clean background)
    / mean in brain. Background excludes the ghost region and the dilated object."""
    gr = ghost_regions(obj, pe_axis, n_shots)
    bg = background_mask(mean_img, obj, margin=4, exclude=ndi.binary_dilation(gr, iterations=2))
    gr &= mean_img != 0
    if gr.sum() < 20 or bg.sum() < 20 or not brain.any():
        return {'ghost_pct': float('nan')}
    g = float(np.mean(mean_img[gr]))
    b = float(np.mean(mean_img[bg]))
    s = float(np.mean(mean_img[brain]))
    return {'ghost_pct': 100.0 * (g - b) / s, 'ghost_mean': g, 'bg_mean': b}


# ---- contrast -------------------------------------------------------------------------------
def csf_contrast(vol: np.ndarray, brain: np.ndarray, sigma: float, csf_factor: float = 1.6) -> Dict[str, float]:
    """T2w CSF vs parenchyma inside the brain mask. CSF = voxels brighter than
    ``csf_factor`` x the brain median (ventricles); parenchyma = the rest of the
    eroded mask. CNR = (CSF - parenchyma) / sigma."""
    v = vol[brain]
    if v.size == 0:
        return {}
    med = np.median(v)
    csf = v > csf_factor * med
    par = ~csf
    if csf.sum() < 10:
        return {'csf_fraction': float(csf.mean())}
    c, p = float(v[csf].mean()), float(v[par].mean())
    return {'csf_mean': c, 'parenchyma_mean': p, 'contrast_ratio': c / p,
            'cnr': (c - p) / sigma if sigma else float('nan'), 'csf_fraction': float(csf.mean())}


# ---- efficiency ------------------------------------------------------------------------------
def snr_efficiency(snr: float, voxel_nl: Optional[float], time_s: Optional[float]) -> Dict[str, float]:
    """SNR per sqrt(minute) and per nL per sqrt(minute).

    SNR scales with voxel volume and with sqrt(acquisition time); dividing both
    out puts protocols with different resolution and duration on one scale.
    """
    out = {}
    if time_s:
        out['snr_per_sqrt_min'] = snr / sqrt(time_s / 60.0)
        if voxel_nl:
            out['snr_per_nl_sqrt_min'] = snr / voxel_nl / sqrt(time_s / 60.0)
    return out


def summarize(values: Iterable[float]) -> Dict[str, float]:
    v = np.asarray(list(values), dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {}
    q = np.percentile(v, [5, 25, 50, 75, 95])
    return {'p5': q[0], 'q1': q[1], 'median': q[2], 'q3': q[3], 'p95': q[4], 'mean': float(v.mean())}
