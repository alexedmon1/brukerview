"""B0 field maps: shim quality in the brain and predicted EPI displacement."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .geometry import resample, resample_mask
from .metrics import summarize


def b0_stats(hz: np.ndarray, brain: np.ndarray, freq_mhz: Optional[float] = None) -> Dict[str, float]:
    """Spread of the off-resonance (Hz) within the brain, after removing the mean
    (a global frequency offset is corrected by the scanner; only the spread
    distorts EPI)."""
    v = hz[brain]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {}
    v = v - np.median(v)
    out = {'b0_sd_hz': float(np.std(v)), 'b0_iqr_hz': float(np.subtract(*np.percentile(v, [75, 25]))),
           'b0_p5_p95_hz': float(np.subtract(*np.percentile(v, [95, 5]))),
           'b0_absmax_p99_hz': float(np.percentile(np.abs(v), 99))}
    if freq_mhz:
        out['b0_sd_ppm'] = out['b0_sd_hz'] / freq_mhz
    return out


def displacement_mm(hz_in_epi: np.ndarray, pe_fov_mm: float, eff_bw_hz: float) -> np.ndarray:
    """EPI shift along phase encoding: d = df * FOV_PE / BW_eff.

    ``PVM_EpiEffBandwidth`` is 1 / effective echo spacing (shots and parallel
    imaging included), so df / BW_eff is the shift as a fraction of the PE FOV.
    """
    return hz_in_epi * pe_fov_mm / eff_bw_hz


def predicted_distortion(hz: np.ndarray, hz_aff: np.ndarray, hz_support: np.ndarray,
                         epi_aff: np.ndarray, epi_shape, brain_epi: np.ndarray,
                         pe_fov_mm: float, eff_bw_hz: float, pe_pixel_mm: float) -> Dict[str, float]:
    """Displacement statistics (absolute, mm and pixels) inside the brain of one EPI."""
    f = resample(hz, hz_aff, epi_aff, epi_shape, order=1)
    sup = resample_mask(hz_support, hz_aff, epi_aff, epi_shape)
    m = brain_epi & sup
    if not m.any():
        return {}
    f = f - np.median(f[m])
    d = np.abs(displacement_mm(f[m], pe_fov_mm, eff_bw_hz))
    s = summarize(d)
    return {'disp_median_mm': s['median'], 'disp_p95_mm': s['p95'],
            'disp_p95_px': s['p95'] / pe_pixel_mm, 'mm_per_100hz': 100.0 * pe_fov_mm / eff_bw_hz,
            '_map': np.where(m, displacement_mm(f, pe_fov_mm, eff_bw_hz), np.nan)}
