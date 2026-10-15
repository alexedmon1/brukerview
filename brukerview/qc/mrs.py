"""Single-voxel spectroscopy QC: unsuppressed water linewidth as a shim metric."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np


def load_spectrum(scan_dir: Path, sw_hz: float) -> Optional[Dict[str, np.ndarray]]:
    """Frequency axis (Hz, water-centred) and real (absorption) spectrum.

    Uses ``pdata/1/fid_proc.64`` (complex128). That FID still carries the
    digital-filter group delay (tens of near-zero points before the echo top),
    which would wrap the spectrum into a sinc; the FID is therefore started at its
    magnitude maximum and zero-filled back to length. FFT, then the zero-order phase
    that maximises the real peak.
    """
    p = Path(scan_dir) / 'pdata' / '1' / 'fid_proc.64'
    if not p.exists():
        return None
    fid = np.fromfile(p, dtype='<c16')
    if fid.size < 16:
        return None
    n = fid.size
    fid = np.concatenate([fid[int(np.argmax(np.abs(fid))):], np.zeros(0, complex)])
    fid = np.pad(fid, (0, n - fid.size))
    spec = np.fft.fftshift(np.fft.fft(fid))
    freq = np.fft.fftshift(np.fft.fftfreq(fid.size, d=1.0 / sw_hz))
    k = int(np.argmax(np.abs(spec)))
    spec = spec * np.exp(-1j * np.angle(spec[k]))
    freq = freq - freq[k]
    return {'freq_hz': freq, 'real': spec.real, 'mag': np.abs(spec)}


def fwhm(freq: np.ndarray, y: np.ndarray) -> float:
    """Full width at half maximum of the tallest peak, linear interpolation."""
    k = int(np.argmax(y))
    half = y[k] / 2.0
    i = k
    while i > 0 and y[i] > half:
        i -= 1
    j = k
    while j < y.size - 1 and y[j] > half:
        j += 1
    if i == 0 or j == y.size - 1:
        return float('nan')
    left = freq[i] + (half - y[i]) * (freq[i + 1] - freq[i]) / (y[i + 1] - y[i])
    right = freq[j - 1] + (half - y[j - 1]) * (freq[j] - freq[j - 1]) / (y[j] - y[j - 1])
    return float(abs(right - left))


def water_linewidth(scan_dir: Path, sw_hz: float, freq_mhz: Optional[float]) -> Dict:
    sp = load_spectrum(scan_dir, sw_hz)
    if sp is None:
        return {}
    out = {'fwhm_hz': fwhm(sp['freq_hz'], sp['real']), 'fwhm_mag_hz': fwhm(sp['freq_hz'], sp['mag']),
           '_spectrum': sp}
    if freq_mhz:
        out['fwhm_ppm'] = out['fwhm_hz'] / freq_mhz
    return out
