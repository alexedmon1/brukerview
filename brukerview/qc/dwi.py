"""Diffusion QC: shell bookkeeping, per-shell SNR and a quick tensor fit."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .metrics import snr_difference


def bmatrices(method: Dict) -> Optional[np.ndarray]:
    """Effective b-matrices (s/mm^2, n x 3 x 3) including imaging gradients."""
    b = method.get('PVM_DwBMat')
    if b is None:
        return None
    arr = np.asarray(b, dtype=float).reshape(-1, 3, 3)
    return arr


def shell_labels(method: Dict, b0_threshold: float = 100.0) -> np.ndarray:
    """Nominal shell (0 or one of PVM_DwBvalEach) for each diffusion volume."""
    eff = np.asarray(method.get('PVM_DwEffBval', []), dtype=float).ravel()
    nominal = np.array([0.0] + [float(x) for x in np.atleast_1d(method.get('PVM_DwBvalEach', []))])
    lab = nominal[np.argmin(np.abs(eff[:, None] - nominal[None, :]), axis=1)]
    lab[eff < b0_threshold] = 0.0
    return lab


def arrange(data: np.ndarray, extra_names: List[str], n_diff: int) -> np.ndarray:
    """Put a DWI load() array into ``(x, y, z, n_diff, n_rep)``.

    The diffusion axis is found by its frame-group name (any order on disk), or
    failing that by its length; every other group is folded into repetitions.
    Raises if no axis has one volume per b-value.
    """
    if data.ndim == 3:
        data = data[..., None]
        extra_names = ['diffusion']
    names = list(extra_names)
    if 'diffusion' in names:
        d = names.index('diffusion')
    else:
        same = [i for i in range(len(names)) if data.shape[3 + i] == n_diff]
        if not same:
            raise ValueError(f'no frame group with {n_diff} diffusion volumes in {data.shape}')
        d = same[0]
    arr = np.moveaxis(data, 3 + d, 3)
    if arr.shape[3] != n_diff:
        raise ValueError(f'diffusion axis has {arr.shape[3]} volumes but there are {n_diff} b-values')
    return arr.reshape(arr.shape[:4] + (-1,))


def shell_snr(data: np.ndarray, shells: np.ndarray, brain: np.ndarray) -> List[Dict[str, float]]:
    """Per-shell single-image SNR from two repetitions (difference method).

    ``data``: (x, y, z, n_vol, n_rep>=2). For each volume the difference of the first
    two repetitions gives sigma; signal and sigma are averaged over the shell's
    volumes (sigma as RMS), so SNR refers to one diffusion-weighted image.
    """
    out = []
    for b in np.unique(shells):
        idx = np.flatnonzero(shells == b)
        sig, var, per_dir = [], [], []
        for i in idx:
            r = snr_difference(data[..., i, 0], data[..., i, 1], brain)
            sig.append(r['signal'])
            var.append(r['sigma'] ** 2)
            per_dir.append(r['snr'])
        sigma = float(np.sqrt(np.mean(var)))
        mu = float(np.mean(sig))
        out.append({'b': float(b), 'n_vol': int(idx.size), 'signal': mu, 'sigma': sigma,
                    'snr': mu / sigma if sigma > 0 else float('nan'),
                    'snr_dir_sd': float(np.std(per_dir)),
                    'snr_dir_min': float(np.min(per_dir)), 'snr_dir_max': float(np.max(per_dir))})
    return out


def fit_tensor(signal: np.ndarray, bmat: np.ndarray, mask: np.ndarray,
               max_b: float = 1100.0, beff: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Weighted log-linear tensor fit; returns FA and MD (mm^2/s) maps.

    Only volumes with b <= ``max_b`` are used (DTI is a low-b model; b=2000 data
    are dominated by non-Gaussian diffusion). Weights are the squared OLS-predicted
    signal (Salvador et al. 2005).
    """
    trace = np.trace(bmat, axis1=1, axis2=2) if beff is None else beff
    use = trace <= max_b
    B = bmat[use]
    X = np.column_stack([-B[:, 0, 0], -B[:, 1, 1], -B[:, 2, 2],
                         -2 * B[:, 0, 1], -2 * B[:, 0, 2], -2 * B[:, 1, 2], np.ones(use.sum())])
    S = np.clip(signal[mask][:, use].astype(np.float64), 1e-3, None)   # (n_vox, n_vol)
    y = np.log(S)
    beta_ols, *_ = np.linalg.lstsq(X, y.T, rcond=None)                    # (7, n_vox)
    # weights = predicted signal squared, capped at twice each voxel's largest
    # measured signal so an extrapolating OLS fit cannot dominate the system
    pred = np.minimum(np.exp(X @ beta_ols).T, 2.0 * S.max(axis=1, keepdims=True))
    w = pred ** 2                                                          # (n_vox, n_vol)
    XtWX = np.einsum('vn,ni,nj->vij', w, X, X)
    XtWy = np.einsum('vn,ni,vn->vi', w, X, y)
    beta = np.linalg.solve(XtWX + 1e-12 * np.eye(7), XtWy[..., None])[..., 0].T
    D = np.zeros((y.shape[0], 3, 3))
    D[:, 0, 0], D[:, 1, 1], D[:, 2, 2] = beta[0], beta[1], beta[2]
    D[:, 0, 1] = D[:, 1, 0] = beta[3]
    D[:, 0, 2] = D[:, 2, 0] = beta[4]
    D[:, 1, 2] = D[:, 2, 1] = beta[5]
    ev = np.clip(np.linalg.eigvalsh(D), 0, None)
    md = ev.mean(axis=1)
    num = np.sqrt(((ev - md[:, None]) ** 2).sum(axis=1))
    den = np.sqrt((ev ** 2).sum(axis=1))
    fa = np.where(den > 0, np.sqrt(1.5) * num / den, 0.0)
    FA = np.zeros(mask.shape, np.float32)
    MD = np.zeros(mask.shape, np.float32)
    FA[mask], MD[mask] = np.clip(fa, 0, 1), md
    return FA, MD
