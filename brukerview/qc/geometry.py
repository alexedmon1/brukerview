"""World-coordinate resampling and masks shared by the QC metrics.

All scans of one study share the scanner's subject coordinate system, so a
mask drawn once on the anatomical reference can be carried into any other
scan through the ``visu_pars`` geometry (``BrukerImage.nifti_affine``). No
registration is involved: residual mismatch is real (EPI distortion, motion
between scans) and is part of what QC should show.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi

logger = logging.getLogger(__name__)


def resample(src: np.ndarray, src_aff: np.ndarray, tgt_aff: np.ndarray,
             tgt_shape: Sequence[int], order: int = 1, cval: float = 0.0) -> np.ndarray:
    """Sample ``src`` (3-D, affine ``src_aff``) on the voxel grid of a target."""
    ijk = np.indices(tuple(tgt_shape[:3]), dtype=np.float32).reshape(3, -1)
    xyz = tgt_aff[:3, :3] @ ijk + tgt_aff[:3, 3:4]
    inv = np.linalg.inv(src_aff)
    coords = inv[:3, :3] @ xyz + inv[:3, 3:4]
    out = ndi.map_coordinates(np.asarray(src, dtype=np.float32), coords, order=order,
                              cval=cval, mode='constant', prefilter=False)
    return out.reshape(tuple(tgt_shape[:3]))


def resample_mask(mask: np.ndarray, src_aff: np.ndarray, tgt_aff: np.ndarray,
                  tgt_shape: Sequence[int]) -> np.ndarray:
    """Partial-volume resampling of a binary mask, thresholded at 50%."""
    return resample(mask.astype(np.float32), src_aff, tgt_aff, tgt_shape, order=1) >= 0.5


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    v = values[np.isfinite(values)].ravel()
    hist, edges = np.histogram(v, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centers[np.argmax(between)])


def largest_component(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def fill_holes_2d(mask: np.ndarray) -> np.ndarray:
    """Fill holes slice by slice along each axis (closes ventricles, sulci)."""
    out = mask.copy()
    for ax in range(mask.ndim):
        out = np.moveaxis(np.stack([ndi.binary_fill_holes(s) for s in np.moveaxis(out, ax, 0)]), 0, ax)
    return out


def object_mask(vol: np.ndarray, smooth: float = 1.0, min_fraction: float = 0.01) -> np.ndarray:
    """Signal-bearing tissue (head/body) of one image: Otsu on a smoothed copy.

    Components smaller than ``min_fraction`` of the largest are dropped, holes
    filled. This is the "object" for ghost and background regions, not a brain mask.
    """
    s = ndi.gaussian_filter(np.asarray(vol, dtype=np.float32), smooth)
    m = s > otsu_threshold(s)
    lab, n = ndi.label(m)
    if n:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        keep = np.flatnonzero(sizes >= min_fraction * sizes.max()) + 1
        m = np.isin(lab, keep)
    return fill_holes_2d(m)


# ---- brain mask on the anatomical reference -----------------------------------------
def _bet2() -> Optional[str]:
    exe = shutil.which('bet2')
    if exe is None and os.environ.get('FSLDIR'):
        cand = Path(os.environ['FSLDIR']) / 'bin' / 'bet2'
        exe = str(cand) if cand.exists() else None
    return exe


def bet_hull(vol: np.ndarray, affine: np.ndarray, frac: float = 0.45,
             scale: float = 10.0) -> Optional[np.ndarray]:
    """FSL ``bet2`` on a 10x voxel-scaled copy (rodent brains are too small for BET's
    human defaults). Two passes: the second is seeded at the centre of the first,
    since the image centre of gravity of a rodent head scan is often outside the brain.
    Returns None if FSL is not available or BET fails.
    """
    exe = _bet2()
    if exe is None:
        return None
    import nibabel as nib
    aff = affine.copy()
    aff[:3, :] *= scale
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / 'ref.nii.gz'
        nib.save(nib.Nifti1Image(np.asarray(vol, np.float32), aff), str(src))
        env = dict(os.environ, FSLOUTPUTTYPE='NIFTI_GZ')
        seed: Optional[Tuple[int, ...]] = None
        mask = None
        for _ in range(2):
            out = Path(td) / 'bet'
            cmd = [exe, str(src), str(out), '-f', f'{frac}', '-m', '-n']
            if seed is not None:
                cmd += ['-c', *[str(int(c)) for c in seed]]
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            mfile = Path(td) / 'bet_mask.nii.gz'
            if r.returncode != 0 or not mfile.exists():
                logger.warning('bet2 failed: %s', r.stderr.strip())
                return mask
            mask = np.asarray(nib.load(str(mfile)).dataobj) > 0
            seed = tuple(np.round(ndi.center_of_mass(mask)).astype(int))
    return mask


def brain_mask_t2(vol: np.ndarray, affine: np.ndarray, use_bet: bool = True,
                  band: Tuple[float, float] = (0.85, 1.5), open_iter: int = 3) -> Tuple[np.ndarray, str]:
    """Brain mask for a T2-weighted (RARE) rodent head image.

    1. hull: FSL bet2 (if available), else the image's object mask;
    2. keep voxels whose smoothed intensity lies within ``band`` x the median of
       the hull's eroded core (brain parenchyma; T2 muscle is darker, fat and CSF
       brighter);
    3. morphological opening to cut thin bridges to extracranial tissue, largest
       component, closing, slice-wise hole filling (restores ventricles), and
       intersection with the hull.
    Returns the mask and a one-line description of how it was made.
    """
    vol = np.asarray(vol, dtype=np.float32)
    hull = bet_hull(vol, affine) if use_bet else None
    how = 'FSL bet2 hull (f=0.45, 10x voxel scaling, seeded 2nd pass)'
    if hull is None or hull.sum() == 0:
        hull = object_mask(vol)
        how = 'Otsu object hull'
    s = ndi.gaussian_filter(vol, 1.0)
    core = ndi.binary_erosion(hull, iterations=8)
    if core.sum() < 100:
        core = hull
    med = float(np.median(s[core]))
    m = (s > band[0] * med) & (s < band[1] * med) & hull
    st = ndi.generate_binary_structure(3, 1)
    m = ndi.binary_opening(m, structure=st, iterations=open_iter)
    m = largest_component(m)
    m = ndi.binary_closing(m, structure=st, iterations=open_iter)
    m = fill_holes_2d(ndi.binary_fill_holes(m)) & hull
    how += f' + intensity band {band[0]}-{band[1]}x core median, opening/closing r={open_iter}, hole fill'
    return m, how


# ---- slab coverage --------------------------------------------------------------------
def slab_coverage(brain: np.ndarray, brain_aff: np.ndarray, tgt_aff: np.ndarray,
                  tgt_shape: Sequence[int], slice_thick: Optional[float]) -> Tuple[float, float]:
    """Fraction of brain voxels inside a 2-D multi-slice acquisition.

    Returns ``(in_slab, sampled)``: inside the slab's outer extent (gaps
    included), and actually inside a slice's nominal thickness (gaps excluded).
    """
    idx = np.argwhere(brain).T.astype(np.float64)
    if idx.size == 0:
        return float('nan'), float('nan')
    xyz = brain_aff[:3, :3] @ idx + brain_aff[:3, 3:4]
    inv = np.linalg.inv(tgt_aff)
    ijk = inv[:3, :3] @ xyz + inv[:3, 3:4]
    nx, ny, nz = tgt_shape[:3]
    inplane = (ijk[0] > -0.5) & (ijk[0] < nx - 0.5) & (ijk[1] > -0.5) & (ijk[1] < ny - 0.5)
    k = ijk[2]
    in_slab = inplane & (k > -0.5) & (k < nz - 0.5)
    dz = float(np.linalg.norm(tgt_aff[:3, 2]))
    if slice_thick is None or slice_thick >= dz:
        sampled = in_slab
    else:
        sampled = in_slab & (np.abs(k - np.round(k)) * dz <= slice_thick / 2.0)
    n = idx.shape[1]
    return float(in_slab.sum() / n), float(sampled.sum() / n)


def brain_center_world(brain: np.ndarray, aff: np.ndarray) -> np.ndarray:
    c = np.array(ndi.center_of_mass(brain))
    return aff[:3, :3] @ c + aff[:3, 3]


def axis_aligned_slices(vol: np.ndarray, aff: np.ndarray, center: np.ndarray,
                        half_extent: np.ndarray, res: float, order: int = 1):
    """Three orthogonal planes through ``center`` on a world-axis-aligned grid.

    Gives every scan the same display orientation regardless of how it was
    acquired. Returns a list of 2-D arrays, one per world axis held fixed
    (x, y, z), each indexed ``[row, col]`` ready for ``imshow``.
    """
    planes = []
    n = np.maximum(np.round(2 * half_extent / res).astype(int), 2)
    for fixed in range(3):
        free = [a for a in range(3) if a != fixed]
        u = np.linspace(-half_extent[free[0]], half_extent[free[0]], n[free[0]])
        v = np.linspace(-half_extent[free[1]], half_extent[free[1]], n[free[1]])
        U, V = np.meshgrid(u, v, indexing='ij')
        xyz = np.zeros((3, U.size))
        xyz[fixed] = center[fixed]
        xyz[free[0]] = center[free[0]] + U.ravel()
        xyz[free[1]] = center[free[1]] + V.ravel()
        inv = np.linalg.inv(aff)
        ijk = inv[:3, :3] @ xyz + inv[:3, 3:4]
        img = ndi.map_coordinates(np.asarray(vol, np.float32), ijk, order=order, cval=np.nan,
                                  mode='constant', prefilter=False).reshape(U.shape)
        planes.append(img.T[::-1])  # rows = second free axis, top = +
    return planes
