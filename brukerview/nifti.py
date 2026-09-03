"""NIfTI export (requires nibabel)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .reader import BrukerImage


def default_name(img: BrukerImage) -> str:
    proto = re.sub(r'[^A-Za-z0-9_.-]+', '_', img.protocol) or 'scan'
    return f'{img.scan_name}_{proto}_p{img.proc_no}.nii.gz'


def to_nifti(img: BrukerImage, out: Optional[Union[str, Path]] = None,
             scaled: bool = True):
    import nibabel as nib

    data = img.load(scaled=scaled)
    # nibabel wants at most one non-spatial axis for a plain 4D file; fold
    # extra frame groups into a single 4th dimension (first group fastest).
    if data.ndim > 4:
        data = data.reshape(data.shape[:3] + (-1,), order='F')
    data = np.ascontiguousarray(data)
    nii = nib.Nifti1Image(data, img.nifti_affine())
    dx, dy, dz = img.spacing
    zooms = [dx, dy, dz]
    if data.ndim == 4:
        tr = img.repetition_time
        zooms.append(tr / 1000.0 if tr and any(g.name in ('FG_MOVIE', 'FG_CYCLE') for g in img.extra_groups) else 1.0)
    nii.header.set_zooms(zooms)
    nii.header.set_xyzt_units('mm', 'sec')
    nii.header['descrip'] = f'{img.protocol} {img.method}'[:79].encode()
    out = Path(out) if out else Path(default_name(img))
    if out.is_dir():
        out = out / default_name(img)
    nib.save(nii, str(out))
    return out
