"""NIfTI support: view ``.nii``/``.nii.gz`` files, and export Bruker images.

Both directions need nibabel, which is imported lazily so the rest of
brukerview keeps working without it.

:class:`NiftiImage` presents a NIfTI file through the same interface as
:class:`~brukerview.reader.BrukerImage` (``load``, ``spacing``, ``shape``,
``extra_groups``, ``summary``, ``nifti_affine``), so the viewer, the montage,
``list`` and ``info`` treat both formats the same way.
"""

from __future__ import annotations

import re
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .reader import BrukerImage, FrameGroup

#: File names recognised as NIfTI (``.img`` is left out: the ``.hdr`` beside it
#: is the file nibabel wants, and listing both would show every volume twice).
NIFTI_SUFFIXES = ('.nii', '.nii.gz', '.hdr')

# Frame groups whose axis is time, so its NIfTI zoom is a repetition time.
_TIME_GROUPS = ('FG_MOVIE', 'FG_CYCLE', 'FG_VOLUME')

# Array axis order used for display: first axis to the patient's left, second
# posterior, third superior. This is how Bruker stores an axial slice package,
# so reordering NIfTI to match keeps the viewer's on-screen orientation,
# left/right handedness and montage layout identical for the two formats.
DISPLAY_AXCODES = ('L', 'P', 'S')

_TIME_UNIT_MS = {'sec': 1000.0, 'msec': 1.0, 'usec': 0.001}


def _nibabel():
    try:
        import nibabel
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit('NIfTI support needs nibabel: pip install nibabel') from exc
    return nibabel


def is_nifti_path(path: Union[str, Path]) -> bool:
    return Path(path).name.lower().endswith(NIFTI_SUFFIXES)


# ---- reading -----------------------------------------------------------------------
class NiftiImage:
    """A NIfTI file, presented like a :class:`BrukerImage`.

    Parameters
    ----------
    path
        ``.nii``, ``.nii.gz`` or ``.hdr`` file.
    reorient
        Reorder the array to :data:`DISPLAY_AXCODES` (the default) so slices are
        laid out as they are for a Bruker scan. ``False`` keeps the file's own
        axis order, which is what you want when checking how data is stored.
    """

    method_params: Dict[str, Any] = {}

    def __init__(self, path: Union[str, Path], reorient: bool = True):
        nib = _nibabel()
        self.path = Path(path).expanduser()
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.reorient = reorient
        self._nii = nib.load(str(self.path))
        self.header = self._nii.header
        # Same role as BrukerImage.visu: the parameter dict `info --param` reads.
        self.visu: Dict[str, Any] = dict(self.header.items())

    # ---- identification -------------------------------------------------
    @property
    def scan_name(self) -> str:
        name = self.path.name
        for suffix in ('.nii.gz', '.nii', '.hdr'):
            if name.lower().endswith(suffix):
                return name[:-len(suffix)]
        return name

    #: No pdata numbering for a plain file; kept so listings line up.
    proc_no = '-'

    # A NIfTI file plays the part of both the pdata directory (paths in
    # messages) and the 2dseq (mtime of the pixel data).
    @property
    def pdata_dir(self) -> Path:
        return self.path

    @property
    def data_path(self) -> Path:
        return self.path

    @property
    def scan_dir(self) -> Path:
        return self.path.parent

    @property
    def protocol(self) -> str:
        descrip = self.header.get('descrip')
        text = descrip.tobytes().decode('latin-1') if isinstance(descrip, np.ndarray) else str(descrip or '')
        return text.split('\x00')[0].strip()

    @property
    def method(self) -> str:
        return 'nifti'

    @property
    def subject_id(self) -> str:
        return ''

    @property
    def study_id(self) -> str:
        return ''

    @property
    def title(self) -> str:
        return f'{self.path.name}  {self.protocol}'.rstrip()

    # ---- geometry ----------------------------------------------------------
    @cached_property
    def _ornt(self) -> Optional[np.ndarray]:
        """nibabel orientation transform to :data:`DISPLAY_AXCODES`, or None."""
        if not self.reorient:
            return None
        from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
        xfm = ornt_transform(io_orientation(self._nii.affine), axcodes2ornt(DISPLAY_AXCODES))
        identity = np.column_stack((np.arange(3), np.ones(3)))
        return None if np.array_equal(xfm, identity) else xfm

    @property
    def _stored_size(self) -> Tuple[int, int, int]:
        """Spatial shape as stored, padded to three axes."""
        shape = tuple(int(s) for s in self._nii.shape[:3])
        return shape + (1,) * (3 - len(shape))  # type: ignore[return-value]

    @cached_property
    def size(self) -> Tuple[int, ...]:
        """Spatial shape of :meth:`load` output, ``(nx, ny, nz)``."""
        stored = self._stored_size
        if self._ornt is None:
            return stored
        return tuple(stored[int(ax)] for ax in np.argsort(self._ornt[:, 0]))

    @cached_property
    def _affine(self) -> np.ndarray:
        """Index-to-world (RAS) affine of the array :meth:`load` returns."""
        aff = np.asarray(self._nii.affine, dtype=float)
        if self._ornt is None:
            return aff
        from nibabel.orientations import inv_ornt_aff
        return aff @ inv_ornt_aff(self._ornt, self._stored_size)

    @property
    def dim(self) -> int:
        """Spatial dimensionality, like ``VisuCoreDim`` (2 for a single plane)."""
        return min(3, len(self._nii.shape))

    @property
    def is_image(self) -> bool:
        return len(self._nii.shape) >= 2

    @property
    def spacing(self) -> Tuple[float, float, float]:
        """Voxel size in mm as ``(dx, dy, dz)``, from the affine."""
        zooms = np.sqrt((self._affine[:3, :3] ** 2).sum(axis=0))
        dx, dy, dz = (float(z) if z > 0 else 1.0 for z in zooms)
        return (dx, dy, dz)

    @property
    def extent(self) -> Tuple[float, ...]:
        return tuple(n * d for n, d in zip(self.size, self.spacing))

    @property
    def units(self) -> str:
        return str(self.header.get_xyzt_units()[0] or 'mm')

    @property
    def n_slices(self) -> int:
        return self.size[2]

    @property
    def slice_distance(self) -> float:
        return self.spacing[2]

    @property
    def orientation_label(self) -> str:
        """Axis codes of the array :meth:`load` returns, e.g. ``LPS``."""
        from nibabel.orientations import aff2axcodes
        return ''.join(c or '?' for c in aff2axcodes(self._affine))

    # ---- non-spatial axes -------------------------------------------------
    @cached_property
    def extra_groups(self) -> List[FrameGroup]:
        """The dimensions past the third, as frame groups (4th = volumes)."""
        names = ['FG_VOLUME', 'FG_DIM5', 'FG_DIM6', 'FG_DIM7']
        return [FrameGroup(int(n), names[i] if i < len(names) else f'FG_DIM{i + 4}', '')
                for i, n in enumerate(self._nii.shape[3:])]

    @property
    def frame_groups(self) -> List[FrameGroup]:
        # Slices are a spatial axis in NIfTI, so there is no FG_SLICE to add.
        return self.extra_groups

    @property
    def shape(self) -> Tuple[int, ...]:
        return (*self.size, *(g.size for g in self.extra_groups))

    @property
    def frame_count(self) -> int:
        return int(np.prod([g.size for g in self.extra_groups])) if self.extra_groups else 1

    @property
    def dtype(self) -> np.dtype:
        return self.header.get_data_dtype()

    @property
    def echo_times(self) -> List[float]:
        return []

    @property
    def repetition_time(self) -> Optional[float]:
        """The 4th dimension's zoom in ms, if the file has one."""
        zooms = self.header.get_zooms()
        if len(zooms) < 4 or not zooms[3]:
            return None
        unit = self.header.get_xyzt_units()[1]
        if unit not in _TIME_UNIT_MS:
            # pixdim[4] is left at 1 with no time unit when it means nothing;
            # anything else is almost always seconds, which is NIfTI's default.
            return None if float(zooms[3]) == 1.0 else float(zooms[3]) * 1000.0
        return float(zooms[3]) * _TIME_UNIT_MS[unit]

    @property
    def scan_time_s(self) -> Optional[float]:
        return None

    def group_labels(self, group: FrameGroup) -> List[str]:
        n = group.size
        tr = self.repetition_time
        if group.name == 'FG_VOLUME' and tr:
            return [f't = {i * tr / 1000.0:.2f} s' for i in range(n)]
        return [f'{group.label} {i + 1}/{n}' for i in range(n)]

    # ---- data --------------------------------------------------------------
    def load(self, scaled: bool = True) -> np.ndarray:
        """Image array shaped ``(nx, ny, nz, *extra_groups)``.

        With ``scaled=True`` the header's ``scl_slope``/``scl_inter`` are
        applied; otherwise the values as stored on disk are returned.
        """
        if not self.is_image:
            raise ValueError(f'{self.path}: not an image (shape {self._nii.shape})')
        proxy = self._nii.dataobj
        if scaled:
            data = np.asanyarray(proxy)
            if np.dtype(self.dtype).itemsize <= 4 and data.dtype != np.float32:
                # nibabel applies the scaling in float64; keep the float32 that
                # BrukerImage.load returns unless the file itself is wider.
                data = data.astype(np.float32)
        elif hasattr(proxy, 'get_unscaled'):
            data = proxy.get_unscaled()
        else:  # an in-memory image built by hand, never scaled
            data = np.asanyarray(proxy)
        if data.ndim < 3:
            data = data.reshape(data.shape + (1,) * (3 - data.ndim))
        if self._ornt is not None:
            from nibabel.orientations import apply_orientation
            data = apply_orientation(data, self._ornt)
        return data

    def load_raw(self) -> np.ndarray:
        return self.load(scaled=False)

    # ---- world geometry ------------------------------------------------------
    def nifti_affine(self) -> np.ndarray:
        """Index-to-world affine in NIfTI RAS convention."""
        return self._affine.copy()

    def affine(self) -> np.ndarray:
        """Index-to-world affine in Bruker/DICOM LPS convention."""
        return np.diag([-1.0, -1.0, 1.0, 1.0]) @ self._affine

    # ---- summary ---------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        dx, dy, dz = self.spacing
        tr = self.repetition_time
        return {
            'scan': self.scan_name,
            'proc': self.proc_no,
            'protocol': self.protocol,
            'method': self.method,
            'dim': f'{self.dim}D',
            'matrix': 'x'.join(str(s) for s in self.size),
            'slices': self.n_slices,
            'extra': ', '.join(f'{g.label} {g.size}' for g in self.extra_groups),
            'voxel_mm': f'{dx:.3g}x{dy:.3g}x{dz:.3g}',
            'TE_ms': '',
            'TR_ms': f'{tr:g}' if tr else '',
            'dtype': np.dtype(self.dtype).name,
            'frames': self.frame_count,
            'path': str(self.path),
        }

    def __repr__(self) -> str:
        return f'NiftiImage({self.path}, shape={self.shape})'


def find_niftis(path: Union[str, Path], max_depth: int = 2,
                reorient: bool = True) -> List[NiftiImage]:
    """NIfTI files at ``path`` (a file, or a directory searched shallowly)."""
    path = Path(path).expanduser()
    if path.is_file():
        return [NiftiImage(path, reorient=reorient)] if is_nifti_path(path) else []
    found: List[NiftiImage] = []

    def walk(d: Path, depth: int):
        try:
            entries = sorted(d.iterdir(), key=lambda c: c.name.lower())
        except OSError:
            return
        for child in entries:
            if child.is_file() and is_nifti_path(child):
                try:
                    found.append(NiftiImage(child, reorient=reorient))
                except Exception as exc:  # unreadable header: skip, don't abort listing
                    print(f'warning: skipping {child}: {exc}')
            elif child.is_dir() and depth > 0 and not child.name.startswith('.') and child.name != 'pdata':
                walk(child, depth - 1)

    if path.is_dir():
        walk(path, max_depth)
    return found


# ---- export ------------------------------------------------------------------------
def default_name(img: Union[BrukerImage, NiftiImage]) -> str:
    if isinstance(img, NiftiImage):
        return img.path.name
    proto = re.sub(r'[^A-Za-z0-9_.-]+', '_', img.protocol) or 'scan'
    return f'{img.scan_name}_{proto}_p{img.proc_no}.nii.gz'


def to_nifti(img: Union[BrukerImage, NiftiImage], out: Optional[Union[str, Path]] = None,
             scaled: bool = True):
    nib = _nibabel()

    data = img.load(scaled=scaled)
    # nibabel wants at most one non-spatial axis for a plain 4D file; fold
    # extra frame groups into a single 4th dimension (first group fastest).
    if data.ndim > 4:
        data = data.reshape(data.shape[:3] + (-1,), order='F')
    data = np.ascontiguousarray(data)
    nii = nib.Nifti1Image(data, img.nifti_affine())
    dx, dy, dz = img.spacing
    zooms = [dx, dy, dz]
    tr = img.repetition_time
    is_time = data.ndim == 4 and tr and any(g.name in _TIME_GROUPS for g in img.extra_groups)
    if data.ndim == 4:
        zooms.append(tr / 1000.0 if is_time else 1.0)
    nii.header.set_zooms(zooms)
    # Leaving the time unit unset marks a 4th axis that is not time (echoes,
    # diffusion directions), so pixdim[4]=1 is not read back as a 1 s TR.
    nii.header.set_xyzt_units('mm', 'sec' if is_time else None)
    nii.header['descrip'] = f'{img.protocol} {img.method}'[:79].encode()
    out = Path(out) if out else Path(default_name(img))
    if out.is_dir():
        out = out / default_name(img)
    if out.exists() and out.resolve() == Path(img.data_path).resolve():
        raise ValueError(f'{out}: would overwrite the input file; pass -o')
    nib.save(nii, str(out))
    return out
