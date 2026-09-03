"""Read Bruker ParaVision reconstructed images (``pdata/N/2dseq``) into numpy.

Everything needed to interpret ``2dseq`` lives in ``visu_pars`` next to it:
matrix size, data type, byte order, frame count, frame-group structure,
intensity scaling, field of view and geometry. No ``method``/``acqp`` parsing
is required to display an image, although ``method`` is read (lazily) for
the sequence name shown in listings.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .jcampdx import read_jcampdx

logger = logging.getLogger(__name__)

WORD_TYPES = {
    '_8BIT_UNSGN_INT': np.uint8,
    '_8BIT_SGN_INT': np.int8,
    '_16BIT_SGN_INT': np.int16,
    '_16BIT_UNSGN_INT': np.uint16,
    '_32BIT_SGN_INT': np.int32,
    '_32BIT_UNSGN_INT': np.uint32,
    '_32BIT_FLOAT': np.float32,
    '_64BIT_FLOAT': np.float64,
}


@dataclass(frozen=True)
class FrameGroup:
    """One entry of ``VisuFGOrderDesc``: a non-spatial dimension of the data."""
    size: int
    name: str        # e.g. FG_SLICE, FG_ECHO, FG_DIFFUSION, FG_MOVIE, FG_CYCLE
    comment: str

    @property
    def label(self) -> str:
        return self.name.removeprefix('FG_').lower()


class BrukerImage:
    """A reconstructed Bruker image: one ``pdata/<proc>`` directory.

    Parameters
    ----------
    pdata_dir
        Directory containing ``2dseq`` and ``visu_pars``.
    """

    def __init__(self, pdata_dir: Union[str, Path]):
        self.pdata_dir = Path(pdata_dir)
        self.visu_path = self.pdata_dir / 'visu_pars'
        self.data_path = self.pdata_dir / '2dseq'
        if not self.visu_path.exists():
            raise FileNotFoundError(f'{self.pdata_dir}: no visu_pars')
        self.visu: Dict[str, Any] = read_jcampdx(self.visu_path)

    # ---- identification -------------------------------------------------
    @property
    def scan_dir(self) -> Path:
        return self.pdata_dir.parent.parent

    @property
    def proc_no(self) -> str:
        return self.pdata_dir.name

    @property
    def scan_name(self) -> str:
        return self.scan_dir.name

    @cached_property
    def method_params(self) -> Dict[str, Any]:
        path = self.scan_dir / 'method'
        return read_jcampdx(path) if path.exists() else {}

    @property
    def method(self) -> str:
        return str(self.method_params.get('Method', '')) or str(self.visu.get('VisuAcqSequenceName', ''))

    @property
    def protocol(self) -> str:
        return str(self.visu.get('VisuAcquisitionProtocol', ''))

    @property
    def subject_id(self) -> str:
        return str(self.visu.get('VisuSubjectId', ''))

    @property
    def study_id(self) -> str:
        return str(self.visu.get('VisuStudyId', ''))

    @property
    def title(self) -> str:
        return f'{self.scan_name}/pdata/{self.proc_no}  {self.protocol}'

    # ---- geometry ----------------------------------------------------------
    @property
    def dim(self) -> int:
        return int(self.visu.get('VisuCoreDim', 0))

    @property
    def is_image(self) -> bool:
        """False for spectroscopy (1D) and other non-image datasets."""
        return self.dim >= 2 and self.data_path.exists()

    @property
    def size(self) -> Tuple[int, ...]:
        return tuple(int(s) for s in np.atleast_1d(self.visu.get('VisuCoreSize', [])))

    @property
    def extent(self) -> Tuple[float, ...]:
        return tuple(float(e) for e in np.atleast_1d(self.visu.get('VisuCoreExtent', [])))

    @property
    def frame_count(self) -> int:
        return int(self.visu.get('VisuCoreFrameCount', 1))

    @property
    def dtype(self) -> np.dtype:
        word = self.visu.get('VisuCoreWordType', '_16BIT_SGN_INT')
        if word not in WORD_TYPES:
            raise ValueError(f'{self.pdata_dir}: unsupported VisuCoreWordType {word}')
        order = '<' if self.visu.get('VisuCoreByteOrder', 'littleEndian') == 'littleEndian' else '>'
        return np.dtype(WORD_TYPES[word]).newbyteorder(order)

    @cached_property
    def frame_groups(self) -> List[FrameGroup]:
        """Frame groups in file order (first = fastest varying)."""
        desc = self.visu.get('VisuFGOrderDesc') or []
        groups = []
        for entry in desc:
            if isinstance(entry, (tuple, list)) and len(entry) >= 2:
                groups.append(FrameGroup(int(entry[0]), str(entry[1]),
                                         str(entry[2]) if len(entry) > 2 else ''))
        if not groups and self.frame_count > 1:
            groups.append(FrameGroup(self.frame_count, 'FG_SLICE' if self.dim == 2 else 'FG_FRAME', ''))
        return groups

    @property
    def slice_group_index(self) -> Optional[int]:
        for i, g in enumerate(self.frame_groups):
            if g.name == 'FG_SLICE':
                return i
        return None

    @property
    def n_slices(self) -> int:
        if self.dim >= 3:
            return self.size[2]
        i = self.slice_group_index
        return self.frame_groups[i].size if i is not None else 1

    @property
    def extra_groups(self) -> List[FrameGroup]:
        """Frame groups other than FG_SLICE, in file order."""
        return [g for g in self.frame_groups if g.name != 'FG_SLICE']

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of :meth:`load` output: ``(nx, ny, nz, *extra_groups)``."""
        return (self.size[0], self.size[1], self.n_slices, *[g.size for g in self.extra_groups])

    @property
    def slice_distance(self) -> float:
        dist = self.visu.get('VisuCoreSlicePacksSliceDist')
        if dist is not None and np.size(dist):
            return float(np.atleast_1d(dist)[0])
        thick = self.visu.get('VisuCoreFrameThickness')
        if thick is not None and np.size(thick):
            return float(np.atleast_1d(thick)[0])
        return 1.0

    @property
    def spacing(self) -> Tuple[float, float, float]:
        """Voxel size in mm as ``(dx, dy, dz)``."""
        dx = self.extent[0] / self.size[0]
        dy = self.extent[1] / self.size[1]
        if self.dim >= 3:
            dz = self.extent[2] / self.size[2]
        else:
            dz = self.slice_distance
        return (dx, dy, dz)

    @property
    def units(self) -> str:
        u = self.visu.get('VisuCoreUnits')
        if isinstance(u, list) and u:
            return str(u[0])
        return str(u or 'mm')

    @property
    def echo_times(self) -> List[float]:
        return [float(t) for t in np.atleast_1d(self.visu.get('VisuAcqEchoTime', []))]

    @property
    def repetition_time(self) -> Optional[float]:
        tr = self.visu.get('VisuAcqRepetitionTime')
        return float(np.atleast_1d(tr)[0]) if tr is not None and np.size(tr) else None

    @property
    def scan_time_s(self) -> Optional[float]:
        t = self.visu.get('VisuAcqScanTime')
        return float(t) / 1000.0 if isinstance(t, (int, float)) else None

    @property
    def orientation_label(self) -> str:
        return str(self.visu.get('VisuCoreOrientationLabel', '') or '')

    def group_labels(self, group: FrameGroup) -> List[str]:
        """Human-readable labels for each index of an extra frame group."""
        n = group.size
        if group.name == 'FG_ECHO' and len(self.echo_times) == n:
            return [f'TE {te:g} ms' for te in self.echo_times]
        if group.name == 'FG_DIFFUSION':
            comments = self.visu.get('VisuFGElemComment')
            if isinstance(comments, list) and len(comments) == n:
                return [str(c) for c in comments]
        if group.name in ('FG_MOVIE', 'FG_CYCLE') and self.repetition_time:
            return [f't = {i * self.repetition_time / 1000.0:.2f} s' for i in range(n)]
        return [f'{group.label} {i + 1}/{n}' for i in range(n)]

    # ---- data --------------------------------------------------------------
    def _frame_scaling(self) -> Tuple[np.ndarray, np.ndarray]:
        n = self.frame_count
        slope = np.atleast_1d(np.asarray(self.visu.get('VisuCoreDataSlope', [1.0]), dtype=np.float64))
        offs = np.atleast_1d(np.asarray(self.visu.get('VisuCoreDataOffs', [0.0]), dtype=np.float64))
        if slope.size == 1:
            slope = np.repeat(slope, n)
        if offs.size == 1:
            offs = np.repeat(offs, n)
        return slope[:n], offs[:n]

    @property
    def uniform_scaling(self) -> bool:
        slope, offs = self._frame_scaling()
        return bool(np.all(slope == slope[0]) and np.all(offs == offs[0]))

    def load_raw(self) -> np.ndarray:
        """Frames as stored: ``(frames, [nz,] ny, nx)`` in the file's dtype."""
        if not self.is_image:
            raise ValueError(f'{self.pdata_dir}: not an image dataset (VisuCoreDim={self.dim})')
        if self.dim == 2:
            frame_shape = (self.size[1], self.size[0])
        else:
            frame_shape = (self.size[2], self.size[1], self.size[0])
        expected = self.frame_count * int(np.prod(frame_shape))
        raw = np.fromfile(self.data_path, dtype=self.dtype)
        if raw.size != expected:
            raise ValueError(
                f'{self.data_path}: has {raw.size} values, expected {expected} '
                f'({self.frame_count} frames of {frame_shape})')
        return raw.reshape((self.frame_count, *frame_shape))

    def load(self, scaled: bool = True) -> np.ndarray:
        """Image array shaped ``(nx, ny, nz, *extra_groups)``.

        With ``scaled=True`` the per-frame ``VisuCoreDataSlope``/``Offs`` are
        applied and a float32 array is returned; otherwise the stored integers.
        """
        raw = self.load_raw()
        if scaled:
            slope, offs = self._frame_scaling()
            bcast = (slice(None),) + (None,) * (raw.ndim - 1)
            data = raw.astype(np.float32) * slope[bcast].astype(np.float32) + offs[bcast].astype(np.float32)
        else:
            data = raw

        # frames -> frame groups (first group is fastest varying, so reversed)
        group_sizes = [g.size for g in self.frame_groups]
        if not group_sizes and self.frame_count == 1:
            data = data[0]
        elif group_sizes and int(np.prod(group_sizes)) == self.frame_count:
            data = data.reshape((*reversed(group_sizes), *data.shape[1:]))
        else:
            warnings.warn(f'{self.pdata_dir}: frame groups {group_sizes} do not '
                          f'multiply to {self.frame_count} frames; treating as one axis')
            self.frame_groups[:] = [FrameGroup(self.frame_count, 'FG_FRAME', '')]

        # to (nx, ny, [nz], *groups in file order)
        data = data.transpose(tuple(reversed(range(data.ndim))))

        if self.dim == 2:
            i = self.slice_group_index
            if i is not None:
                data = np.moveaxis(data, 2 + i, 2)
            else:
                data = data[:, :, np.newaxis]

        if str(self.visu.get('VisuCoreDiskSliceOrder', '')) == 'disk_reverse_slice_order':
            data = data[:, :, ::-1]
        return data

    # ---- world geometry ------------------------------------------------------
    def affine(self) -> np.ndarray:
        """Index-to-world (mm, Bruker/DICOM patient LPS-style) 4x4 affine."""
        orient = np.asarray(self.visu.get('VisuCoreOrientation', np.eye(3).ravel()), dtype=float).reshape(-1, 3, 3)
        pos = np.asarray(self.visu.get('VisuCorePosition', [0, 0, 0]), dtype=float).reshape(-1, 3)
        rot = orient[0]
        dx, dy, dz = self.spacing
        aff = np.eye(4)
        aff[:3, 0] = rot[0] * dx
        aff[:3, 1] = rot[1] * dy

        slice_vec = rot[2] * dz
        if self.dim == 2 and self.n_slices > 1 and pos.shape[0] >= self.n_slices:
            i = self.slice_group_index
            stride = int(np.prod([g.size for g in self.frame_groups[:i]])) if i else 1
            idx = np.arange(self.n_slices) * stride
            if idx[-1] < pos.shape[0]:
                steps = np.diff(pos[idx], axis=0)
                if np.allclose(steps, steps[0], atol=1e-3):
                    slice_vec = steps[0]
                else:
                    logger.warning('%s: slices are not evenly spaced; using nominal slice distance', self.pdata_dir)
        aff[:3, 2] = slice_vec
        # VisuCorePosition is the corner of the first voxel; centre it.
        aff[:3, 3] = pos[0] + 0.5 * (aff[:3, 0] + aff[:3, 1])
        if self.dim >= 3:
            aff[:3, 3] += 0.5 * aff[:3, 2]
        return aff

    def nifti_affine(self) -> np.ndarray:
        """Affine in NIfTI RAS convention (negate x and y of the LPS affine)."""
        return np.diag([-1.0, -1.0, 1.0, 1.0]) @ self.affine()

    # ---- summary ---------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        dx, dy, dz = self.spacing if self.is_image and len(self.size) >= 2 else (0, 0, 0)
        return {
            'scan': self.scan_name,
            'proc': self.proc_no,
            'protocol': self.protocol,
            'method': self.method.replace('Bruker:', '').replace('User:', ''),
            'dim': f'{self.dim}D',
            'matrix': 'x'.join(str(s) for s in self.size),
            'slices': self.n_slices if self.is_image else 0,
            'extra': ', '.join(f'{g.label} {g.size}' for g in self.extra_groups),
            'voxel_mm': f'{dx:.3g}x{dy:.3g}x{dz:.3g}' if self.is_image and len(self.size) >= 2 else '',
            'TE_ms': ('/'.join(f'{t:g}' for t in self.echo_times[:3]) + ('...' if len(self.echo_times) > 3 else '')) if self.echo_times else '',
            'TR_ms': f'{self.repetition_time:g}' if self.repetition_time else '',
            'dtype': self.dtype.name,
            'frames': self.frame_count,
            'path': str(self.pdata_dir),
        }

    def __repr__(self) -> str:
        return f'BrukerImage({self.pdata_dir}, {self.protocol!r}, shape={self.shape if self.is_image else None})'
