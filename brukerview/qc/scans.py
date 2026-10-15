"""Scan inventory for QC: protocol parameters and the role each scan plays.

Everything here comes from the text parameter files (``method``, ``acqp``,
``pdata/1/visu_pars``); no image data is read.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..jcampdx import read_jcampdx
from ..reader import time_per_repetition
from ..study import pdata_dirs, scan_dirs

logger = logging.getLogger(__name__)

# Roles, in report order.
ROLES = {
    'anat': 'Anatomical (RARE)',
    'dwi': 'Diffusion (DTI-EPI)',
    'epi_ref': 'Spin-echo EPI references (reverse PE)',
    'epi_single': 'Single-volume EPI',
    'bold': 'Functional (T2* GE-EPI)',
    'fieldmap': 'B0 field maps',
    'mrs': 'Spectroscopy / shim voxels',
    'localizer': 'Localizers',
    'adjustment': 'EPI ghost-correction adjustment scans',
    'other': 'Other',
}
ANALYSED_ROLES = ('anat', 'dwi', 'epi_ref', 'epi_single', 'bold', 'fieldmap', 'mrs')


def _first(v: Any, default=None):
    if v is None:
        return default
    if isinstance(v, (list, tuple)):
        return v[0] if len(v) else default
    return v


def _floats(v: Any) -> List[float]:
    if v is None:
        return []
    return [float(x) for x in np.atleast_1d(v) if isinstance(x, (int, float, np.number))]


def _strip(v: Any) -> str:
    return str(v or '').replace('Bruker:', '').replace('User:', '').strip('<>')


@dataclass
class Scan:
    """One acquisition: its folder, parsed parameter files and derived fields."""
    path: Path
    method: Dict[str, Any]
    acqp: Dict[str, Any]
    visu: Dict[str, Any]
    params: Dict[str, Any] = field(default_factory=dict)
    role: str = 'other'
    proc_dir: Optional[Path] = None   # the reconstruction QC reads (pdata/N)

    @property
    def number(self) -> int:
        return int(self.path.name)

    @property
    def pdata(self) -> Path:
        return self.proc_dir or self.path / 'pdata' / '1'

    @property
    def label(self) -> str:
        return f"E{self.number} {self.params.get('name', '')}".strip()


def choose_pdata(scan_dir: Path, method: Dict[str, Any], proc: Optional[str] = None) -> Optional[Path]:
    """The reconstruction to measure: ``proc`` if given and present; for a field
    map the one whose image is in Hz (ParaVision may put magnitude in another);
    otherwise pdata/1, else the first one."""
    procs = pdata_dirs(scan_dir)
    if not procs:
        return None
    if proc is not None and (scan_dir / 'pdata' / str(proc)) in procs:
        return scan_dir / 'pdata' / str(proc)
    if _strip(method.get('Method')) == 'FieldMap' and len(procs) > 1:
        for p in procs:
            units = read_jcampdx(p / 'visu_pars').get('VisuCoreDataUnits')
            if 'hz' in str(units).lower():
                return p
    first = scan_dir / 'pdata' / '1'
    return first if first in procs else procs[0]


def _scan_name(acqp: Dict[str, Any], visu: Dict[str, Any]) -> str:
    name = str(acqp.get('ACQ_scan_name') or visu.get('VisuAcquisitionProtocol') or '')
    return re.sub(r'\s*\(E\d+\)\s*$', '', name)


def extract_params(path: Path, method: Dict, acqp: Dict, visu: Dict) -> Dict[str, Any]:
    """Protocol parameters that matter for image quality, in real units."""
    m = method
    meth = _strip(m.get('Method') or visu.get('VisuAcqSequenceName'))
    dim = 3 if '3D' in str(m.get('PVM_SpatDimEnum', '')) else 2
    res = _floats(m.get('PVM_SpatResol'))
    fov = _floats(m.get('PVM_Fov'))
    matrix = [int(x) for x in _floats(m.get('PVM_Matrix'))]
    thick = float(m['PVM_SliceThick']) if dim == 2 and 'PVM_SliceThick' in m else (res[2] if len(res) > 2 else None)
    gap = _first(_floats(m.get('PVM_SPackArrSliceGap'))) if dim == 2 else None
    n_slices = int(_first(_floats(m.get('PVM_SPackArrNSlices')), 0) or 0) if dim == 2 else (matrix[2] if len(matrix) > 2 else None)
    te = _first(_floats(visu.get('VisuAcqEchoTime')))
    if not te and 'PVM_EchoTime' in m:
        te = float(m['PVM_EchoTime'])
    tr = _first(_floats(m.get('PVM_RepetitionTime')))
    shots = int(m['PVM_EpiNShots']) if 'PVM_EpiNShots' in m else None
    nr = int(m.get('PVM_NRepetitions', 1) or 1)
    na = int(m.get('PVM_NAverages', 1) or 1)
    scan_time = float(m['PVM_ScanTime']) / 1000.0 if 'PVM_ScanTime' in m else None
    vox_mm3 = (res[0] * res[1] * (thick or 1.0)) if len(res) >= 2 else None
    pft = max(_floats(m.get('PVM_EncPft')) or [1.0])
    ppi = max(_floats(m.get('PVM_EncPpi')) or [1.0])
    grad_enc = [str(g) for g in np.atleast_1d(visu.get('VisuAcqGradEncoding') or [])]
    pe_axis = grad_enc.index('phase_enc') if 'phase_enc' in grad_enc else None
    # VisuAcqGradEncoding follows the image axes, PVM_Fov is (read, phase[, slice]):
    # the phase-encode FOV is the image extent along pe_axis, not PVM_Fov[pe_axis].
    extent = _floats(visu.get('VisuCoreExtent'))
    pe_fov = extent[pe_axis] if pe_axis is not None and pe_axis < len(extent) else None
    if pe_fov is not None and len(fov) > 1 and abs(fov[1] - pe_fov) > 1e-3:
        logger.warning('%s: phase-encode extent %.3g mm differs from PVM_Fov[1] %.3g mm', path, pe_fov, fov[1])

    p: Dict[str, Any] = {
        'scan': int(path.name),
        'name': _scan_name(acqp, visu),
        'protocol': str(visu.get('VisuAcquisitionProtocol') or acqp.get('ACQ_protocol_name') or ''),
        'method': meth,
        'orientation': str(_first(m.get('PVM_SPackArrSliceOrient'), '') or ''),
        'read_dir': str(_first(m.get('PVM_SPackArrReadOrient'), '') or ''),
        'dim': f'{dim}D',
        'matrix': 'x'.join(str(x) for x in matrix),
        'fov_mm': 'x'.join(f'{x:g}' for x in fov),
        'inplane_um': 'x'.join(f'{x * 1000:.0f}' for x in res[:2]),
        'slice_thick_mm': thick,
        'slice_gap_mm': gap,
        'n_slices': n_slices,
        'voxel_nl': vox_mm3,  # 1 mm^3 = 1 uL; report in nL below
        'tr_ms': tr,
        'te_ms': te,
        'flip_deg': _first(_floats(visu.get('VisuAcqFlipAngle'))),
        'averages': na,
        'repetitions': nr,
        'rare_factor': int(m['PVM_RareFactor']) if meth == 'RARE' and 'PVM_RareFactor' in m else None,
        'epi_shots': shots,
        'partial_fourier': pft,
        'accel': ppi,
        'bw_khz': float(m['PVM_EffSWh']) / 1000.0 if 'PVM_EffSWh' in m else None,
        'epi_esp_ms': float(m['PVM_EpiEchoSpacing']) if 'PVM_EpiEchoSpacing' in m else None,
        'epi_eff_bw_hz': float(m['PVM_EpiEffBandwidth']) if 'PVM_EpiEffBandwidth' in m else None,
        'pe_axis': pe_axis,
        'pe_fov_mm': pe_fov,
        'reversed_pe': str(m.get('PVM_EpiReversedPE', '')) if 'PVM_EpiReversedPE' in m else None,
        'fat_sup': str(m.get('PVM_FatSupOnOff', '')) or None,
        'fov_sat': str(m.get('PVM_FovSatOnOff', '')) or None,
        'trigger': str(m.get('PVM_TriggerModule', '')) or None,
        'dummy_scans': int(m['PVM_DummyScans']) if 'PVM_DummyScans' in m else None,
        'receivers': int(m.get('PVM_EncNReceivers', 1) or 1),
        'scan_time_s': scan_time,
        'echo_type': str(visu.get('VisuAcqEchoSequenceType') or ''),
    }
    if p['voxel_nl'] is not None:
        p['voxel_nl'] = p['voxel_nl'] * 1000.0
    # Time per volume, by the same rule the viewer's time labels use
    # (reader.time_per_repetition): scan time / volumes. PVM_ScanTime excludes
    # dummy scans and includes averages, so this is the real sampling interval.
    # TR x shots is only the fallback (one TR per shot for segmented EPI).
    if meth in ('EPI', 'DtiEpi') and tr:
        n_vol = nr * (int(m.get('PVM_DwNDiffExp', 1) or 1) if meth == 'DtiEpi' else 1)
        p['vol_tr_s'] = time_per_repetition(scan_time, n_vol, tr * (shots or 1) / 1000.0)
    if meth == 'DtiEpi':
        p['b_values'] = '/'.join(f'{b:g}' for b in _floats(m.get('PVM_DwBvalEach')))
        p['n_dirs'] = int(m.get('PVM_DwNDiffDir', 0) or 0)
        p['n_b0'] = int(m.get('PVM_DwAoImages', 0) or 0)
        p['delta_ms'] = _first(_floats(m.get('PVM_DwGradDur')))
        p['Delta_ms'] = _first(_floats(m.get('PVM_DwGradSep')))
    if meth == 'PRESS':
        p.update(dim='SV', n_slices=None, matrix=None, inplane_um=None, fov_mm=None, voxel_nl=None)
        vox = _floats(m.get('PVM_VoxArrSize'))
        p['voxel_nl'] = float(np.prod(vox)) * 1000.0 if len(vox) == 3 else None
        p['voxel_mm'] = 'x'.join(f'{v:.1f}' for v in vox)
        p['spec_sw_hz'] = _first(_floats(m.get('PVM_SpecSWH')))
        p['spec_points'] = int(_first(_floats(m.get('PVM_SpecMatrix')), 0) or 0)
        p['water_sup'] = str(m.get('PVM_WsOnOff', ''))
        p['freq_mhz'] = _first(_floats(m.get('PVM_FrqWork')))
    return p


def classify(scan: Scan) -> str:
    p, name = scan.params, scan.params.get('protocol', '') + ' ' + scan.params.get('name', '')
    meth = p.get('method')
    if (scan.path / 'EpiAdjOrigDset').exists() or name.upper().startswith('ADJ_GHOST'):
        return 'adjustment'
    if meth == 'FieldMap':
        return 'fieldmap'
    if meth == 'PRESS':
        return 'mrs'
    if 'localizer' in name.lower() or 'tripilot' in name.lower():
        return 'localizer'
    if meth == 'DtiEpi':
        return 'dwi'
    if meth in ('EPI', 'T2S_EPI'):
        if p.get('repetitions', 1) > 1:
            return 'bold'
        # a topup reference is a spin-echo EPI or one acquired with reversed PE;
        # any other one-volume EPI (a gradient-echo scout) is not
        spin_echo = p.get('echo_type') in ('SpinEcho', 'Both')
        return 'epi_ref' if spin_echo or p.get('reversed_pe') == 'Yes' else 'epi_single'
    if meth in ('RARE', 'FLASH', 'FISP', 'MSME', 'TurboRARE'):
        return 'anat'
    return 'other'


def load_scan(path: Path, proc: Optional[str] = None) -> Optional[Scan]:
    method_p, acqp_p = path / 'method', path / 'acqp'
    if not method_p.exists():
        return None
    method = read_jcampdx(method_p)
    acqp = read_jcampdx(acqp_p) if acqp_p.exists() else {}
    pdata = choose_pdata(path, method, proc)
    visu = read_jcampdx(pdata / 'visu_pars') if pdata else {}
    scan = Scan(path, method, acqp, visu, proc_dir=pdata)
    scan.params = extract_params(path, method, acqp, visu)
    scan.role = classify(scan)
    return scan


def inventory(study_dir: Path, proc: Optional[str] = None) -> List[Scan]:
    """All scans of one study, in acquisition order."""
    out = []
    for d in scan_dirs(Path(study_dir)):
        s = load_scan(d, proc)
        if s is not None:
            out.append(s)
    return out


def study_info(study_dir: Path) -> Dict[str, str]:
    study_dir = Path(study_dir)
    info = {'study_dir': str(study_dir), 'name': study_dir.name}
    subj = study_dir / 'subject'
    if subj.exists():
        s = read_jcampdx(subj)
        for key, k in [('subject_id', 'SUBJECT_id'), ('study_name', 'SUBJECT_study_name'),
                       ('operator', 'SUBJECT_name_string'), ('sex', 'SUBJECT_gender'),
                       ('weight_kg', 'SUBJECT_weight'), ('remarks', 'SUBJECT_remarks')]:
            if s.get(k) not in (None, ''):
                info[key] = str(s.get(k)).replace('^', ' ').strip()
    for sc in scan_dirs(study_dir):
        v = sc / 'pdata' / '1' / 'visu_pars'
        if v.exists():
            vp = read_jcampdx(v)
            info.setdefault('date', str(vp.get('VisuStudyDate', '')).split(',')[0].replace('T', ' '))
            info.setdefault('station', str(vp.get('VisuStation', '') or ''))
            info.setdefault('software', f"ParaVision {vp.get('VisuCreatorVersion', '')}")
            break
    return info


# Parameters shown in the protocol comparison table: key -> (label, formatter)
PARAM_COLUMNS = [
    ('name', 'Protocol', str),
    ('orientation', 'Slice orient.', str),
    ('read_dir', 'Read dir.', str),
    ('dim', 'Dim', str),
    ('matrix', 'Matrix', str),
    ('fov_mm', 'FOV (mm)', str),
    ('inplane_um', 'In-plane (µm)', str),
    ('slice_thick_mm', 'Slice / thk (mm)', lambda v: f'{v:.2f}'),
    ('slice_gap_mm', 'Gap (mm)', lambda v: f'{v:.2f}'),
    ('n_slices', 'Slices', lambda v: f'{v:d}'),
    ('voxel_nl', 'Voxel (nL)', lambda v: f'{v:.1f}'),
    ('tr_ms', 'TR (ms)', lambda v: f'{v:g}'),
    ('vol_tr_s', 'Vol. TR (s)', lambda v: f'{v:.2f}'),
    ('te_ms', 'TE (ms)', lambda v: f'{v:.1f}'),
    ('flip_deg', 'Flip (°)', lambda v: f'{v:g}'),
    ('averages', 'NA', lambda v: f'{v:d}'),
    ('repetitions', 'NR', lambda v: f'{v:d}'),
    ('rare_factor', 'RARE f.', lambda v: f'{v:d}'),
    ('epi_shots', 'Shots', lambda v: f'{v:d}'),
    ('partial_fourier', 'PF', lambda v: f'{v:.2f}'),
    ('bw_khz', 'BW (kHz)', lambda v: f'{v:.0f}'),
    ('epi_esp_ms', 'ESP (ms)', lambda v: f'{v:.3f}'),
    ('epi_eff_bw_hz', 'PE BW (Hz/FOV)', lambda v: f'{v:.0f}'),
    ('reversed_pe', 'Rev. PE', str),
    ('b_values', 'b (s/mm²)', str),
    ('n_dirs', 'Dirs', lambda v: f'{v:d}'),
    ('n_b0', 'b0', lambda v: f'{v:d}'),
    ('delta_ms', 'δ (ms)', lambda v: f'{v:g}'),
    ('Delta_ms', 'Δ (ms)', lambda v: f'{v:g}'),
    ('fat_sup', 'Fat sup.', str),
    ('fov_sat', 'FOV sat.', str),
    ('trigger', 'Trigger', str),
    ('dummy_scans', 'Dummies', lambda v: f'{v:d}'),
    ('voxel_mm', 'Voxel (mm)', str),
    ('spec_sw_hz', 'SW (Hz)', lambda v: f'{v:.0f}'),
    ('water_sup', 'Water sup.', str),
    ('scan_time_s', 'Time', lambda v: f'{int(v // 60)}:{int(round(v % 60)):02d}'),
]


def fmt_param(key: str, value: Any) -> str:
    if value is None or value == '':
        return ''
    for k, _, f in PARAM_COLUMNS:
        if k == key:
            try:
                return f(value)
            except (TypeError, ValueError):
                return str(value)
    return str(value)


def parameter_table(scans: List[Scan]):
    """Columns that are defined for any scan, and which of them vary.

    Returns ``(columns, rows, varying)`` with ``columns`` a list of
    ``(key, label)``, ``rows`` lists of formatted strings, ``varying`` the set
    of keys whose formatted value is not identical across the scans.
    """
    cols = [(k, lab) for k, lab, _ in PARAM_COLUMNS if any(fmt_param(k, s.params.get(k)) for s in scans)]
    rows = [[fmt_param(k, s.params.get(k)) for k, _ in cols] for s in scans]
    varying = {k for i, (k, _) in enumerate(cols) if len({r[i] for r in rows}) > 1 and k != 'name'}
    return cols, rows, varying
