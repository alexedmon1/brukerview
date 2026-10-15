"""Run every QC metric on a study. Returns plain dicts (metrics) and arrays (for figures)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import ndimage as ndi

from ..reader import BrukerImage
from . import dwi as dwi_mod
from . import mrs as mrs_mod
from .fieldmap import b0_stats, predicted_distortion
from .geometry import (brain_center_world, brain_mask_t2, object_mask, resample_mask,
                       slab_coverage)
from .metrics import (BG_RATIO_VALID, background_mask, csf_contrast, drift_percent_per_min, dvars,
                      ghost_ratio, ghost_regions, sigma_air_pockets, sigma_laplacian, snr_background,
                      snr_consecutive, snr_efficiency, summarize, tsnr_map)
from .scans import Scan

logger = logging.getLogger(__name__)


@dataclass
class Reference:
    scan: Scan
    image: BrukerImage
    data: np.ndarray
    affine: np.ndarray
    mask: np.ndarray
    how: str
    center: np.ndarray = field(default=None)

    @property
    def volume_mm3(self) -> float:
        return float(self.mask.sum() * abs(np.linalg.det(self.affine[:3, :3])))


@dataclass
class FieldMap:
    scan: Scan
    hz: np.ndarray
    affine: np.ndarray
    support: np.ndarray


@dataclass
class Result:
    scan: Scan
    metrics: Dict[str, Any] = field(default_factory=dict)
    arrays: Dict[str, Any] = field(default_factory=dict)


# ---- helpers ----------------------------------------------------------------------------
def _image(scan: Scan) -> BrukerImage:
    return BrukerImage(scan.pdata)


def brain_masks(ref: Reference, aff: np.ndarray, shape, two_d: bool):
    """Brain mask in a scan's grid and its 1-voxel eroded core (in-plane only for 2-D)."""
    brain = resample_mask(ref.mask, ref.affine, aff, shape)
    st = np.zeros((3, 3, 3), bool)
    if two_d:
        st[:, :, 1] = ndi.generate_binary_structure(2, 1)
    else:
        st = ndi.generate_binary_structure(3, 1)
    core = ndi.binary_erosion(brain, structure=st)
    return brain, core


def choose_reference(scans: List[Scan], number: Optional[int] = None) -> Optional[Scan]:
    if number is not None:
        for s in scans:
            if s.number == number:
                return s
        raise ValueError(f'reference scan {number} not found')
    cands = [s for s in scans if s.role == 'anat' and s.params.get('dim') == '3D' and s.params.get('method') == 'RARE']
    cands = cands or [s for s in scans if s.role == 'anat']
    if not cands:
        return None
    return min(cands, key=lambda s: (s.params.get('voxel_nl') or 1e9, -s.number))


def build_reference(scan: Scan, mask_path: Optional[Path] = None, use_bet: bool = True) -> Reference:
    img = _image(scan)
    data = img.load()
    data = data if data.ndim == 3 else data.reshape(data.shape[:3] + (-1,)).mean(-1)
    aff = img.nifti_affine()
    if mask_path:
        import nibabel as nib
        mi = nib.load(str(mask_path))
        m = np.asarray(mi.dataobj) > 0
        mask = m if m.shape == data.shape and np.allclose(mi.affine, aff, atol=1e-3) \
            else resample_mask(m, mi.affine, aff, data.shape)
        how = f'user-supplied mask {Path(mask_path).name}'
    else:
        mask, how = brain_mask_t2(data, aff, use_bet=use_bet)
    ref = Reference(scan, img, data, aff, mask, how)
    ref.center = brain_center_world(mask, aff)
    return ref


def load_fieldmap(scan: Scan) -> FieldMap:
    img = _image(scan)
    hz = img.load().astype(np.float32)
    support = ndi.binary_erosion(hz != 0, iterations=1)
    return FieldMap(scan, hz, img.nifti_affine(), support)


def pick_fieldmap(fmaps: List[FieldMap], scan: Scan) -> Optional[FieldMap]:
    before = [f for f in fmaps if f.scan.number < scan.number]
    pool = before or fmaps
    return max(pool, key=lambda f: f.scan.number) if pool else None


def _epi_geometry(res: Result, img: BrukerImage, mean_img: np.ndarray, brain: np.ndarray,
                  core: np.ndarray, ref: Reference, fmap: Optional[FieldMap]):
    """Ghosting, coverage, predicted distortion: shared by every EPI role."""
    p, m = res.scan.params, res.metrics
    aff = img.nifti_affine()
    obj = object_mask(mean_img)
    pe = p.get('pe_axis')
    if pe is not None:
        m.update({k: v for k, v in ghost_ratio(mean_img, obj, core, pe, p.get('epi_shots') or 1).items()})
    in_slab, sampled = slab_coverage(ref.mask, ref.affine, aff, mean_img.shape, p.get('slice_thick_mm'))
    m['coverage_slab_pct'] = 100 * in_slab
    m['coverage_sampled_pct'] = 100 * sampled
    if fmap is not None and p.get('epi_eff_bw_hz') and p.get('pe_fov_mm') and pe is not None:
        d = predicted_distortion(fmap.hz, fmap.affine, fmap.support, aff, mean_img.shape, brain,
                                 p['pe_fov_mm'], p['epi_eff_bw_hz'], img.spacing[pe])
        res.arrays['disp_map'] = d.pop('_map', None)
        m.update(d)
        m['fieldmap_scan'] = fmap.scan.number
    res.arrays['obj'] = obj
    return obj


# ---- per role -----------------------------------------------------------------------------
def analyze_anat(scan: Scan, ref: Reference) -> Result:
    res = Result(scan)
    img = _image(scan)
    data = img.load()
    vol = data if data.ndim == 3 else data.reshape(data.shape[:3] + (-1,))[..., 0]
    aff = img.nifti_affine()
    if scan is ref.scan:
        brain = ref.mask
        core = ndi.binary_erosion(brain)
    else:
        brain, core = brain_masks(ref, aff, vol.shape, two_d=scan.params.get('dim') == '2D')
    obj = object_mask(vol)
    bg = background_mask(vol, obj, margin=4)
    s = snr_background(vol, core, bg, scan.params.get('receivers', 1))
    p = scan.params
    lo, hi = BG_RATIO_VALID
    two_d = p.get('dim') == '2D'
    if lo <= s['bg_ratio'] <= hi:
        method = 'background'
    else:  # no usable air around the head: air pockets inside it, else the Laplacian residual
        ap = sigma_air_pockets(vol, p.get('receivers', 1), two_d=two_d)
        if lo <= ap['ratio'] <= hi:
            sig, method = ap['sigma'], 'air pockets'
            s['bg_ratio'] = ap['ratio']
        else:
            sig, method = sigma_laplacian(vol, core, two_d=two_d), 'laplacian'
        s = dict(s, sigma=sig, snr=s['signal'] / sig)
    res.metrics.update({'snr': s['snr'], 'sigma': s['sigma'], 'signal': s['signal'],
                        'noise_method': method, 'bg_ratio': s['bg_ratio']})
    res.metrics.update(csf_contrast(vol, core, s['sigma']))
    res.metrics.update(snr_efficiency(s['snr'], p.get('voxel_nl'), p.get('scan_time_s')))
    if res.metrics.get('cnr') is not None:
        res.metrics.update({f'cnr_{k[4:]}': v for k, v in
                            snr_efficiency(res.metrics['cnr'], p.get('voxel_nl'), p.get('scan_time_s')).items()})
    res.arrays.update(vol=vol, aff=aff, brain=brain, bg=bg)
    return res


def analyze_dwi(scan: Scan, ref: Reference, fmap: Optional[FieldMap]) -> Result:
    res = Result(scan)
    img = _image(scan)
    data = img.load()  # (x, y, z, n_diff, n_rep)
    if data.ndim == 4:
        data = data[..., None]
    aff = img.nifti_affine()
    shells = dwi_mod.shell_labels(scan.method)
    brain, core = brain_masks(ref, aff, data.shape[:3], two_d=True)
    avg = data.mean(axis=-1)
    b0 = avg[..., shells == 0].mean(axis=-1)
    m = res.metrics
    if data.shape[-1] >= 2:
        rows = dwi_mod.shell_snr(data, shells, core)
        res.arrays['shell_snr'] = rows
        for r in rows:
            m[f"snr_b{int(r['b'])}"] = r['snr']
        b0r = next(r for r in rows if r['b'] == 0)
        m['snr_b0'] = b0r['snr']
        m['sigma_diff'] = b0r['sigma']
    obj = _epi_geometry(res, img, b0, brain, core, ref, fmap)
    gr = ghost_regions(obj, scan.params['pe_axis'], scan.params.get('epi_shots') or 1) \
        if scan.params.get('pe_axis') is not None else None
    bg = background_mask(data[..., int(np.flatnonzero(shells == 0)[0]), 0], obj, margin=4,
                         exclude=ndi.binary_dilation(gr, iterations=2) if gr is not None else None)
    m['snr_b0_bg'] = snr_background(data[..., int(np.flatnonzero(shells == 0)[0]), 0], core, bg,
                                    scan.params.get('receivers', 1))['snr']
    p = scan.params
    if m.get('snr_b0'):
        m.update({f'b0_{k}': v for k, v in snr_efficiency(m['snr_b0'], p.get('voxel_nl'), p.get('scan_time_s')).items()})
    bmat = dwi_mod.bmatrices(scan.method)
    if bmat is not None and core.any():
        FA, MD = dwi_mod.fit_tensor(avg, bmat, brain, beff=np.asarray(scan.method.get('PVM_DwEffBval'), float))
        m['fa_median'] = float(np.median(FA[core]))
        m['md_median_um2_ms'] = float(np.median(MD[core]) * 1000)
        res.arrays.update(FA=FA, MD=MD)
    shell_means = {b: avg[..., shells == b].mean(axis=-1) for b in np.unique(shells)}
    res.arrays.update(b0=b0, shells=shell_means, aff=aff, brain=brain, core=core)
    return res


def analyze_epi_ref(scan: Scan, ref: Reference, fmap: Optional[FieldMap]) -> Result:
    res = Result(scan)
    img = _image(scan)
    data = img.load()
    vol = data if data.ndim == 3 else data.reshape(data.shape[:3] + (-1,)).mean(-1)
    aff = img.nifti_affine()
    brain, core = brain_masks(ref, aff, vol.shape, two_d=True)
    obj = _epi_geometry(res, img, vol, brain, core, ref, fmap)
    pe = scan.params.get('pe_axis')
    gr = ghost_regions(obj, pe, scan.params.get('epi_shots') or 1) if pe is not None else None
    bg = background_mask(vol, obj, margin=4, exclude=ndi.binary_dilation(gr, iterations=2) if gr is not None else None)
    s = snr_background(vol, core, bg, scan.params.get('receivers', 1))
    res.metrics.update(snr=s['snr'], sigma=s['sigma'], bg_ratio=s['bg_ratio'])
    res.metrics.update(snr_efficiency(s['snr'], scan.params.get('voxel_nl'), scan.params.get('scan_time_s')))
    res.arrays.update(mean=vol, aff=aff, brain=brain, core=core)
    return res


def analyze_bold(scan: Scan, ref: Reference, fmap: Optional[FieldMap]) -> Result:
    res = Result(scan)
    img = _image(scan)
    data = img.load()
    ts = data.reshape(data.shape[:3] + (-1,)).astype(np.float32)
    aff = img.nifti_affine()
    p, m = scan.params, res.metrics
    vol_tr = p.get('vol_tr_s') or (p.get('scan_time_s', 0) / max(ts.shape[-1], 1))
    mean = ts.mean(-1)
    brain, core = brain_masks(ref, aff, mean.shape, two_d=True)
    obj = _epi_geometry(res, img, mean, brain, core, ref, fmap)
    tmap = tsnr_map(ts, order=1)
    tv = tmap[core]
    s = summarize(tv)
    m.update(tsnr=s.get('median'), tsnr_q1=s.get('q1'), tsnr_q3=s.get('q3'), n_vols=int(ts.shape[-1]))
    if ts.shape[-1] >= 20:
        m['tsnr_first10'] = float(np.median(tsnr_map(ts[..., :10], order=1)[core]))
    # single-image SNR, difference method on consecutive volumes (same estimator as DWI)
    m['snr_diff'] = snr_consecutive(ts, core).get('snr')
    m['tsnr_per_sqrt_s'] = m['tsnr'] / np.sqrt(vol_tr) if m['tsnr'] else None
    if p.get('voxel_nl') and m['tsnr']:
        m['tsnr_per_nl_sqrt_s'] = m['tsnr'] / p['voxel_nl'] / np.sqrt(vol_tr)
    series = ts[core].mean(axis=0)
    duration = ts.shape[-1] * vol_tr
    # a slope over < 1 min of data is noise, not drift
    m['drift_pct_per_min'] = drift_percent_per_min(series, vol_tr) if duration >= 60 else None
    dv = dvars(ts, core)
    m['dvars_median_pct'] = float(np.median(dv)) if dv.size else None
    m['first_vol_pct'] = float(100 * (series[0] / series[1:].mean() - 1)) if series.size > 1 else None
    slice_tsnr = [float(np.median(tmap[..., k][core[..., k]])) if core[..., k].sum() > 20 else np.nan
                  for k in range(tmap.shape[2])]
    res.arrays.update(mean=mean, tsnr=tmap, aff=aff, brain=brain, core=core, series=series, dvars=dv,
                      slice_tsnr=slice_tsnr, vol_tr=vol_tr)
    return res


def analyze_fieldmap(fm: FieldMap, ref: Reference) -> Result:
    res = Result(fm.scan)
    brain = resample_mask(ref.mask, ref.affine, fm.affine, fm.hz.shape) & fm.support
    freq = fm.scan.method.get('PVM_FrqWork')
    f0 = float(np.atleast_1d(freq)[0]) if freq is not None else None
    res.metrics.update(b0_stats(fm.hz, brain, freq_mhz=f0))
    res.metrics['brain_support_pct'] = float(100 * brain.sum() / max(resample_mask(ref.mask, ref.affine, fm.affine, fm.hz.shape).sum(), 1))
    v = fm.hz[brain]
    res.arrays.update(hz=fm.hz, aff=fm.affine, brain=brain, values=v - np.median(v) if v.size else v)
    return res


def analyze_mrs(scan: Scan) -> Result:
    res = Result(scan)
    p = scan.params
    out = mrs_mod.water_linewidth(scan.path, p.get('spec_sw_hz') or 0, p.get('freq_mhz'))
    res.arrays['spectrum'] = out.pop('_spectrum', None)
    res.metrics.update(out)
    return res


# ---- driver ------------------------------------------------------------------------------
def run_all(scans: List[Scan], reference: Optional[int] = None, brain_mask: Optional[Path] = None,
            use_bet: bool = True, only: Optional[set] = None, progress=print) -> Dict[str, Any]:
    want = (lambda s: True) if not only else (lambda s: s.number in only)
    ref_scan = choose_reference(scans, reference)
    if ref_scan is None:
        raise SystemExit('no anatomical scan to build a brain mask from (use --reference)')
    progress(f'reference E{ref_scan.number} {ref_scan.params["name"]}: building brain mask')
    ref = build_reference(ref_scan, brain_mask, use_bet)
    progress(f'  brain mask {ref.volume_mm3:.0f} mm3 ({ref.how})')

    out: Dict[str, Any] = {'reference': ref, 'results': {r: [] for r in
                                                         ('anat', 'dwi', 'epi_ref', 'bold', 'fieldmap', 'mrs')}}
    fmaps = [load_fieldmap(s) for s in scans if s.role == 'fieldmap']
    for fm in fmaps:
        if want(fm.scan):
            progress(f'E{fm.scan.number} field map')
            out['results']['fieldmap'].append(analyze_fieldmap(fm, ref))
    for s in scans:
        if not want(s):
            continue
        try:
            if s.role == 'anat':
                progress(f'E{s.number} anatomical {s.params["name"]}')
                out['results']['anat'].append(analyze_anat(s, ref))
            elif s.role == 'dwi':
                progress(f'E{s.number} diffusion {s.params["name"]}')
                out['results']['dwi'].append(analyze_dwi(s, ref, pick_fieldmap(fmaps, s)))
            elif s.role == 'epi_ref':
                progress(f'E{s.number} SE-EPI {s.params["name"]}')
                out['results']['epi_ref'].append(analyze_epi_ref(s, ref, pick_fieldmap(fmaps, s)))
            elif s.role == 'bold':
                progress(f'E{s.number} BOLD {s.params["name"]}')
                out['results']['bold'].append(analyze_bold(s, ref, pick_fieldmap(fmaps, s)))
            elif s.role == 'mrs':
                progress(f'E{s.number} spectroscopy {s.params["name"]}')
                out['results']['mrs'].append(analyze_mrs(s))
        except Exception as e:  # keep going: one broken scan should not sink the report
            logger.exception('E%s failed', s.number)
            progress(f'  E{s.number} FAILED: {e}')
    return out
