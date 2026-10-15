"""QC metrics on synthetic data with known answers."""
import numpy as np
import pytest

pytest.importorskip('scipy')

from brukerview.qc import charts, dwi, fieldmap, geometry, metrics, mrs  # noqa: E402
from brukerview.qc.report import md_to_html  # noqa: E402

RNG = np.random.default_rng(0)


def sos_magnitude(signal, sigma, n_ch, shape):
    """Sum-of-squares magnitude of n channels with equal signal share."""
    acc = np.zeros(shape)
    for _ in range(n_ch):
        acc += (signal / np.sqrt(n_ch) + RNG.normal(0, sigma, shape)) ** 2 + RNG.normal(0, sigma, shape) ** 2
    return np.sqrt(acc)


@pytest.mark.parametrize('n', [1, 2, 4])
def test_magnitude_sd_factor_matches_simulation(n):
    noise = sos_magnitude(0.0, 1.0, n, (200000,))
    assert metrics.magnitude_sd_factor(n) == pytest.approx(noise.std(), rel=0.01)
    assert metrics.magnitude_sd_factor(1) == pytest.approx(0.655, abs=1e-3)


def test_snr_difference_recovers_sigma():
    shape = (64, 64, 4)
    brain = np.zeros(shape, bool)
    brain[16:48, 16:48] = True
    a = 100 + RNG.normal(0, 5, shape)
    b = 100 + RNG.normal(0, 5, shape)
    r = metrics.snr_difference(a, b, brain)
    assert r['sigma'] == pytest.approx(5, rel=0.03)
    assert r['snr'] == pytest.approx(20, rel=0.03)


def test_background_snr_with_two_channels():
    shape = (96, 96, 6)
    obj = np.zeros(shape, bool)
    obj[30:66, 30:66] = True
    vol = sos_magnitude(np.where(obj, 200.0, 0.0), 4.0, 2, shape)
    bg = metrics.background_mask(vol, obj, margin=4)
    r = metrics.snr_background(vol, obj, bg, n_channels=2)
    assert r['sigma'] == pytest.approx(4.0, rel=0.05)
    assert metrics.BG_RATIO_VALID[0] < r['bg_ratio'] < metrics.BG_RATIO_VALID[1]


def test_air_pockets_find_noise_inside_tissue():
    shape = (80, 80, 40)
    vol = sos_magnitude(np.full(shape, 300.0), 6.0, 2, shape)
    pocket = np.zeros(shape, bool)
    pocket[30:42, 30:42, 14:26] = True  # an airway: no signal
    vol[pocket] = sos_magnitude(0.0, 6.0, 2, (int(pocket.sum()),))
    r = metrics.sigma_air_pockets(vol, n_channels=2, fraction=0.01)
    assert r['sigma'] == pytest.approx(6.0, rel=0.1)
    assert 2.4 < r['ratio'] < 3.1


def test_tsnr_ignores_linear_drift():
    t = np.arange(100)
    ts = (1000 + 2.0 * t)[None, None, None, :] + RNG.normal(0, 10, (8, 8, 2, 100))
    tm = metrics.tsnr_map(ts, order=1)
    assert np.median(tm) == pytest.approx(ts.mean() / 10, rel=0.05)
    assert metrics.drift_percent_per_min(ts.mean((0, 1, 2)), 1.0) == pytest.approx(100 * 120 / ts.mean(), rel=0.02)


def test_ghost_ratio_measures_nyquist_ghost():
    shape = (64, 64, 3)
    obj = np.zeros(shape, bool)
    obj[24:40, 20:36] = True
    img = 10 + np.where(obj, 1000.0, 0.0)
    img += 50 * np.roll(obj, 32, axis=1)  # 5% N/2 ghost along PE (axis 1)
    r = metrics.ghost_ratio(img, obj, obj, pe_axis=1, n_shots=1)
    assert r['ghost_pct'] == pytest.approx(5.0, abs=0.3)


def test_tensor_fit_recovers_fa_md():
    evals = np.array([1.7e-3, 0.3e-3, 0.3e-3])
    D = np.diag(evals)
    dirs = RNG.normal(size=(30, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    bmat = np.concatenate([np.zeros((3, 3, 3)), np.stack([1000 * np.outer(g, g) for g in dirs])])
    S = 1000 * np.exp(-np.einsum('nij,ij->n', bmat, D))
    data = np.broadcast_to(S, (2, 2, 1, S.size)).copy()
    mask = np.ones((2, 2, 1), bool)
    FA, MD = dwi.fit_tensor(data, bmat, mask)
    md = evals.mean()
    fa = np.sqrt(1.5) * np.linalg.norm(evals - md) / np.linalg.norm(evals)
    assert FA[0, 0, 0] == pytest.approx(fa, abs=1e-3)
    assert MD[0, 0, 0] == pytest.approx(md, rel=1e-3)


def test_shell_labels_from_effective_bvalues():
    method = {'PVM_DwEffBval': [34, 34, 527, 1031, 2040], 'PVM_DwBvalEach': [500, 1000, 2000]}
    assert list(dwi.shell_labels(method)) == [0, 0, 500, 1000, 2000]


def test_displacement_formula():
    # 100 Hz with a 24 mm PE FOV and 6271 Hz effective PE bandwidth -> 0.383 mm
    assert fieldmap.displacement_mm(np.array(100.0), 24.0, 6270.9) == pytest.approx(0.3827, abs=1e-3)


def test_slab_coverage_counts_gaps():
    brain = np.ones((10, 10, 40), bool)
    aff = np.diag([1.0, 1.0, 0.1, 1.0])          # brain spans z = 0..3.9 mm
    tgt = np.diag([1.0, 1.0, 1.0, 1.0])          # 2 slices at z = 0, 1 mm, thickness 0.5
    in_slab, sampled = geometry.slab_coverage(brain, aff, tgt, (10, 10, 2), slice_thick=0.5)
    assert in_slab == pytest.approx(15 / 40, abs=0.03)   # z <= 1.5 mm
    assert sampled == pytest.approx(8 / 40, abs=0.03)    # |z - k| <= 0.25 mm


def test_water_fwhm_with_group_delay(tmp_path):
    sw, n, fwhm = 6000.0, 2048, 20.0
    t = np.arange(n) / sw
    fid = np.exp(-np.pi * fwhm * t) * np.exp(1j * 0.7)
    fid = np.concatenate([np.zeros(70), fid[:n - 70]])  # digital filter delay
    (tmp_path / 'pdata' / '1').mkdir(parents=True)
    fid.astype('<c16').tofile(tmp_path / 'pdata' / '1' / 'fid_proc.64')
    r = mrs.water_linewidth(tmp_path, sw, 300.0)
    assert r['fwhm_hz'] == pytest.approx(fwhm, abs=3.0)  # bin width 2.9 Hz


def test_otsu_and_object_mask():
    vol = np.zeros((40, 40, 10))
    vol[10:30, 10:30] = 100
    vol += RNG.normal(0, 3, vol.shape)
    m = geometry.object_mask(vol)
    assert m[20, 20, 5] and not m[2, 2, 5]


def test_charts_and_markdown_render():
    svg = charts.line_chart([{'name': 'a', 'slot': 1, 'points': [(0, 1, 'tip'), (1, 2, 'tip2')]},
                             {'name': 'b', 'slot': 2, 'points': [(0, 2), (1, 3)]}], 'x', 'y')
    assert svg.startswith('<svg') and '<title>tip</title>' in svg and 'endlabel' in svg
    dots = charts.dot_rows(['r1', 'r2'], [{'title': 'm', 'values': [1.0, None], 'better': 'high'}])
    assert 'best' in dots and 'n/a' in dots
    html = md_to_html('# Head\n\nSome **bold** text.\n- item one\n- item `two`')
    assert '<h2>Head</h2>' in html and '<strong>bold</strong>' in html and '<li>item <code>two</code></li>' in html


# ---- fixes from the 0.1.0 code review -------------------------------------------------
from pathlib import Path  # noqa: E402

from brukerview.qc import report, scans  # noqa: E402
from brukerview.reader import time_per_repetition  # noqa: E402


def epi_params(**over):
    method = {'Method': 'Bruker:EPI', 'PVM_Fov': [32, 24], 'PVM_SpatResol': [0.25, 0.25], 'PVM_Matrix': [128, 96],
              'PVM_SliceThick': 0.8, 'PVM_RepetitionTime': 1000, 'PVM_EpiNShots': 2, 'PVM_NRepetitions': 10,
              'PVM_ScanTime': 20000, 'PVM_EpiEffBandwidth': 6000.0}
    visu = {'VisuAcqGradEncoding': ['read_enc', 'phase_enc'], 'VisuCoreExtent': [32, 24],
            'VisuAcqEchoSequenceType': 'GradientEcho', 'VisuAcqEchoTime': [10]}
    method.update(over.pop('method', {}))
    visu.update(over.pop('visu', {}))
    return method, visu


def test_pe_fov_follows_image_axis_not_pvm_fov_index():
    m, v = epi_params()
    assert scans.extract_params(Path('21'), m, {}, v)['pe_fov_mm'] == 24      # PE on image axis 1
    m, v = epi_params(visu={'VisuAcqGradEncoding': ['phase_enc', 'read_enc'], 'VisuCoreExtent': [24, 32]})
    # PE on image axis 0: its length is VisuCoreExtent[0] = 24, not PVM_Fov[0] = 32 (the read FOV)
    assert scans.extract_params(Path('21'), m, {}, v)['pe_fov_mm'] == 24


def test_volume_time_is_scan_time_over_volumes():
    # NA=2 doubles the time per volume; TR x shots does not know that
    m, v = epi_params(method={'PVM_NAverages': 2, 'PVM_ScanTime': 40000})
    assert scans.extract_params(Path('21'), m, {}, v)['vol_tr_s'] == pytest.approx(4.0)
    assert time_per_repetition(40.0, 10, 2.0) == pytest.approx(4.0)
    assert time_per_repetition(None, 10, 2.0) == 2.0           # no scan time: TR x shots
    m, v = epi_params(method={'Method': 'Bruker:DtiEpi', 'PVM_DwNDiffExp': 75, 'PVM_NRepetitions': 2,
                              'PVM_ScanTime': 1500000, 'PVM_EpiNShots': 4, 'PVM_RepetitionTime': 2500})
    assert scans.extract_params(Path('13'), m, {}, v)['vol_tr_s'] == pytest.approx(10.0)


@pytest.mark.parametrize('echo, rev, role', [('GradientEcho', 'No', 'epi_single'),
                                              ('Both', 'No', 'epi_ref'), ('SpinEcho', 'No', 'epi_ref'),
                                              ('GradientEcho', 'Yes', 'epi_ref')])
def test_single_volume_epi_roles(tmp_path, echo, rev, role):
    m, v = epi_params(method={'PVM_NRepetitions': 1, 'PVM_EpiReversedPE': rev},
                      visu={'VisuAcqEchoSequenceType': echo})
    s = scans.Scan(tmp_path / '15', m, {}, v)
    s.params = scans.extract_params(Path('15'), m, {}, v)
    assert scans.classify(s) == role


def test_fieldmap_pdata_in_hz_is_chosen(tmp_path):
    scan = tmp_path / '12'
    for n, units in (('1', '<a.u.>'), ('2', '<Hz>')):
        d = scan / 'pdata' / n
        d.mkdir(parents=True)
        (d / '2dseq').write_bytes(b'')
        (d / 'visu_pars').write_text(f'##$VisuCoreDataUnits=( 1, 65 )\n{units}\n##END=\n')
    fm = {'Method': 'Bruker:FieldMap'}
    assert scans.choose_pdata(scan, fm).name == '2'
    assert scans.choose_pdata(scan, fm, proc='1').name == '1'                # --proc wins
    assert scans.choose_pdata(scan, {'Method': 'Bruker:RARE'}).name == '1'


def test_fieldmap_support_keeps_on_resonance_zeros():
    from brukerview.qc.analyze import FieldMap  # noqa: F401  (import check)
    from brukerview.qc.geometry import fill_holes_2d
    from scipy import ndimage as ndi
    hz = np.zeros((20, 20, 10), np.float32)
    hz[5:15, 5:15, 2:8] = 40.0
    hz[9:11, 9:11, 4:6] = 0.0          # exactly on resonance, inside the object
    support = fill_holes_2d(ndi.binary_fill_holes(hz != 0)) & np.isfinite(hz)
    assert support[10, 10, 5] and not support[1, 1, 5]


def test_dwi_axes_found_by_name_in_any_order():
    x = np.arange(4 * 3 * 2 * 5 * 2, dtype=float).reshape(4, 3, 2, 5, 2)    # (x, y, z, diffusion, cycle)
    swapped = np.moveaxis(x, 3, 4)                                         # repetitions stored faster
    a = dwi.arrange(x, ['diffusion', 'cycle'], 5)
    b = dwi.arrange(swapped, ['cycle', 'diffusion'], 5)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (4, 3, 2, 5, 2)
    with pytest.raises(ValueError, match='7 b-values'):
        dwi.arrange(x, ['diffusion', 'cycle'], 7)


def test_dwi_shell_snr_without_b0():
    shells = np.array([1000.0, 1000.0, 2000.0])
    data = 100 + RNG.normal(0, 5, (16, 16, 2, 3, 2))
    rows = dwi.shell_snr(data, shells, np.ones((16, 16, 2), bool))
    assert [r['b'] for r in rows] == [1000.0, 2000.0]


def test_dot_rows_draw_numpy_scalars():
    svg = charts.dot_rows(['a', 'b', 'c'], [{'title': 'm', 'better': 'high',
                                             'values': [np.float64(1.0), np.float32(2.0), np.int64(3)],
                                             'lo': [np.float32(0.5), None, None], 'hi': [np.float32(1.5), None, None]}])
    assert 'n/a' not in svg and svg.count('class="dot') == 3 and 'whisker' in svg


def test_sweep_skips_scan_without_tsnr():
    class R:  # just what sweep_series reads
        def __init__(self, n, thick, tsnr):
            self.scan = type('S', (), {'number': n, 'params': {'slice_thick_mm': thick, 'slice_gap_mm': 0.2,
                                                                'n_slices': 9}})()
            self.metrics = {'tsnr': tsnr, 'coverage_sampled_pct': 80.0}
    sweeps = [(None, [R(1, 0.5, 8.0), R(2, 0.6, None), R(3, 0.8, 11.0)])]
    s1, s2 = report.sweep_series(sweeps, ['TR 750'])
    assert [p[0] for p in s1[0]['points']] == [0.5, 0.8] and len(s2[0]['points']) == 3


def test_background_ratio_uses_the_trimmed_sample():
    shape = (96, 96, 6)
    obj = np.zeros(shape, bool)
    obj[30:66, 30:66] = True
    vol = sos_magnitude(np.where(obj, 200.0, 0.0), 4.0, 2, shape)
    bg = metrics.background_mask(vol, obj, margin=4)
    idx = np.flatnonzero(bg.ravel())[:: 150]          # well under 1% of the background
    vol.ravel()[idx] = 5000.0                          # stray ghost/flow voxels
    assert metrics.BG_RATIO_VALID[0] < metrics.background_ratio(vol, bg) < metrics.BG_RATIO_VALID[1]
