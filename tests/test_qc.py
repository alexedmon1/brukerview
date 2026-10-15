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
