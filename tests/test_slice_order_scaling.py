"""Slice order and intensity scaling rules, in the Python reader and (opt-in) the
ImageJ macro, which must agree pixel for pixel.

The ImageJ half runs the real macro headless and is skipped unless
BRUKERVIEW_IMAGEJ_TESTS=1 and an ImageJ/Fiji install is found.
"""
import os
import re
import subprocess

import numpy as np
import pytest

from brukerview.imagej import MACRO_PATH, find_imagej, is_wsl
from brukerview.reader import BrukerImage
from test_reader import make_scan


def set_visu(pdata, **params):
    """Replace or add ``##$Key=value`` entries in a synthetic visu_pars."""
    path = pdata / 'visu_pars'
    text = path.read_text()
    for key, value in params.items():
        pat = re.compile(rf'^##\$({key})=.*?(?=^##)', re.M | re.S)
        entry = f'##${key}={value}\n'
        text = pat.sub(entry, text) if pat.search(text) else text.replace('##END=', entry + '##END=')
    path.write_text(text)


def scan_with(tmp_path, nslices=4, nechoes=2, **params):
    scan, raw = make_scan(tmp_path, nslices=nslices, nechoes=nechoes)
    pdata = scan / 'pdata' / '1'
    set_visu(pdata, **params)
    return pdata, raw


def world(aff, i, j, k):
    return (aff @ np.array([i, j, k, 1.0]))[:3]


# ---- slice order -----------------------------------------------------------------------
@pytest.mark.parametrize('value', ['disk_reverse_slice_order', '( 1 )\n<disk_reverse_slice_order>'])
def test_reversed_slices_move_array_and_affine_together(tmp_path, value):
    normal = BrukerImage(scan_with(tmp_path / 'a')[0])
    rev = BrukerImage(scan_with(tmp_path / 'b', VisuCoreDiskSliceOrder=value)[0])
    assert rev.reversed_slices and not normal.reversed_slices
    a, b = normal.load(), rev.load()
    n = a.shape[2]
    for k in range(n):
        # the same tissue: identical pixels at the same world position
        np.testing.assert_array_equal(b[:, :, k], a[:, :, n - 1 - k])
        np.testing.assert_allclose(world(rev.affine(), 2, 3, k), world(normal.affine(), 2, 3, n - 1 - k))


def test_pv360_array_form_is_recognised(tmp_path):
    pdata, _ = scan_with(tmp_path, VisuCoreDiskSliceOrder='( 1 )\n<disk_normal_slice_order>')
    assert not BrukerImage(pdata).reversed_slices


# ---- intensity scaling -----------------------------------------------------------------
def test_per_slice_slope_is_repeated_over_echoes(tmp_path):
    slopes = [1.0, 2.0, 3.0, 4.0]          # one per slice; frames = 2 echoes x 4 slices
    pdata, _ = scan_with(tmp_path, nslices=4, nechoes=2,
                         VisuCoreDataSlope=f'( 4 )\n{" ".join(map(str, slopes))}')
    img = BrukerImage(pdata)
    raw, scaled = img.load(scaled=False), img.load()
    for z, s in enumerate(slopes):
        np.testing.assert_allclose(scaled[:, :, z], raw[:, :, z] * s)


def test_slope_of_unusable_length_is_refused(tmp_path):
    pdata, _ = scan_with(tmp_path, nslices=4, nechoes=2, VisuCoreDataSlope='( 3 )\n1 2 3')
    with pytest.raises(ValueError, match='3 values for 8 frames'):
        BrukerImage(pdata).load()


def test_signed_8bit(tmp_path):
    pdata, _ = scan_with(tmp_path, nslices=2, nechoes=1, VisuCoreWordType='_8BIT_SGN_INT',
                         VisuCoreDataSlope='( 2 )\n1 1')
    vals = (np.arange(2 * 6 * 8) % 256 - 128).astype(np.int8)
    vals.tofile(pdata / '2dseq')
    np.testing.assert_array_equal(np.sort(BrukerImage(pdata).load(scaled=False).ravel()), np.sort(vals))


# ---- the ImageJ macro must agree --------------------------------------------------------
IJ = find_imagej() if os.environ.get('BRUKERVIEW_IMAGEJ_TESTS') == '1' else None
needs_imagej = pytest.mark.skipif(IJ is None, reason='set BRUKERVIEW_IMAGEJ_TESTS=1 with ImageJ installed')

DUMP_MACRO = r"""
arg = getArgument();
parts = split(arg, ";");
runMacro(parts[0], parts[1]);
// a completed import ends by renaming the image "<scan>_<protocol> [pN]";
// anything else means the macro stopped (exit() only ends the inner macro)
if (nImages == 0 || indexOf(getTitle(), " [p") < 0) {
    File.saveString("FAILED", parts[2]);
    exit;
}
getDimensions(w, h, c, z, t);
s = "" + w + " " + h + " " + c + " " + z + " " + t + "\n";
for (tt = 1; tt <= t; tt++) for (zz = 1; zz <= z; zz++) for (cc = 1; cc <= c; cc++) {
    Stack.setPosition(cc, zz, tt);
    for (y = 0; y < h; y++) for (x = 0; x < w; x++) s = s + d2s(getPixel(x, y), 9) + " ";
    s = s + "\n";
}
File.saveString(s, parts[2]);
"""


def _host(p, exe):
    if exe.suffix.lower() == '.exe' and is_wsl():
        return subprocess.check_output(['wslpath', '-w', str(p)], text=True).strip()
    return str(p)


def run_macro(pdata, workdir, raw=False):
    """(z, t, planes) as ImageJ imports them; planes[t][z] is a (ny, nx) array."""
    dump, out = workdir / 'dump.ijm', workdir / 'out.txt'
    dump.write_text(DUMP_MACRO)
    out.unlink(missing_ok=True)
    arg = f'{_host(MACRO_PATH, IJ)};{_host(pdata, IJ)}{"|raw" if raw else ""}|nobc|nozoom;{_host(out, IJ)}'
    subprocess.run([str(IJ), '-batch', _host(dump, IJ), arg], cwd=str(IJ.parent),
                   stdin=subprocess.DEVNULL, capture_output=True, timeout=300)
    if not out.exists():
        return None
    lines = out.read_text().splitlines()
    if lines[0] == 'FAILED':
        return None
    w, h, c, z, t = map(int, lines[0].split())
    planes = [np.array(l.split(), float).reshape(h, w) for l in lines[1:] if l.strip()]
    return z, t, [planes[i * z:(i + 1) * z] for i in range(t)]


def assert_matches_python(pdata, workdir, raw=False):
    got = run_macro(pdata, workdir, raw=raw)
    assert got is not None, 'ImageJ produced no output'
    z, t, planes = got
    py = BrukerImage(pdata).load(scaled=not raw)
    py = py.reshape(py.shape[:3] + (-1,))
    assert (z, t) == (py.shape[2], py.shape[3])
    for ti in range(t):
        for zi in range(z):
            np.testing.assert_allclose(planes[ti][zi], py[:, :, zi, ti].T, rtol=1e-6)


@pytest.fixture
def ij_dir():
    """ImageJ under WSL is a Windows program: keep its files on a Windows drive."""
    base = os.environ.get('BRUKERVIEW_IMAGEJ_TMP')
    if base:
        import tempfile
        return __import__('pathlib').Path(tempfile.mkdtemp(dir=base))
    return None


@needs_imagej
@pytest.mark.parametrize('order', ['disk_normal_slice_order', '( 1 )\n<disk_reverse_slice_order>'])
def test_imagej_slice_order_matches_python(tmp_path, ij_dir, order):
    d = ij_dir or tmp_path
    pdata, _ = scan_with(d, VisuCoreDiskSliceOrder=order)
    assert_matches_python(pdata, d)


@needs_imagej
def test_imagej_per_slice_slope_matches_python(tmp_path, ij_dir):
    d = ij_dir or tmp_path
    pdata, _ = scan_with(d, nslices=4, nechoes=2, VisuCoreDataSlope='( 4 )\n1 2 3 4')
    assert_matches_python(pdata, d)


@needs_imagej
def test_imagej_keeps_full_slope_precision(tmp_path, ij_dir):
    # string conversion in the macro used to round this to 0.04074 (1e-4 error)
    d = ij_dir or tmp_path
    pdata, _ = scan_with(d, nslices=3, nechoes=2, VisuCoreDataSlope='( 6 )\n' + ' '.join(['0.04073596'] * 6))
    assert_matches_python(pdata, d)


@needs_imagej
def test_imagej_refuses_unusable_slope(tmp_path, ij_dir):
    d = ij_dir or tmp_path
    pdata, _ = scan_with(d, nslices=4, nechoes=2, VisuCoreDataSlope='( 3 )\n1 2 3')
    assert run_macro(pdata, d) is None


@needs_imagej
def test_imagej_signed_8bit_matches_python(tmp_path, ij_dir):
    d = ij_dir or tmp_path
    pdata, _ = scan_with(d, nslices=2, nechoes=1, VisuCoreWordType='_8BIT_SGN_INT',
                         VisuCoreDataSlope='( 2 )\n1 1')
    (np.arange(2 * 6 * 8) % 256 - 128).astype(np.int8).tofile(pdata / '2dseq')
    assert_matches_python(pdata, d, raw=True)
