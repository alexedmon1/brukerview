"""Round-trip tests on a synthetic 2dseq plus the sample data if present."""
import os
from pathlib import Path

import numpy as np
import pytest

from brukerview.reader import BrukerImage
from brukerview.study import find_images

SAMPLE_ROOT = Path(os.environ.get('BRUKERVIEW_TEST_DATA', '/mnt/e/research/mouse-mri'))


def make_scan(tmp_path, nx=8, ny=6, nslices=3, nechoes=2, slope=2.0):
    """MSME-like scan: echo fastest, then slice."""
    scan = tmp_path / '7'
    pdata = scan / 'pdata' / '1'
    pdata.mkdir(parents=True)
    (scan / 'acqp').write_text('##TITLE=acqp\n##END=\n')
    (scan / 'method').write_text('##TITLE=method\n##$Method=<Bruker:MSME>\n##END=\n')
    frames = nslices * nechoes
    data = np.arange(frames * ny * nx, dtype=np.int16).reshape(frames, ny, nx)
    data.astype('<i2').tofile(pdata / '2dseq')
    (pdata / 'visu_pars').write_text(f"""##TITLE=Parameter List
##$VisuCoreFrameCount={frames}
##$VisuCoreDim=2
##$VisuCoreSize=( 2 )
{nx} {ny}
##$VisuCoreExtent=( 2 )
16 18
##$VisuCoreFrameThickness=( 1 )
0.5
##$VisuCoreUnits=( 2, 65 )
<mm> <mm>
##$VisuCoreOrientation=( {frames}, 9 )
{' '.join(['1 0 0 0 1 0 0 0 1'] * frames)}
##$VisuCorePosition=( {frames}, 3 )
{' '.join(f'-8 -9 {s * 0.7:.1f}' for s in range(nslices) for e in range(nechoes))}
##$VisuCoreDataOffs=( {frames} )
@{frames}*(0)
##$VisuCoreDataSlope=( {frames} )
@{frames}*({slope})
##$VisuCoreWordType=_16BIT_SGN_INT
##$VisuCoreByteOrder=littleEndian
##$VisuFGOrderDescDim=2
##$VisuFGOrderDesc=( 2 )
({nechoes}, <FG_ECHO>, <>, 0, 1) ({nslices}, <FG_SLICE>, <>, 1, 2)
##$VisuAcqEchoTime=( {nechoes} )
{' '.join(str(10 * (i + 1)) for i in range(nechoes))}
##$VisuAcqRepetitionTime=( 1 )
2400
##$VisuAcquisitionProtocol=( 65 )
<MSME>
##END=
""")
    return scan, data


def test_synthetic_layout(tmp_path):
    scan, data = make_scan(tmp_path)
    img = BrukerImage(scan / 'pdata' / '1')
    assert img.protocol == 'MSME'
    assert img.method == 'Bruker:MSME'
    assert img.shape == (8, 6, 3, 2)
    assert img.n_slices == 3
    assert [g.label for g in img.extra_groups] == ['echo']
    assert img.group_labels(img.extra_groups[0]) == ['TE 10 ms', 'TE 20 ms']
    assert img.spacing == (2.0, 3.0, 0.5)

    vol = img.load(scaled=False)
    assert vol.shape == (8, 6, 3, 2)
    # frame index = slice * nechoes + echo; value at (x, y) = frame*ny*nx + y*nx + x
    for s in range(3):
        for e in range(2):
            frame = s * 2 + e
            assert vol[3, 2, s, e] == frame * 6 * 8 + 2 * 8 + 3

    scaled = img.load()
    assert scaled.dtype == np.float32
    assert np.allclose(scaled, vol * 2.0)

    aff = img.affine()
    assert np.allclose(aff[:3, 2], [0, 0, 0.7])          # slice step from positions
    assert np.allclose(aff[:3, 3], [-8 + 1.0, -9 + 1.5, 0])  # corner -> voxel centre
    ras = img.nifti_affine()
    assert np.allclose(ras[0, 0], -2.0) and np.allclose(ras[1, 1], -3.0)


def test_scan_level_visu_pars_is_not_a_reconstruction(tmp_path):
    """ParaVision writes a visu_pars beside the fid too; only pdata/N has a 2dseq."""
    from brukerview.study import is_pdata_dir, is_scan_dir, resolve_image
    scan, _ = make_scan(tmp_path)
    (scan / 'visu_pars').write_text('##TITLE=acquisition visu_pars\n##$VisuCoreDim=2\n##END=\n')

    assert not is_pdata_dir(scan)
    assert is_scan_dir(scan)
    img = resolve_image(scan)
    assert img.pdata_dir == scan / 'pdata' / '1'
    assert img.is_image and img.shape == (8, 6, 3, 2)

    assert len(find_images(tmp_path)) == 1


def test_unreconstructed_pdata_is_skipped(tmp_path):
    from brukerview.study import pdata_dirs
    scan, _ = make_scan(tmp_path)
    empty = scan / 'pdata' / '2'
    empty.mkdir()
    (empty / 'visu_pars').write_text('##TITLE=x\n##END=\n')
    assert [p.name for p in pdata_dirs(scan)] == ['1']


def test_find_images(tmp_path):
    make_scan(tmp_path)
    imgs = find_images(tmp_path)
    assert len(imgs) == 1
    assert imgs[0].scan_name == '7'
    assert imgs[0].summary()['matrix'] == '8x6'


def test_nifti_roundtrip(tmp_path):
    nib = pytest.importorskip('nibabel')
    from brukerview.nifti import to_nifti
    scan, _ = make_scan(tmp_path)
    img = BrukerImage(scan / 'pdata' / '1')
    out = to_nifti(img, tmp_path / 'out.nii.gz')
    nii = nib.load(str(out))
    assert nii.shape == (8, 6, 3, 2)
    assert np.allclose(nii.header.get_zooms()[:3], (2.0, 3.0, 0.5))
    assert np.allclose(np.asarray(nii.dataobj), img.load())


@pytest.mark.skipif(not SAMPLE_ROOT.exists(), reason='sample Bruker data not available')
def test_sample_data_loads():
    imgs = find_images(SAMPLE_ROOT)
    assert len(imgs) >= 1
    for img in imgs:
        if not img.is_image:
            continue
        data = img.load()
        assert data.shape == img.shape, img.pdata_dir
        assert np.isfinite(data).all()
