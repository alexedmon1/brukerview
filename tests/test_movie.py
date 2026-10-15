"""Movie playback/export in the slice viewer (headless)."""
import matplotlib

matplotlib.use('Agg')

import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from brukerview.cli import main  # noqa: E402
from brukerview.reader import BrukerImage  # noqa: E402
from brukerview.viewer import SliceViewer  # noqa: E402
from test_reader import make_scan  # noqa: E402


@pytest.fixture
def viewer(tmp_path):
    scan, _ = make_scan(tmp_path, nslices=3, nechoes=4)
    return SliceViewer(BrukerImage(scan / 'pdata' / '1'))


def test_default_dim_is_longest_non_slice(viewer):
    assert viewer.dim_names == ['slice', 'echo']
    assert viewer.movie_dim == 1


def test_resolve_dim(viewer):
    assert viewer.resolve_dim('slice') == 0
    assert viewer.resolve_dim('Echo') == 1
    with pytest.raises(ValueError, match='slice, echo'):
        viewer.resolve_dim('time')


def test_tick_wraps_and_play_toggles(viewer):
    viewer.toggle_play()
    assert viewer.playing
    for _ in range(4):
        viewer._tick()
    assert viewer.idx == [0, 0]  # 4 echoes: back to the first
    viewer._tick()
    assert viewer.idx == [0, 1]
    viewer.cycle_movie_dim()
    assert viewer.movie_dim == 0
    viewer.set_fps(1000)
    assert viewer.fps == 60
    viewer.toggle_play()
    assert not viewer.playing


def test_save_gif_has_one_frame_per_position(viewer, tmp_path):
    out = viewer.save_movie(str(tmp_path / 'echo.gif'), fps=5)
    with Image.open(out) as im:
        assert im.n_frames == 4
    out = viewer.save_movie(str(tmp_path / 'slices.gif'), dim='slice')
    with Image.open(out) as im:
        assert im.n_frames == 3


def test_cli_movie_export(tmp_path, capsys):
    scan, _ = make_scan(tmp_path, nslices=5, nechoes=1)
    out = tmp_path / 'm.gif'
    main(['show', str(scan), '--movie', str(out), '--fps', '4'])
    assert out.exists() and str(out) in capsys.readouterr().out
    with Image.open(out) as im:
        assert im.n_frames == 5  # no extra dimension -> plays slices


def test_unsupported_format(viewer, tmp_path):
    with pytest.raises(SystemExit, match='.gif'):
        viewer.save_movie(str(tmp_path / 'x.webm'))


def test_time_labels_use_scan_time_per_repetition(viewer):
    from brukerview.reader import FrameGroup
    img = viewer.img
    cyc = FrameGroup(4, 'FG_CYCLE', '')
    img.visu['VisuAcqRepetitionTime'] = 1000   # per shot
    img.visu['VisuAcqScanTime'] = 8000         # 4 volumes of 2 shots
    assert img.group_labels(cyc) == ['t = 0.00 s', 't = 2.00 s', 't = 4.00 s', 't = 6.00 s']
    del img.visu['VisuAcqScanTime']
    assert img.group_labels(cyc)[1] == 't = 1.00 s'  # falls back to TR


def test_time_resolves_to_a_nifti_volume_axis(tmp_path):
    nib = pytest.importorskip('nibabel')
    import numpy as np
    from brukerview.nifti import NiftiImage
    p = tmp_path / 'bold.nii.gz'
    nib.save(nib.Nifti1Image(np.random.rand(8, 6, 3, 5).astype('float32'), np.eye(4)), str(p))
    v = SliceViewer(NiftiImage(p))
    assert v.dim_names == ['slice', 'volume']
    assert v.resolve_dim('time') == 1
