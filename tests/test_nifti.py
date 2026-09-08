"""Viewing NIfTI files: orientation handling, the BrukerImage-shaped interface, CLI."""
import numpy as np
import pytest

nib = pytest.importorskip('nibabel')

from brukerview.cli import main
from brukerview.nifti import NiftiImage, find_niftis, is_nifti_path, to_nifti


def write_nifti(path, data, affine=None, zooms=None, units=None, descrip=None):
    nii = nib.Nifti1Image(data, np.eye(4) if affine is None else affine)
    if zooms:
        nii.header.set_zooms(zooms)
    if units:
        nii.header.set_xyzt_units(*units)
    if descrip:
        nii.header['descrip'] = descrip.encode()
    nib.save(nii, str(path))
    return path


def make_las(tmp_path, name='anat.nii.gz'):
    """A 3D file stored LAS (so reorienting to LPS has work to do)."""
    data = np.arange(16 * 12 * 8, dtype=np.float32).reshape(16, 12, 8)
    affine = np.array([[-0.5, 0, 0, 4.0], [0, 0.6, 0, -3.0], [0, 0, 1.2, -5.0], [0, 0, 0, 1]])
    return write_nifti(tmp_path / name, data, affine, descrip='t2 anat'), data, affine


def test_suffixes():
    assert is_nifti_path('a.nii') and is_nifti_path('a.NII.GZ') and is_nifti_path('a.hdr')
    assert not is_nifti_path('a.nii.txt') and not is_nifti_path('2dseq')


def test_geometry_and_summary(tmp_path):
    path, data, _ = make_las(tmp_path)
    img = NiftiImage(path)
    assert img.is_image and img.dim == 3
    assert img.size == (16, 12, 8) and img.n_slices == 8
    assert img.shape == img.load().shape == (16, 12, 8)
    assert np.allclose(img.spacing, (0.5, 0.6, 1.2))
    assert np.allclose(img.extent, (8.0, 7.2, 9.6))
    s = img.summary()
    assert (s['scan'], s['method'], s['matrix'], s['dtype']) == ('anat', 'nifti', '16x12x8', 'float32')
    assert s['protocol'] == 't2 anat' and s['extra'] == '' and s['TR_ms'] == ''
    assert img.extra_groups == [] and img.frame_count == 1


def test_reorient_to_display_axes_keeps_world_coordinates(tmp_path):
    path, data, affine = make_las(tmp_path)
    img = NiftiImage(path)
    stored = NiftiImage(path, reorient=False)
    assert stored.orientation_label == 'LAS'
    assert img.orientation_label == 'LPS'
    assert np.allclose(stored.nifti_affine(), affine, atol=1e-4)   # header floats are 32-bit

    # Same voxel, same place in the world, whichever axis order it arrives in.
    reoriented = img.load()
    idx = (3, 5, 6)
    world = nib.affines.apply_affine(affine, idx)
    match = np.argwhere(reoriented == data[idx])
    assert len(match) == 1
    assert np.allclose(nib.affines.apply_affine(img.nifti_affine(), match[0]), world, atol=1e-4)
    # Only the y axis flips between LAS and LPS.
    assert np.array_equal(reoriented, data[:, ::-1, :])
    assert np.allclose(img.spacing, stored.spacing)


def test_lps_file_is_left_alone(tmp_path):
    path = write_nifti(tmp_path / 'lps.nii', np.zeros((4, 4, 4), np.int16),
                       np.diag([-1.0, -1.0, 1.0, 1.0]))
    img = NiftiImage(path)
    assert img._ornt is None and img.orientation_label == 'LPS'


def test_fourth_dimension_is_a_frame_group(tmp_path):
    data = np.zeros((6, 6, 4, 3), np.int16)
    path = write_nifti(tmp_path / 'func.nii', data, np.diag([1.0, 1.0, 2.0, 1.0]),
                       zooms=(1, 1, 2, 1.5), units=('mm', 'sec'))
    img = NiftiImage(path)
    (group,) = img.extra_groups
    assert (group.size, group.name, group.label) == (3, 'FG_VOLUME', 'volume')
    assert img.frame_groups == img.extra_groups   # slices are spatial, no FG_SLICE
    assert img.repetition_time == 1500.0
    assert img.group_labels(group) == ['t = 0.00 s', 't = 1.50 s', 't = 3.00 s']
    assert img.summary()['extra'] == 'volume 3'
    assert img.shape == (6, 6, 4, 3) and img.frame_count == 3


def test_unused_time_zoom_is_not_a_tr(tmp_path):
    path = write_nifti(tmp_path / 'notime.nii', np.zeros((2, 2, 2, 4), np.int16))
    img = NiftiImage(path)
    assert img.repetition_time is None and img.summary()['TR_ms'] == ''
    assert img.group_labels(img.extra_groups[0])[:2] == ['volume 1/4', 'volume 2/4']


def test_millisecond_time_units(tmp_path):
    path = write_nifti(tmp_path / 'ms.nii', np.zeros((2, 2, 2, 2), np.int16),
                       zooms=(1, 1, 1, 2000), units=('mm', 'msec'))
    assert NiftiImage(path).repetition_time == 2000.0


def test_scaling_and_raw(tmp_path):
    stored = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)
    lps = np.diag([-1.0, -1.0, 1.0, 1.0])       # already display order: no reordering
    nii = nib.Nifti1Image(stored, lps)
    nii.header.set_slope_inter(3.0, 10.0)
    nii.header.set_data_dtype(np.int16)
    path = tmp_path / 'scaled.nii'
    nib.save(nii, str(path))

    img = NiftiImage(path)
    assert np.allclose(img.load(), stored * 3.0 + 10.0)
    assert img.load().dtype == np.float32
    assert np.array_equal(img.load(scaled=False), stored)
    assert img.load(scaled=False).dtype == np.int16
    assert img.dtype == np.int16


def test_two_dimensional_file_gets_a_slice_axis(tmp_path):
    path = write_nifti(tmp_path / 'plane.nii', np.zeros((5, 4), np.int16))
    img = NiftiImage(path)
    assert img.dim == 2 and img.load().shape == (5, 4, 1) and img.n_slices == 1
    assert img.spacing[2] == 1.0     # no third column in the affine to measure


def test_find_niftis(tmp_path):
    make_las(tmp_path, 'a.nii.gz')
    (tmp_path / 'sub').mkdir()
    write_nifti(tmp_path / 'sub' / 'b.nii', np.zeros((2, 2, 2), np.int16))
    (tmp_path / 'notes.txt').write_text('hello')
    names = sorted(i.scan_name for i in find_niftis(tmp_path))
    assert names == ['a', 'b']
    assert [i.scan_name for i in find_niftis(tmp_path / 'a.nii.gz')] == ['a']
    assert find_niftis(tmp_path / 'notes.txt') == []


def test_export_round_trip_and_overwrite_guard(tmp_path):
    path, data, _ = make_las(tmp_path)
    img = NiftiImage(path)
    out = to_nifti(img, tmp_path / 'copy.nii.gz')
    again = NiftiImage(out)
    assert np.allclose(again.load(), img.load())
    assert np.allclose(again.nifti_affine(), img.nifti_affine())
    with pytest.raises(ValueError, match='overwrite'):
        to_nifti(img, path)


def test_export_marks_only_a_real_time_axis(tmp_path):
    func = write_nifti(tmp_path / 'func.nii', np.zeros((4, 4, 2, 3), np.int16),
                       zooms=(1, 1, 1, 2.0), units=('mm', 'sec'))
    assert NiftiImage(to_nifti(NiftiImage(func), tmp_path / 'func2.nii.gz')).repetition_time == 2000.0
    plain = write_nifti(tmp_path / 'plain.nii', np.zeros((4, 4, 2, 3), np.int16))
    assert NiftiImage(to_nifti(NiftiImage(plain), tmp_path / 'plain2.nii.gz')).repetition_time is None


# ---- CLI ---------------------------------------------------------------------------
def test_cli_list_mixes_formats(tmp_path, capsys):
    make_las(tmp_path, 'a.nii.gz')
    write_nifti(tmp_path / 'func.nii', np.zeros((4, 4, 2, 5), np.int16))
    main(['list', str(tmp_path)])
    out = capsys.readouterr().out
    assert 'a' in out and 'func' in out and 'volume 5' in out


def test_cli_info_and_show(tmp_path, capsys):
    path, _, _ = make_las(tmp_path)
    main(['info', str(path)])
    out = capsys.readouterr().out
    assert 'nifti' in out and 'LPS' in out and '16x12x8' in out

    main(['info', str(path), '--no-reorient'])
    assert 'LAS' in capsys.readouterr().out

    main(['info', str(path), '--param', 'descrip'])
    assert 'descrip =' in capsys.readouterr().out

    png = tmp_path / 'snap.png'
    main([str(path), '--save', str(png)])     # bare path -> show
    assert png.stat().st_size > 0


def test_cli_missing_path(tmp_path):
    with pytest.raises(SystemExit, match='no such file'):
        main(['show', str(tmp_path / 'absent.nii.gz')])


def test_cli_imagej_refuses_nifti(tmp_path):
    path, _, _ = make_las(tmp_path)
    with pytest.raises(SystemExit, match='2dseq'):
        main(['imagej', str(path)])
