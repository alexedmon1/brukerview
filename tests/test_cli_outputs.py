"""Where `nifti` and `fsleyes` read from and write to."""
import pytest

pytest.importorskip('nibabel')

from brukerview import cli  # noqa: E402
from brukerview.reader import BrukerImage  # noqa: E402
from test_reader import make_scan  # noqa: E402


def test_nifti_o_without_suffix_is_a_new_folder(tmp_path):
    scan, _ = make_scan(tmp_path / 'study')
    out = tmp_path / 'outdir'
    cli.main(['nifti', str(scan), '-o', str(out)])
    assert out.is_dir() and len(list(out.glob('*.nii.gz'))) == 1


def test_nifti_o_with_suffix_is_a_file(tmp_path):
    scan, _ = make_scan(tmp_path / 'study')
    out = tmp_path / 't2.nii.gz'
    cli.main(['nifti', str(scan), '-o', str(out)])
    assert out.is_file()


def test_nifti_all_on_study_with_one_image_makes_folder(tmp_path):
    scan, _ = make_scan(tmp_path / 'study')
    out = tmp_path / 'out'
    cli.main(['nifti', '--all', str(scan.parent), '-o', str(out)])
    assert out.is_dir()


def test_all_on_a_file_means_that_file(tmp_path):
    study = tmp_path / 'study'
    scan, _ = make_scan(study)
    nii = tmp_path / 'other.nii.gz'
    cli.main(['nifti', str(scan), '-o', str(nii)])
    # a NIfTI beside a Bruker study: --all on the file must not pull the study in
    nii_in_study = study / 'x.nii.gz'
    nii.rename(nii_in_study)
    images = cli._all_images(str(nii_in_study), all_procs=False)
    assert len(images) == 1 and str(images[0].path) == str(nii_in_study)


def test_fsleyes_cache_name_separates_raw_and_paths(tmp_path):
    a, _ = make_scan(tmp_path / 'one')
    b, _ = make_scan(tmp_path / 'two')
    ia, ib = BrukerImage(a / 'pdata' / '1'), BrukerImage(b / 'pdata' / '1')
    names = {cli._cache_name(ia, raw=False), cli._cache_name(ia, raw=True), cli._cache_name(ib, raw=False)}
    assert len(names) == 3
    assert all(n.endswith('.nii.gz') and all(ch.isalnum() or ch in '._-' for ch in n) for n in names)
