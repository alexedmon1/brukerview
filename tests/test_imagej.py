"""ImageJ discovery and path translation, on WSL and on native Linux/macOS."""
from pathlib import Path

import pytest

from brukerview import imagej


def make_imagej_dir(root: Path, exe_name: str) -> Path:
    d = root / 'ImageJ'
    (d / 'plugins').mkdir(parents=True)
    exe = d / exe_name
    exe.write_text('#!/bin/sh\n')
    exe.chmod(0o755)
    return exe


def test_explicit_dir_finds_linux_launcher(tmp_path):
    exe = make_imagej_dir(tmp_path, 'ImageJ')
    assert imagej.find_imagej(str(exe.parent)) == exe


def test_env_var_is_used(tmp_path, monkeypatch):
    exe = make_imagej_dir(tmp_path, 'ImageJ-linux64')
    monkeypatch.setenv('BRUKERVIEW_IMAGEJ', str(exe))
    assert imagej.find_imagej() == exe


def test_windows_candidates_only_searched_under_wsl(monkeypatch):
    """A dual-boot Linux box may mount Windows at /mnt/c; the .exe is unusable there."""
    seen = []

    def fake_exists(self):
        seen.append(str(self))
        return False

    monkeypatch.setattr(Path, 'exists', fake_exists)
    monkeypatch.setattr(Path, 'is_dir', lambda self: False)
    monkeypatch.setattr(imagej.shutil, 'which', lambda name: None)

    monkeypatch.setattr(imagej, 'is_wsl', lambda: False)
    assert imagej.find_imagej() is None
    assert not any('/mnt/c' in s for s in seen)

    seen.clear()
    monkeypatch.setattr(imagej, 'is_wsl', lambda: True)
    assert imagej.find_imagej() is None
    assert any('/mnt/c' in s for s in seen)
    # the Windows install keeps priority under WSL, as before
    assert '/mnt/c' in seen[0]


def test_host_path_converts_only_for_windows_exe_under_wsl(monkeypatch):
    monkeypatch.setattr(imagej.subprocess, 'check_output',
                        lambda *a, **k: 'E:\\data\\5\\pdata\\1\n')
    p = Path('/mnt/e/data/5/pdata/1')

    monkeypatch.setattr(imagej, 'is_wsl', lambda: True)
    assert imagej._host_path(p, Path('/mnt/c/Program Files/ImageJ/ImageJ.exe')) == \
        'E:\\data\\5\\pdata\\1'
    # a Linux launcher under WSL still takes Linux paths
    assert imagej._host_path(p, Path('/opt/ImageJ/ImageJ')) == str(p)

    monkeypatch.setattr(imagej, 'is_wsl', lambda: False)
    assert imagej._host_path(p, Path('/opt/Fiji.app/ImageJ-linux64')) == str(p)


def test_launch_builds_macro_command(tmp_path, monkeypatch):
    exe = make_imagej_dir(tmp_path, 'ImageJ')
    calls = {}

    def fake_popen(cmd, **kwargs):
        calls['cmd'], calls['kwargs'] = cmd, kwargs
        return None

    monkeypatch.setattr(imagej.subprocess, 'Popen', fake_popen)
    pdata = tmp_path / '5' / 'pdata' / '1'
    cmd = imagej.launch(pdata, exe, raw=True, bc=False)
    assert cmd[:2] == [str(exe), '-macro']
    assert cmd[2] == str(imagej.MACRO_PATH)
    assert cmd[3] == f'{pdata}|raw|nobc'
    assert calls['kwargs']['cwd'] == str(exe.parent)


def test_install_macro(tmp_path):
    exe = make_imagej_dir(tmp_path, 'ImageJ')
    dest = imagej.install_macro(exe.parent)
    assert dest.read_text() == imagej.MACRO_PATH.read_text()


def make_fiji_app(root: Path) -> Path:
    """macOS layout: launcher in Contents/MacOS, plugins at the bundle root."""
    app = root / 'Fiji.app'
    (app / 'plugins').mkdir(parents=True)
    (app / 'Contents' / 'MacOS').mkdir(parents=True)
    exe = app / 'Contents' / 'MacOS' / 'ImageJ-macosx'
    exe.write_text('#!/bin/sh\n')
    return exe


def test_fiji_app_bundle_is_found_and_installable(tmp_path):
    exe = make_fiji_app(tmp_path)
    assert imagej.find_imagej(str(tmp_path / 'Fiji.app')) == exe   # the bundle folder
    assert imagej.find_imagej(str(exe)) == exe                       # the launcher itself
    dest = imagej.install_macro(exe.parent)                          # what install-imagej passes
    assert dest == tmp_path / 'Fiji.app' / 'plugins' / imagej.MACRO_NAME


def test_launch_lets_stderr_through(tmp_path, monkeypatch):
    exe = make_imagej_dir(tmp_path, 'ImageJ')
    seen = {}
    monkeypatch.setattr(imagej.subprocess, 'Popen', lambda cmd, **kw: seen.update(kw))
    imagej.launch(tmp_path / 'p', exe)
    assert 'stderr' not in seen and seen['stdin'] is imagej.subprocess.DEVNULL


def test_missing_imagej_exits_cleanly(tmp_path, monkeypatch):
    from test_reader import make_scan
    from brukerview import cli
    scan, _ = make_scan(tmp_path)
    monkeypatch.setattr(imagej, 'find_imagej', lambda explicit=None: None)
    with pytest.raises(SystemExit, match='ImageJ not found'):
        cli.main(['imagej', str(scan)])
