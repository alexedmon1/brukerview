"""ImageJ discovery and path translation, on WSL and on native Linux/macOS."""
from pathlib import Path

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
