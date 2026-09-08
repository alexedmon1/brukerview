"""Launch ImageJ / Fiji with the bundled 2dseq import macro."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

MACRO_NAME = 'Import_Bruker_2dseq.ijm'
MACRO_PATH = Path(__file__).parent / 'macros' / MACRO_NAME

# Only searched under WSL: a dual-boot Linux box can have a Windows partition
# mounted at /mnt/c too, and a Windows .exe is unusable without wslpath.
_WSL_CANDIDATES = [
    '/mnt/c/Program Files/ImageJ/ImageJ.exe',
    '/mnt/c/Program Files/Fiji.app/ImageJ-win64.exe',
    '/mnt/c/Fiji.app/ImageJ-win64.exe',
]

_CANDIDATES = [
    # native Linux
    '~/Fiji.app/ImageJ-linux64',
    '/opt/Fiji.app/ImageJ-linux64',
    '/usr/local/Fiji.app/ImageJ-linux64',
    '~/bin/ImageJ/ImageJ',
    '~/ImageJ/ImageJ',
    '/opt/ImageJ/ImageJ',
    '/usr/local/ImageJ/ImageJ',
    '/usr/share/imagej/ImageJ',
    # macOS
    '/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx',
    '/Applications/ImageJ.app/Contents/MacOS/ImageJ',
]


def is_wsl() -> bool:
    try:
        return 'microsoft' in Path('/proc/version').read_text().lower()
    except OSError:
        return False


def find_imagej(explicit: Optional[str] = None) -> Optional[Path]:
    """Path to an ImageJ/Fiji executable, or None."""
    candidates: List[str] = []
    if explicit:
        candidates.append(explicit)
    if os.environ.get('BRUKERVIEW_IMAGEJ'):
        candidates.append(os.environ['BRUKERVIEW_IMAGEJ'])
    if is_wsl():
        candidates += _WSL_CANDIDATES
    candidates += _CANDIDATES
    for c in candidates:
        p = Path(c).expanduser()
        if p.is_dir():
            for name in ('ImageJ.exe', 'ImageJ-win64.exe', 'ImageJ-linux64', 'ImageJ'):
                if (p / name).exists():
                    return p / name
        elif p.exists():
            return p
    for name in ('fiji', 'imagej', 'ImageJ'):
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def _host_path(p: Path, exe: Path) -> str:
    """Path as the ImageJ process will see it (Windows form when exe is .exe under WSL)."""
    if exe.suffix.lower() == '.exe' and is_wsl():
        return subprocess.check_output(['wslpath', '-w', str(p)], text=True).strip()
    return str(p)


def launch(pdata_dir: Path, exe: Optional[Path] = None, raw: bool = False,
           bc: bool = True) -> List[str]:
    exe = exe or find_imagej()
    if exe is None:
        raise FileNotFoundError(
            'ImageJ not found. Pass --imagej /path/to/ImageJ(.exe) or set BRUKERVIEW_IMAGEJ.')
    arg = _host_path(pdata_dir, exe)
    if raw:
        arg += '|raw'
    if not bc:
        arg += '|nobc'
    cmd = [str(exe), '-macro', _host_path(MACRO_PATH, exe), arg]
    # stdin must not be inherited: a Windows JVM started from WSL blocks on it.
    # The ImageJ.exe launcher also needs its own folder as the working directory.
    subprocess.Popen(cmd, cwd=str(exe.parent), stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)
    return cmd


def install_macro(imagej_dir: Path) -> Path:
    """Copy the macro into ``<ImageJ>/plugins`` so it appears in the Plugins menu."""
    plugins = imagej_dir / 'plugins'
    if not plugins.is_dir():
        raise FileNotFoundError(f'{imagej_dir} has no plugins folder; is it an ImageJ/Fiji install?')
    dest = plugins / MACRO_NAME
    shutil.copyfile(MACRO_PATH, dest)
    return dest
