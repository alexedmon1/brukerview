"""Locate Bruker scans and reconstructions on disk."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Union

from .reader import BrukerImage

_NUMERIC = re.compile(r'^\d+$')


def _sort_key(p: Path):
    return (0, int(p.name)) if _NUMERIC.match(p.name) else (1, p.name.lower())


def is_pdata_dir(p: Path) -> bool:
    return (p / 'visu_pars').is_file()


def is_scan_dir(p: Path) -> bool:
    return (p / 'pdata').is_dir() and ((p / 'acqp').is_file() or (p / 'method').is_file())


def is_study_dir(p: Path) -> bool:
    return (p / 'subject').is_file() or any(is_scan_dir(c) for c in p.iterdir() if c.is_dir())


def pdata_dirs(scan_dir: Path) -> List[Path]:
    pd = scan_dir / 'pdata'
    if not pd.is_dir():
        return []
    return sorted((c for c in pd.iterdir() if c.is_dir() and is_pdata_dir(c)), key=_sort_key)


def scan_dirs(study_dir: Path) -> List[Path]:
    return sorted((c for c in study_dir.iterdir() if c.is_dir() and is_scan_dir(c)), key=_sort_key)


def find_study_dirs(root: Path, max_depth: int = 3) -> List[Path]:
    """Study directories under ``root`` (including ``root`` itself)."""
    found = []
    if is_study_dir(root):
        return [root]
    if max_depth <= 0:
        return found
    for child in sorted(root.iterdir(), key=_sort_key):
        if child.is_dir() and child.name != 'pdata':
            found.extend(find_study_dirs(child, max_depth - 1))
    return found


def find_images(path: Union[str, Path], all_procs: bool = False) -> List[BrukerImage]:
    """Every reconstruction reachable from ``path``.

    ``path`` may be a pdata dir, a scan dir, a study dir, or a folder of
    studies. Only the first pdata per scan is returned unless ``all_procs``.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        path = path.parent
    if is_pdata_dir(path):
        return [BrukerImage(path)]
    if is_scan_dir(path):
        procs = pdata_dirs(path)
        return [BrukerImage(p) for p in (procs if all_procs else procs[:1])]
    images: List[BrukerImage] = []
    for study in find_study_dirs(path):
        for scan in scan_dirs(study):
            procs = pdata_dirs(scan)
            for p in (procs if all_procs else procs[:1]):
                try:
                    images.append(BrukerImage(p))
                except Exception as exc:  # unreadable visu_pars: skip, don't abort listing
                    print(f'warning: skipping {p}: {exc}')
    return images


def resolve_image(path: Union[str, Path], proc: Optional[str] = None) -> BrukerImage:
    """A single reconstruction from a pdata dir or scan dir."""
    path = Path(path).expanduser().resolve()
    if path.is_file():
        path = path.parent
    if is_pdata_dir(path):
        return BrukerImage(path)
    if is_scan_dir(path):
        if proc is not None:
            return BrukerImage(path / 'pdata' / str(proc))
        procs = pdata_dirs(path)
        if not procs:
            raise FileNotFoundError(f'{path}: no reconstructed pdata')
        return BrukerImage(procs[0])
    raise ValueError(f'{path} is not a Bruker scan or pdata directory')
