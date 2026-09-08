"""``brukerview`` command line."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Optional, Union

from . import __version__
from .nifti import NiftiImage, find_niftis, is_nifti_path
from .reader import BrukerImage
from .study import find_images, resolve_image, is_pdata_dir, is_scan_dir

COLUMNS = [('scan', 'scan'), ('proc', 'p'), ('protocol', 'protocol'), ('method', 'method'),
           ('matrix', 'matrix'), ('slices', 'slices'), ('extra', 'extra dims'),
           ('voxel_mm', 'voxel (mm)'), ('TE_ms', 'TE ms'), ('TR_ms', 'TR ms')]


def format_table(images: List[Any], with_path: bool = False) -> str:
    cols = COLUMNS + ([('path', 'path')] if with_path else [])
    rows = [img.summary() for img in images]
    widths = {k: max(len(h), *(len(str(r[k])) for r in rows)) for k, h in cols}
    lines = ['  '.join(h.ljust(widths[k]) for k, h in cols)]
    lines.append('  '.join('-' * widths[k] for k, _ in cols))
    for r in rows:
        lines.append('  '.join(str(r[k]).ljust(widths[k]) for k, _ in cols))
    return '\n'.join(lines)


def _pick(images: List[Any], what: str) -> Any:
    """Choose one reconstruction from several, interactively if possible."""
    if len(images) == 1:
        return images[0]
    print(format_table(images))
    if not sys.stdin.isatty():
        sys.exit(f'{what}: several scans found; give a scan directory or run interactively')
    while True:
        choice = input('scan (name or number, blank to quit): ').strip()
        if not choice:
            sys.exit(0)
        matches = [i for i in images if i.scan_name == choice or i.scan_name.startswith(choice)]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            print(f'no scan matches {choice!r}')
        else:
            print('ambiguous: ' + ', '.join(m.scan_name for m in matches))


def _target(path: str) -> Path:
    p = Path(path).expanduser()
    if not p.exists():
        sys.exit(f'{path}: no such file or directory')
    return p


def _select(path: str, proc: Optional[str], reorient: bool = True) -> Union[BrukerImage, NiftiImage]:
    p = _target(path)
    if p.is_file() and is_nifti_path(p):
        return NiftiImage(p, reorient=reorient)
    if p.is_file():
        p = p.parent
    if is_pdata_dir(p) or is_scan_dir(p):
        return resolve_image(p, proc)
    images = [i for i in find_images(p) if i.is_image]
    images += find_niftis(p, reorient=reorient)
    if not images:
        sys.exit(f'{path}: no Bruker image reconstructions or NIfTI files found')
    img = _pick(images, path)
    if proc is not None and isinstance(img, BrukerImage):
        img = BrukerImage(img.scan_dir / 'pdata' / str(proc))
    return img


def _reorient(args) -> bool:
    """Subcommands that read pixel data offer --no-reorient; others do not."""
    return not getattr(args, 'no_reorient', False)


# ---- subcommands -------------------------------------------------------------------
def cmd_list(args):
    p = _target(args.path)
    single_nifti = p.is_file() and is_nifti_path(p)
    images = [] if single_nifti else find_images(args.path, all_procs=args.all_procs)
    images += find_niftis(p)
    if not images:
        sys.exit(f'{args.path}: no Bruker scans or NIfTI files found')
    if args.csv:
        import csv
        w = csv.DictWriter(sys.stdout, fieldnames=list(images[0].summary().keys()))
        w.writeheader()
        for img in images:
            w.writerow(img.summary())
    else:
        print(format_table(images, with_path=args.paths))


def cmd_show(args):
    img = _select(args.path, args.proc, _reorient(args))
    if not img.is_image:
        sys.exit(f'{img.pdata_dir}: not an image (dim={img.dim}); spectroscopy is not supported')
    if args.save:
        import matplotlib
        matplotlib.use('Agg')
    from .viewer import SliceViewer
    v = SliceViewer(img, scaled=not args.raw, cmap=args.cmap,
                    vrange=tuple(args.range) if args.range else None)
    if args.slice is not None:
        v._set_index(0, args.slice - 1)
    if args.frame is not None and len(v.idx) > 1:
        v._set_index(1, args.frame - 1)
    if args.save:
        if args.montage:
            v.show_montage()
            v.plt.gcf().savefig(args.save, dpi=150)
        else:
            v.savefig(args.save)
        print(args.save)
        return
    if args.montage:
        v.show_montage()
    v.run()


def cmd_info(args):
    img = _select(args.path, args.proc, _reorient(args))
    if args.param:
        for key in args.param:
            src = img.visu if key in img.visu else img.method_params
            print(f'{key} = {src.get(key)}')
        return
    s = img.summary()
    print(f'{img.pdata_dir}')
    for k in ('protocol', 'method', 'dim', 'matrix', 'slices', 'extra', 'voxel_mm', 'TE_ms', 'TR_ms', 'dtype', 'frames'):
        print(f'  {k:10s} {s[k]}')
    print(f'  {"subject":10s} {img.subject_id}')
    print(f'  {"study":10s} {img.study_id}')
    print(f'  {"scan time":10s} {img.scan_time_s or 0:.0f} s')
    print(f'  {"orient":10s} {img.orientation_label}')
    print(f'  {"groups":10s} {[(g.size, g.name) for g in img.frame_groups]}')
    if img.is_image:
        print(f'  {"shape":10s} {img.shape}  (nx, ny, nz, ...)')
        import numpy as np
        print('  affine (RAS):')
        for row in np.round(img.nifti_affine(), 4):
            print('    ', row)


def cmd_nifti(args):
    from .nifti import to_nifti
    if args.all:
        images = [i for i in find_images(args.path, all_procs=args.all_procs) if i.is_image]
    else:
        images = [_select(args.path, args.proc)]
    out_dir = Path(args.output) if args.output and (len(images) > 1 or Path(args.output).is_dir()) else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    for img in images:
        if not img.is_image:
            print(f'skip {img.pdata_dir}: not an image')
            continue
        out = to_nifti(img, out_dir or args.output, scaled=not args.raw)
        print(out)


def _cache_dir() -> Path:
    d = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'brukerview'
    d.mkdir(parents=True, exist_ok=True)
    return d


def cmd_fsleyes(args):
    from .nifti import to_nifti, default_name
    exe = shutil.which('fsleyes')
    if not exe:
        sys.exit('fsleyes not found on PATH')
    if args.all:
        images = [i for i in find_images(args.path, all_procs=args.all_procs) if i.is_image]
        images += find_niftis(args.path)
    else:
        images = [_select(args.path, args.proc)]
    files = []
    for img in images:
        if isinstance(img, NiftiImage):
            files.append(str(img.path))
            continue
        out = _cache_dir() / f'{img.study_id[:20].replace(" ", "_").replace(":", "")}_{default_name(img)}'
        if not out.exists() or out.stat().st_mtime < img.data_path.stat().st_mtime:
            to_nifti(img, out, scaled=not args.raw)
        files.append(str(out))
    print('fsleyes', *files)
    subprocess.Popen([exe, *files], start_new_session=True)


def cmd_imagej(args):
    from .imagej import launch, find_imagej
    img = _select(args.path, args.proc)
    if isinstance(img, NiftiImage):
        sys.exit(f'{img.path}: the ImageJ macro imports 2dseq only; try `brukerview show` or `fsleyes`')
    exe = find_imagej(args.imagej)
    cmd = launch(img.pdata_dir, exe, raw=args.raw, bc=not args.no_bc)
    print(' '.join(cmd))


def cmd_install_imagej(args):
    from .imagej import install_macro, find_imagej
    exe = find_imagej(args.imagej_dir)
    if exe is None:
        sys.exit('ImageJ not found; pass the ImageJ/Fiji folder')
    dest = install_macro(exe.parent)
    print(f'installed {dest}\nRestart ImageJ: the command appears as Plugins > Import Bruker 2dseq')


# ---- parser --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='brukerview',
                                description='View Bruker ParaVision images straight from '
                                            'pdata/N/2dseq, and NIfTI files the same way.')
    p.add_argument('--version', action='version', version=__version__)
    sub = p.add_subparsers(dest='command')

    def add_target(sp, proc=True):
        sp.add_argument('path', help='study, scan or pdata directory, or a .nii/.nii.gz file')
        if proc:
            sp.add_argument('-p', '--proc', help='pdata number (default: first)')

    def add_reorient(sp):
        sp.add_argument('--no-reorient', action='store_true',
                        help='NIfTI only: keep the file\'s own axis order instead of '
                             'matching Bruker display orientation')

    sp = sub.add_parser('list', help='table of scans in a study (or folder of studies) '
                                     'and any NIfTI files found')
    sp.add_argument('path', help='study, folder of studies, or a folder with NIfTI files')
    sp.add_argument('--all-procs', action='store_true', help='list every pdata, not just the first')
    sp.add_argument('--paths', action='store_true', help='include the pdata path column')
    sp.add_argument('--csv', action='store_true')
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser('show', help='interactive slice viewer (default command)')
    add_target(sp)
    sp.add_argument('--raw', action='store_true', help='do not apply intensity scaling '
                    '(Bruker VisuCoreDataSlope/Offs, NIfTI scl_slope/inter)')
    sp.add_argument('--cmap', default='gray')
    sp.add_argument('--range', nargs=2, type=float, metavar=('LO', 'HI'), help='fixed display range')
    sp.add_argument('--slice', type=int, help='initial slice (1-based)')
    sp.add_argument('--frame', type=int, help='initial index of the 4th dimension (1-based)')
    sp.add_argument('--montage', action='store_true', help='also open a montage of all slices')
    sp.add_argument('--save', metavar='PNG', help='write a snapshot instead of opening a window')
    add_reorient(sp)
    sp.set_defaults(func=cmd_show)

    sp = sub.add_parser('info', help='key parameters of one reconstruction or NIfTI file')
    add_target(sp)
    sp.add_argument('--param', nargs='+', metavar='KEY',
                    help='print specific visu_pars/method keys (NIfTI: header fields)')
    add_reorient(sp)
    sp.set_defaults(func=cmd_info)

    sp = sub.add_parser('nifti', help='export a Bruker scan to NIfTI (needs nibabel)')
    add_target(sp)
    sp.add_argument('-o', '--output', help='output file, or directory with --all')
    sp.add_argument('--all', action='store_true', help='convert every image scan under PATH')
    sp.add_argument('--all-procs', action='store_true')
    sp.add_argument('--raw', action='store_true', help='keep stored integers, no intensity scaling')
    sp.set_defaults(func=cmd_nifti)

    sp = sub.add_parser('fsleyes', help='open in FSLeyes (Bruker scans via a cached NIfTI, '
                                        'NIfTI files directly)')
    add_target(sp)
    sp.add_argument('--all', action='store_true', help='load every image scan under PATH')
    sp.add_argument('--all-procs', action='store_true')
    sp.add_argument('--raw', action='store_true')
    sp.set_defaults(func=cmd_fsleyes)

    sp = sub.add_parser('imagej', help='open in ImageJ/Fiji via the bundled macro')
    add_target(sp)
    sp.add_argument('--imagej', help='ImageJ executable or folder (or set BRUKERVIEW_IMAGEJ)')
    sp.add_argument('--raw', action='store_true', help='skip intensity scaling (keeps 16-bit)')
    sp.add_argument('--no-bc', action='store_true', help='do not open the Brightness/Contrast panel')
    sp.set_defaults(func=cmd_imagej)

    sp = sub.add_parser('install-imagej', help='copy the macro into ImageJ/plugins')
    sp.add_argument('imagej_dir', nargs='?', help='ImageJ or Fiji folder (auto-detected if omitted)')
    sp.set_defaults(func=cmd_install_imagej)
    return p


def main(argv: Optional[List[str]] = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    known = {'list', 'show', 'info', 'nifti', 'fsleyes', 'imagej', 'install-imagej'}
    if argv and argv[0] not in known and not argv[0].startswith('-'):
        # `brukerview PATH` -> list for a study, show for a scan or a NIfTI file
        p = Path(argv[0]).expanduser()
        single = is_pdata_dir(p) or is_scan_dir(p) or (p.is_file() and is_nifti_path(p))
        argv.insert(0, 'show' if single else 'list')
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return
    args.func(args)


if __name__ == '__main__':
    main()
