"""``brukerview`` command line."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .reader import BrukerImage
from .study import find_images, resolve_image, is_pdata_dir, is_scan_dir

COLUMNS = [('scan', 'scan'), ('proc', 'p'), ('protocol', 'protocol'), ('method', 'method'),
           ('matrix', 'matrix'), ('slices', 'slices'), ('extra', 'extra dims'),
           ('voxel_mm', 'voxel (mm)'), ('TE_ms', 'TE ms'), ('TR_ms', 'TR ms')]


def format_table(images: List[BrukerImage], with_path: bool = False) -> str:
    cols = COLUMNS + ([('path', 'path')] if with_path else [])
    rows = [img.summary() for img in images]
    widths = {k: max(len(h), *(len(str(r[k])) for r in rows)) for k, h in cols}
    lines = ['  '.join(h.ljust(widths[k]) for k, h in cols)]
    lines.append('  '.join('-' * widths[k] for k, _ in cols))
    for r in rows:
        lines.append('  '.join(str(r[k]).ljust(widths[k]) for k, _ in cols))
    return '\n'.join(lines)


def _pick(images: List[BrukerImage], what: str) -> BrukerImage:
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


def _select(path: str, proc: Optional[str]) -> BrukerImage:
    p = Path(path).expanduser()
    if p.is_file():
        p = p.parent
    if is_pdata_dir(p) or is_scan_dir(p):
        return resolve_image(p, proc)
    images = [i for i in find_images(p) if i.is_image]
    if not images:
        sys.exit(f'{path}: no Bruker image reconstructions found')
    img = _pick(images, path)
    if proc is not None:
        img = BrukerImage(img.scan_dir / 'pdata' / str(proc))
    return img


# ---- subcommands -------------------------------------------------------------------
def cmd_list(args):
    images = find_images(args.path, all_procs=args.all_procs)
    if not images:
        sys.exit(f'{args.path}: no Bruker scans found')
    if args.csv:
        import csv
        w = csv.DictWriter(sys.stdout, fieldnames=list(images[0].summary().keys()))
        w.writeheader()
        for img in images:
            w.writerow(img.summary())
    else:
        print(format_table(images, with_path=args.paths))


def cmd_show(args):
    img = _select(args.path, args.proc)
    if not img.is_image:
        sys.exit(f'{img.pdata_dir}: not an image (VisuCoreDim={img.dim}); spectroscopy is not supported')
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
    img = _select(args.path, args.proc)
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
    else:
        images = [_select(args.path, args.proc)]
    files = []
    for img in images:
        out = _cache_dir() / f'{img.study_id[:20].replace(" ", "_").replace(":", "")}_{default_name(img)}'
        if not out.exists() or out.stat().st_mtime < img.data_path.stat().st_mtime:
            to_nifti(img, out, scaled=not args.raw)
        files.append(str(out))
    print('fsleyes', *files)
    subprocess.Popen([exe, *files], start_new_session=True)


def cmd_imagej(args):
    from .imagej import launch, find_imagej
    img = _select(args.path, args.proc)
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
                                description='View Bruker ParaVision images straight from pdata/N/2dseq.')
    p.add_argument('--version', action='version', version=__version__)
    sub = p.add_subparsers(dest='command')

    def add_target(sp, proc=True):
        sp.add_argument('path', help='study, scan, or pdata directory')
        if proc:
            sp.add_argument('-p', '--proc', help='pdata number (default: first)')

    sp = sub.add_parser('list', help='table of scans in a study (or folder of studies)')
    sp.add_argument('path')
    sp.add_argument('--all-procs', action='store_true', help='list every pdata, not just the first')
    sp.add_argument('--paths', action='store_true', help='include the pdata path column')
    sp.add_argument('--csv', action='store_true')
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser('show', help='interactive slice viewer (default command)')
    add_target(sp)
    sp.add_argument('--raw', action='store_true', help='do not apply VisuCoreDataSlope/Offs')
    sp.add_argument('--cmap', default='gray')
    sp.add_argument('--range', nargs=2, type=float, metavar=('LO', 'HI'), help='fixed display range')
    sp.add_argument('--slice', type=int, help='initial slice (1-based)')
    sp.add_argument('--frame', type=int, help='initial index of the 4th dimension (1-based)')
    sp.add_argument('--montage', action='store_true', help='also open a montage of all slices')
    sp.add_argument('--save', metavar='PNG', help='write a snapshot instead of opening a window')
    sp.set_defaults(func=cmd_show)

    sp = sub.add_parser('info', help='key parameters of one reconstruction')
    add_target(sp)
    sp.add_argument('--param', nargs='+', metavar='KEY', help='print specific visu_pars/method keys')
    sp.set_defaults(func=cmd_info)

    sp = sub.add_parser('nifti', help='export to NIfTI (needs nibabel)')
    add_target(sp)
    sp.add_argument('-o', '--output', help='output file, or directory with --all')
    sp.add_argument('--all', action='store_true', help='convert every image scan under PATH')
    sp.add_argument('--all-procs', action='store_true')
    sp.add_argument('--raw', action='store_true', help='keep stored integers, no intensity scaling')
    sp.set_defaults(func=cmd_nifti)

    sp = sub.add_parser('fsleyes', help='convert to a cached NIfTI and open in FSLeyes')
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
        # `brukerview PATH` -> list for a study, show for a scan
        p = Path(argv[0]).expanduser()
        argv.insert(0, 'show' if (is_pdata_dir(p) or is_scan_dir(p)) else 'list')
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return
    args.func(args)


if __name__ == '__main__':
    main()
