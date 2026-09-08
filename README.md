# brukerview

Look at Bruker ParaVision MRI images without converting them first.

`pdata/N/2dseq` already holds the reconstructed image, and the `visu_pars`
next to it holds everything needed to display it (matrix, data type, frame
layout, voxel size, intensity scaling, geometry). `brukerview` reads those two
files directly, so viewing a scan is one command instead of a conversion step
or a hand-filled ImageJ raw-import dialog.

## Install

```bash
# from a clone (editable, so edits to the macro or code take effect immediately)
uv tool install --editable /path/to/brukerview --with nibabel
# or: pip install -e '/path/to/brukerview[nifti]'

# on any machine with Python 3.9+:
pip install 'brukerview[nifti] @ git+https://github.com/alexedmon1/brukerview.git'
```

Requirements: Python 3.9+, numpy, matplotlib (nibabel for NIfTI/FSLeyes).
The ImageJ macro needs only ImageJ 1.47+ or Fiji, on any OS.

Runs on native Linux, WSL and macOS. The interactive viewer needs a
matplotlib GUI backend: on a bare Linux install add one with
`apt install python3-tk` (or `pip install PyQt5`) -- without it `show` says so
instead of opening nothing, and `--save` still works.

## Use

```bash
# what is in a study? (works on a study folder, a folder of studies, or a scan)
brukerview list /data/mouse-mri          # a WSL path like /mnt/e/... works too

# interactive viewer: give a scan folder, a pdata folder, or a study (you get a picker)
brukerview show /data/mouse-mri/FAC600_.../5
brukerview /data/mouse-mri/FAC600_.../5          # same thing

# open in FSLeyes (converts to a cached NIfTI in ~/.cache/brukerview)
brukerview fsleyes /data/mouse-mri/FAC600_.../5
brukerview fsleyes --all /data/mouse-mri/FAC600_...   # every scan as overlays

# open in ImageJ (a local Linux/macOS Fiji or, under WSL, the Windows one)
brukerview imagej /data/mouse-mri/FAC600_.../5

# export
brukerview nifti /data/mouse-mri/FAC600_.../5 -o t2.nii.gz
brukerview nifti --all /data/mouse-mri/FAC600_... -o nifti_out/

# parameters
brukerview info /data/mouse-mri/FAC600_.../5
brukerview info ... --param PVM_SpatResol VisuCoreOrientation
```

Viewer keys: up/down or mouse wheel for slices, left/right for the next
dimension (echo, diffusion direction, time point), `[`/`]` for a third,
right-drag for window/level, `a` auto-contrast, `r` reset, `m` montage of all
slices, `h` help, `q` close. `--save out.png` writes a snapshot without a window.

## ImageJ macro

`brukerview/macros/Import_Bruker_2dseq.ijm` is a plain ImageJ macro with no
dependencies. Install it once:

```bash
brukerview install-imagej                        # auto-detects a local install
brukerview install-imagej ~/bin/ImageJ            # or name the folder
brukerview install-imagej "/mnt/c/Program Files/ImageJ"   # WSL: the Windows install
```

Search order: `--imagej`, `$BRUKERVIEW_IMAGEJ`, `/mnt/c/...` (only under WSL),
then the usual Linux/macOS spots (`~/Fiji.app`, `/opt/Fiji.app`, `~/bin/ImageJ`,
`/opt/ImageJ`, `/usr/share/imagej`, `/Applications/...`), then `PATH`.

Then in ImageJ use **Plugins > Import Bruker 2dseq**, point it at a study,
scan, or `pdata/N` folder, and it opens the image with the right type, matrix,
voxel size and a hyperstack laid out as slices (z) x echoes/diffusion/time (t)
(a third frame group, e.g. repetitions of a DTI, goes to channels).
Intensity scaling to 32-bit is optional (checkbox). The scan parameters land in
**Image > Show Info**.

Display: the range is set from the whole stack's histogram (0.35% saturated),
small windows are zoomed up, and the Brightness/Contrast panel opens alongside
the image. In that panel, **Auto** re-stretches for the current plane, **Reset**
goes back to the full range, and the sliders set min/max by hand. Ctrl+Shift+C
toggles the panel at any time. `brukerview imagej --no-bc` (or the checkbox in
the dialog) keeps it closed. Pointed at a study folder it lists the scans with their
protocol and matrix so you can pick one.

`brukerview imagej SCAN` runs the same macro non-interactively through
`ImageJ -macro`, so a scan opens in ImageJ straight from the shell. Under WSL
the Windows `ImageJ.exe` is used and paths are translated with `wslpath`.
Verified against ImageJ 1.53a on Windows: T2 RARE, MSME (7 slices x 32 echoes),
3-shell DTI (9 x 75), DTI with 2 repetitions (19 x 95 x 2), 3D FISP, fMRI
(19 x 210) and a study/zip-wrapper folder; pixel values match the Python
reader exactly. Re-verified on native Linux (ImageJ 1.53t, Pop!_OS) against
MSME 160x160x5x32: identical pixels, allowing for the +32768 offset ImageJ
applies to signed 16-bit data (it carries a matching calibration, so measured
values are unchanged).

The macro header documents three ImageJ macro-language pitfalls (globals need
`var`, user-function results must go through a variable, `return`/`=` do not
parse comparisons) that matter if you edit it.

## Data layout

`BrukerImage.load()` returns `(nx, ny, nz, *extra)` where the extra axes are the
non-slice frame groups from `VisuFGOrderDesc` in file order (echo, diffusion,
movie/cycle...). The NIfTI affine comes from `VisuCoreOrientation` /
`VisuCorePosition` (DICOM-style patient coordinates) and is flipped to RAS.

Spectroscopy (`VisuCoreDim=1`) is listed but not viewable.

A scan directory carries its own `visu_pars` (describing the raw `fid`), so a
reconstruction is identified by the `2dseq` beside it, not by `visu_pars` alone.

## Tests

```bash
uv run --with pytest --with nibabel pytest
BRUKERVIEW_TEST_DATA=/data/mouse-mri uv run --with pytest --with nibabel pytest
```
