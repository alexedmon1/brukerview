# brukerview

Look at Bruker ParaVision MRI images without converting them first.

`pdata/N/2dseq` already holds the reconstructed image, and the `visu_pars`
next to it holds everything needed to display it (matrix, data type, frame
layout, voxel size, intensity scaling, geometry). `brukerview` reads those two
files directly, so viewing a scan is one command instead of a conversion step
or a hand-filled ImageJ raw-import dialog.

`.nii`/`.nii.gz` files work the same way -- same `list`, `show`, `info` and
viewer keys -- so a scan and its converted or processed version can be compared
without switching tools.

Beyond viewing: play any dimension as a movie (or write it to GIF/MP4), export
to NIfTI, open in FSLeyes or ImageJ, and `brukerview qc` writes an image-quality
report for a whole study (SNR, tSNR, ghosting, coverage, distortion, shim) for
comparing protocols.

## Install

```bash
# from a clone (editable, so edits to the macro or code take effect immediately)
uv tool install --editable /path/to/brukerview --with nibabel --with scipy
# or: pip install -e '/path/to/brukerview[nifti,qc]'

# on any machine with Python 3.9+:
pip install 'brukerview[nifti,qc] @ git+https://github.com/alexedmon1/brukerview.git'
```

Requirements: Python 3.9+, numpy, matplotlib (nibabel for anything NIfTI:
viewing `.nii` files, export, FSLeyes). Extras: `nifti` = nibabel; `qc` =
nibabel + scipy, for `brukerview qc`. GIF movies need nothing more (Pillow comes
with matplotlib); MP4 movies need `ffmpeg` on `PATH`. `qc` uses FSL `bet2` for
the brain-mask hull when it is installed (`$FSLDIR` or `PATH`) and falls back to
a simpler mask without it.
Optional, for `imagej` and `install-imagej`: ImageJ 1.47+ or Fiji, installed
separately (it is an application, not a Python package, so pip does not pull
it in). The macro itself runs on any OS with no other dependencies.

Runs on native Linux, WSL and macOS. On Windows, use it inside WSL2: native
Windows is not supported or tested. Under WSL the Windows ImageJ/Fiji is still
found and used. The interactive viewer needs a
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
brukerview nifti /data/mouse-mri/FAC600_.../5 -o t2.nii.gz      # -o ending .nii/.nii.gz: a file
brukerview nifti /data/mouse-mri/FAC600_.../5 -o nifti_out/     # anything else: a folder (created)
brukerview nifti --all /data/mouse-mri/FAC600_... -o nifti_out/

# parameters
brukerview info /data/mouse-mri/FAC600_.../5
brukerview info ... --param PVM_SpatResol VisuCoreOrientation

# movie of a time series, as a window or a file
brukerview show /data/mouse-mri/FAC600_.../60 --play
brukerview show /data/mouse-mri/FAC600_.../60 --movie bold.gif

# image-quality report for a whole study
brukerview qc /data/mouse-mri/FAC600_... -o qc_out/
```

### Viewer keys and movies

Up/down or mouse wheel for slices, left/right for the next
dimension (echo, diffusion direction, time point), `[`/`]` for a third,
right-drag for window/level, `a` auto-contrast, `r` reset, `m` montage of all
slices, `space` play/pause a movie, `d` choose what the movie steps through
(slice, echo, diffusion direction, time), `+`/`-` faster/slower, `h` help,
`q` close. `--save out.png` writes a snapshot without a window.

Movies loop through one dimension while the others stay where they are. The
default is the longest non-slice dimension: time for BOLD, directions for DTI,
echoes for MSME. Images with no other dimension play through their slices.
`--movie-dim` takes `slice`, a frame-group name (`echo`, `diffusion`, `cycle`)
or `time`. Movies work the same for NIfTI files, where the 4th dimension plays.

```bash
brukerview show .../60 --play                         # open and start playing
brukerview show .../60 --play --fps 20 --movie-dim slice
brukerview show .../60 --slice 5 --movie bold.gif     # write a GIF, no window
brukerview show .../13 --slice 6 --movie dti.mp4      # MP4 needs ffmpeg on PATH
```

### NIfTI files

Give any of the commands a `.nii`/`.nii.gz` file, or a folder holding some:

```bash
brukerview t2.nii.gz                     # same viewer, same keys
brukerview list nifti_out/               # scans and NIfTI files in one table
brukerview list /data/study              # Bruker scans plus any .nii.gz beside them
brukerview info fmri.nii --param pixdim descrip     # header fields
brukerview fsleyes t2.nii.gz             # opened directly, no conversion
```

Arrays are reordered so the first axis runs to the patient's left, the second
posterior and the third superior -- how Bruker stores an axial slice package --
so on-screen orientation, left/right handedness and the montage layout match
what `show` does for a 2dseq. `--no-reorient` (on `show` and `info`) keeps the
file's own axis order; `info` prints the resulting axis codes as `orient`.

The 4th dimension is treated like a Bruker frame group, so left/right steps
through volumes and the title shows `t = ... s` when the header carries a
repetition time. `--raw` skips `scl_slope`/`scl_inter` the way it skips
`VisuCoreDataSlope` for a 2dseq. `imagej` stays Bruker-only: its macro imports
2dseq.

## Acquisition QA/QC report

`brukerview qc` measures image quality for every scan of a study and writes a
self-contained HTML report for comparing protocols, for example in a pilot
session where parameters are being tuned.

```bash
brukerview qc /path/to/STUDY_FOLDER -o qc_out/
brukerview qc STUDY -o qc_out/ --notes interpretation.md   # your own notes in the summary
brukerview qc STUDY -o qc_out/ --brain-mask edited_mask.nii.gz
brukerview qc STUDY -o qc_out/ --reference 20 --scans 13 16 60   # mask from E20, analyse 3 scans
brukerview qc STUDY -o qc_out/ --proc 2                             # measure pdata/2 instead of pdata/1
```

The input is a Bruker study folder (the one holding the numbered scans); `qc`
reads `2dseq` and the parameter files directly, so it does not take NIfTI files.

It writes `report.html` (tables, interactive SVG charts with hover values,
image montages, methods), `qc_metrics.tsv` (one row per analysed scan),
`parameters.tsv` (every scan's protocol parameters) and `brain_mask.nii.gz`.

| Scans | Metrics |
|---|---|
| all | role (anatomical / DWI / SE-EPI reference / single-volume EPI / BOLD / field map / MRS / localizer / EPI adjustment), protocol parameters with the differing ones highlighted |
| anatomical | SNR, CSF–tissue CNR, SNR per √min and per nL·√min |
| DTI-EPI | single-image SNR per shell (two-repetition difference method), FA/MD (tensor fit on b ≤ 1000) |
| BOLD | tSNR (detrended), single-image SNR (consecutive-volume difference), tSNR per √TR<sub>vol</sub> and per nL, drift, steady state; slice-thickness sweeps are detected and plotted |
| all EPI | Nyquist/segment ghost %, brain coverage, displacement predicted from the B0 map (Δf·FOV<sub>PE</sub>/`PVM_EpiEffBandwidth`) |
| field maps / PRESS | ΔB0 spread in the brain; water linewidth |

The brain mask comes from the finest 3-D RARE, or the scan given with
`--reference` (FSL `bet2` hull plus intensity refinement; `--no-bet` skips FSL). It is moved into every other scan through
the `visu_pars` geometry, with no registration. Noise comes from repeated
acquisitions where they exist. Otherwise it comes from air around the head or,
in a tight FOV, from air pockets inside it, and only when that region's
mean/SD shows it contains pure noise. A single-volume EPI counts as a topup
reference only if it is spin echo or acquired with reversed phase encoding.
Field maps are read from the reconstruction stored in Hz. See the report's
Methods section.

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
**Image > Show Info**. The macro and the Python reader share their rules:
`VisuCoreDataSlope`/`Offs` may hold one value, one per frame or one per slice
(any other length is refused), slices stored in reverse order
(`disk_reverse_slice_order`) are put back in order, and 8-bit signed data is
supported.

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
(19 x 210) and a study/zip-wrapper folder. Re-verified on native Linux
(ImageJ 1.53t, Pop!_OS) against MSME 160x160x5x32: identical pixels, allowing
for the +32768 offset ImageJ applies to signed 16-bit data (it carries a
matching calibration, so measured values are unchanged).

Scaled imports used to differ from Python by about 1e-4: the macro passed the
slope to *Multiply* as a short string (0.04073596 became 0.04074). Since
October 2026 it is passed at full precision. Re-checked headless (Windows ImageJ
under WSL) on 3-D RARE, field map, BOLD, 3-group DTI and MSME scans: scaled
values match the Python reader to 1e-11 or better (5e-7 for a 32-bit field
map), and raw values to the +32768 offset above.

The macro header documents three ImageJ macro-language pitfalls (globals need
`var`, user-function results must go through a variable, `return`/`=` do not
parse comparisons) that matter if you edit it.

## Data layout

`NiftiImage` exposes the same interface as `BrukerImage` (`load`, `spacing`,
`shape`, `extra_groups`, `summary`, `nifti_affine`), which is why the viewer,
`list` and `info` need no format-specific code.

`BrukerImage.load()` returns `(nx, ny, nz, *extra)` where the extra axes are the
non-slice frame groups from `VisuFGOrderDesc` in file order (echo, diffusion,
movie/cycle...). The NIfTI affine comes from `VisuCoreOrientation` /
`VisuCorePosition` (DICOM-style patient coordinates) and is flipped to RAS.

Time labels (`t = ... s`) for a Bruker repetition axis use scan time /
repetitions, which is the true time per volume (`reader.time_per_repetition`,
also used by `qc`). ParaVision's scan time excludes dummy scans and includes
averages. The TR alone would be off by the number of shots for segmented EPI.

With `disk_reverse_slice_order` the array is flipped so slices run in order,
and the affine is flipped with it, so exported NIfTIs, FSLeyes and `qc` see the
same voxels at the same positions.

`--all` on a directory means everything under it. On a single file it means
that file only.

Spectroscopy (`VisuCoreDim=1`) is listed but not viewable (`qc` reads it
separately, for the water linewidth).

The QC code lives in `brukerview/qc/`: `scans` (inventory, roles, parameters),
`geometry` (world-space resampling, masks), `metrics`, `dwi`, `fieldmap`, `mrs`,
`analyze` (runs them per scan), and `charts`/`figures`/`report` for the HTML.

A scan directory carries its own `visu_pars` (describing the raw `fid`), so a
reconstruction is identified by the `2dseq` beside it, not by `visu_pars` alone.

## Tests

```bash
uv run --extra dev pytest          # pytest + nibabel + scipy; QC tests skip without scipy
BRUKERVIEW_TEST_DATA=/data/mouse-mri uv run --extra dev pytest

# also run the ImageJ macro for real and compare it with the Python reader
# pixel for pixel (slice order, slope rules and precision, 8-bit signed)
BRUKERVIEW_IMAGEJ_TESTS=1 uv run --extra dev pytest tests/test_slice_order_scaling.py
# under WSL with the Windows ImageJ, put the test data on a Windows drive:
BRUKERVIEW_IMAGEJ_TESTS=1 BRUKERVIEW_IMAGEJ_TMP=/mnt/d/tmp uv run --extra dev pytest tests/test_slice_order_scaling.py
```
