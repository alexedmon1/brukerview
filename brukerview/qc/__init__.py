"""Acquisition QA/QC for a Bruker ParaVision study (``brukerview qc``).

Reads the study straight from ``pdata/1/2dseq`` + parameter files, measures
SNR / tSNR / ghosting / coverage / predicted distortion / shim quality per
scan, and writes a self-contained HTML report plus TSV tables. Needs the
``qc`` extra (scipy, nibabel); FSL ``bet2`` is used for the brain mask hull
when available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional


def run_qc(study_dir, out_dir, reference: Optional[int] = None, brain_mask: Optional[Path] = None,
           use_bet: bool = True, scans: Optional[Iterable[int]] = None, notes: Optional[Path] = None,
           progress=print) -> Path:
    """Run QC on one study and write ``report.html``, ``qc_metrics.tsv``,
    ``parameters.tsv`` and ``brain_mask.nii.gz`` into ``out_dir``. Returns the report path."""
    import nibabel as nib
    import numpy as np

    from .. import __version__
    from .analyze import run_all
    from .report import build_html, write_params_tsv, write_tsv
    from .scans import inventory, study_info

    study_dir, out_dir = Path(study_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_scans = inventory(study_dir)
    if not all_scans:
        raise SystemExit(f'{study_dir}: no scans found')
    only = set(scans) if scans else None
    out = run_all(all_scans, reference=reference, brain_mask=brain_mask, use_bet=use_bet, only=only,
                  progress=progress)
    ref = out['reference']
    nib.save(nib.Nifti1Image(ref.mask.astype(np.uint8), ref.affine), str(out_dir / 'brain_mask.nii.gz'))
    write_tsv(out_dir / 'qc_metrics.tsv', all_scans, out)
    write_params_tsv(out_dir / 'parameters.tsv', all_scans)
    notes_md = Path(notes).read_text() if notes else None
    progress('writing report')
    html = build_html(study_info(study_dir), all_scans, out, notes_md=notes_md, version=__version__)
    path = out_dir / 'report.html'
    path.write_text(html)
    return path
