# als_sustain/backends/pelican/run.py

from __future__ import annotations
import subprocess
from pathlib import Path
from typing import Optional, Union, List

from als_sustain.backends.pelican.setup import PelicanConfig, ensure_pelican_ready


def run_pelican(
    subject_id: str,
    t1w_path: Union[Path, List[Path]],
    subj_visit: Union[str, List[str]],
    output_dir: Path,
    pelican_cfg: Optional[PelicanConfig] = None,
    *,
    force_setup: bool = False
) -> list[Path]:
    """
    Run Pelican container to generate DBM maps from a T1w file.

    Args:
        t1w_path (Path or List[Path]): Path to the T1w NIfTI file(s) of each visit
        subj_visit (Str or List[str]): Subject visit(s) corresponding to the T1w file(s)
        output_dir (Path): Directory where DBM outputs will be stored
        pelican_cfg (PelicanConfig, optional): Configuration returned by ensure_pelican_ready.
            If None, will call ensure_pelican_ready() automatically.
        force_setup (bool): If True, force reinstallation of Pelican container and models

    Returns:
        Path: The main DBM output file (or output directory if multiple)
    """
   

    t1w_paths = [t1w_path] if isinstance(t1w_path, Path) else t1w_path
    t1w_paths = [Path(p).expanduser() for p in t1w_paths]
    subj_visits = [subj_visit] if isinstance(subj_visit, str) else subj_visit

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if pelican_cfg is None:
        pelican_cfg = ensure_pelican_ready(force=force_setup)

    dbm_outputs = []
    # Make a simple CSV file Pelican expects
    participants_csv = output_dir / "participants.csv"
    with participants_csv.open("w") as f:
        # Pelican expects: subject_id, visit, t1_path, [optional: other columns]
        for visit_str, t1w_path in zip(subj_visits, t1w_paths):
            f.write(f"{subject_id},{visit_str},{t1w_path},,,\n")
            dbm_outputs.append(output_dir / f"{subject_id}_V{visit_str.split('_')[1]}_dbm.mnc")
    # Build the Apptainer command
    cmd = [
        "apptainer",
        "exec",
        "--bind", f"{output_dir.parent}:/work",
        str(pelican_cfg.sif_path),
        "/work/PELICAN.sh",
        f"/work/{participants_csv.relative_to(output_dir.parent)}",
        str(pelican_cfg.models_dir),
        f"/work/{output_dir.relative_to(output_dir.parent)}"
    ]

    print(f"Running Pelican on subject {subject_id}")
    subprocess.run(cmd, check=True)

    # Here, assume Pelican produces one main DBM output per subject
    for dbm_output in dbm_outputs:
        if not dbm_output.exists():
            raise RuntimeError(f"Pelican finished but DBM file not found: {dbm_output}")
    
    return dbm_outputs
