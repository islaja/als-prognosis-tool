# als_sustain/backends/pelican/run.py

from __future__ import annotations
import os
import subprocess
from pathlib import Path
from typing import Optional, Union, List

from als_sustain.backends.pelican.setup import PelicanConfig, ensure_pelican_ready

# Get the directory where THIS file (executor.py) is located
CURRENT_DIR = Path(__file__).resolve().parent
# Locate the pelican.sh in the same directory
PELICAN_SCRIPT_PATH = CURRENT_DIR / "PELICAN.sh"


def run_pelican(
    subject_id: str,
    subj_visit: Union[str, List[str]],
    t1w_path: Union[Path, List[Path]],
    output_dir: Path,
    pelican_cfg: Optional[PelicanConfig] = None,
    *,
    force_setup: bool = False
) -> list[Path]:
    """
    The primary entry point for the Pelican pipeline. Orchestrates environment setup, 
    CSV manifest generation, and the execution of the DBM computation shell script.
    
    Args:
        subject_id (str): Unique identifier for the subject.
        subj_visit (str | List[str]): The specific visit labels (e.g., 'V1', 'M12').
        t1w_path (Path | List[Path]): Local path(s) to the T1-weighted NIfTI files.
        output_dir (Path): The root directory for all generated outputs and logs.
        pelican_cfg (PelicanConfig, optional): Existing config object. Auto-generated if None.
        force_setup (bool): Forces a fresh re-download/extraction of models if True.
        
    Returns:
        list[Path]: A list of absolute paths to the successfully generated .mnc DBM files.
        
    Raises:
        FileNotFoundError: If the core PELICAN.sh script is missing.
        RuntimeError: If Pelican finishes execution but fails to produce the expected output.
    """
   
    if not PELICAN_SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Could not find pelican.sh at {PELICAN_SCRIPT_PATH}")
    
    t1w_paths = [t1w_path] if isinstance(t1w_path, Path) else t1w_path
    t1w_paths = [Path(p).resolve() for p in t1w_paths]
    subj_visits = [subj_visit] if isinstance(subj_visit, str) else subj_visit

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure the 'models' folder is downloaded/extracted
    if pelican_cfg is None:
        pelican_cfg = ensure_pelican_ready(force=force_setup)

    should_compute_dbm = False
    dbm_file_paths = []
    # Make a simple CSV file Pelican expects
    processing_list_path = output_dir / "subject_visit.csv"
    with processing_list_path.open("w") as processing_list_file:
        # Pelican expects: subject_id, visit, t1_path, [optional: other columns]
        for visit_str, t1w_path in zip(subj_visits, t1w_paths):
            # Write to csv
            processing_list_file.write(f"{subject_id},{visit_str},{t1w_path},,,\n")
            
            # Built the expected DBM path
            dbm_filename = f"{subject_id}_{visit_str}_dbm.mnc"
            dbm_path = (output_dir / subject_id / visit_str / "vbm" / dbm_filename).resolve()
            dbm_file_paths.append(dbm_path)

            # Logic check: if even one file is missing, we need to run the computation
            if not dbm_path.exists():
                should_compute_dbm = True
    
    if should_compute_dbm:
        log_file_path = output_dir / "pelican.log"
        # Prepare to run pelican
        pelican_args = [str(processing_list_path), str(pelican_cfg.models_dir), str(output_dir)]
        command = ["bash", str(PELICAN_SCRIPT_PATH)] + pelican_args
        print(f"Running Pelican on subject {subject_id}...")
        try:
            with open(log_file_path, "w") as log_file:
                print(f"Logging Pelican output to: {log_file_path}")
                process = subprocess.Popen(
                    command, 
                    env=pelican_cfg.env,        
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,            
                    bufsize=1,           
                )
                # Read output and write it to the log output as it happens
                for line in process.stdout: # type: ignore
                    log_file.write(line)
                    log_file.flush() 

                # Wait for the process to actually finish
                process.wait() 
                       
        except subprocess.CalledProcessError as e:
            print("Script failed!")
            print(f"Error log: {e.stderr}")
            print(f"Logging Pelican output can be found in: {log_file_path}")
            raise

    # Make sure each visit DBM file exist.
    for dbm_file_path in dbm_file_paths:
        if not dbm_file_path.exists():
            raise RuntimeError(f"Pelican finished but DBM file not found: {dbm_file_path}")
    
    return dbm_file_paths

