# als_prognosis/backends/pelican/run.py

from __future__ import annotations
import subprocess
from pathlib import Path
from typing import Optional, Union, List
import shutil
from als_prognosis.backends.pelican.setup import PelicanConfig, ensure_pelican_ready
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

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

    log_file_path = output_dir / "pelican.log"

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
            subj_path = (output_dir / subject_id).resolve()
            dbm_path = (subj_path / visit_str / "vbm" / dbm_filename).resolve()
            dbm_file_paths.append(dbm_path)

            # Logic check: if even one file is missing, we need to run the computation
            if not dbm_path.exists():
                should_compute_dbm = True
                # Remove this folder if already exist but no DBM found. 
                # This is probably because pelican run before but couldn't finish. 
                if subj_path.exists() and subj_path.is_dir():
                    shutil.rmtree(subj_path)
                    log_file_path.unlink(missing_ok=True)
    
    if should_compute_dbm:
        APPROXIMATED_NB_LINES = 6000  
        update_interval = APPROXIMATED_NB_LINES // 100  # Update every 1%
        line_count = 0
        current_pbar_val = 0

        
        display_path = f"{log_file_path.parent.name}/{log_file_path.name}"
        # Prepare to run pelican
        pelican_args = [str(processing_list_path), str(pelican_cfg.models_dir), str(output_dir)]
        command = ["bash", str(PELICAN_SCRIPT_PATH)] + pelican_args
        try:
            with open(log_file_path, "w") as log_file:
                logger.info("     Running Pelican analysis")
                logger.info("     ⚠️ Estimated processing time is 45 to 60 minutes per subject. Please do not interrupt.")
                logger.info(f"     📄 Log is being saved to: {display_path}.")
                pbar = tqdm(total=100, unit="%", desc="     ⏳ Pelican Progress", leave=True)
   
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
                    line_count += 1
                    
                    # Update terminal every ~500 lines
                    if line_count % update_interval == 0:
                        if current_pbar_val < 99: # Cap it at 99% until the process is truly done
                            pbar.update(1)
                            current_pbar_val += 1
                    
                # Wait for the process to actually finish
                process.wait() 
            
            # Check the exit status manually
            if process.returncode != 0:
                logger.error(f"     ❌ Pelican failed (Exit Code: {process.returncode})")
                logger.error(f"     🔍 Check the full trace at: {log_file_path}")    
                pbar.close() 
                raise RuntimeError(f"Pelican execution failed for {subject_id}")
            
            if current_pbar_val < 100:
                pbar.update(100 - current_pbar_val) # Jump to 100%
            pbar.close()
            
            logger.info(f"     ✅ Pelican finished successfully")        
            
        except Exception as e:
            logger.error(f"     🚨 Unexpected error launching Pelican: {str(e)}")
            raise

    # Make sure each visit DBM file exist.
    for dbm_file_path in dbm_file_paths:
        if not dbm_file_path.exists():
            raise RuntimeError(f"   Pelican finished but DBM file not found: {dbm_file_path}")
    
    return dbm_file_paths

