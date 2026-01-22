"""
Dummy Pelican runner for testing purposes.
"""
from pyparsing import Path

def run_pelican_dummy(t1_path: Path, outdir: Path) -> Path:
    """
    Dummy Pelican runner that simulates DBM output.
    """
    dbm_path = outdir / t1_path.name.replace(".nii.gz", "_dummy_dbm.nii.gz")
  
    return dbm_path
