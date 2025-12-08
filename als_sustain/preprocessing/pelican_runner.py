"""Run PELICAN (PELICAN must be installed inside the container).
This wrapper assumes PELICAN exposes a CLI such as:
  pelican --input <t1.nii.gz> --outdir <outdir>
Adapt flags to the actual PELICAN CLI.
"""
import subprocess
import os
from typing import Tuple

def run_pelican(t1_path: str, outdir: str, pelican_exec: str = "pelican") -> str:
    """Run Pelican on a T1w image and return the path to the produced DBM file.

    Parameters
    ----------
    t1_path: path to T1w NIfTI file
    outdir: path to output directory
    pelican_exec: name or path to pelican binary inside the container

    Returns
    -------
    path to DBM output file (string)
    """
    os.makedirs(outdir, exist_ok=True)
    cmd = [pelican_exec, "--input", t1_path, "--outdir", outdir]

    # add any other required flags for your Pelican installation
    subprocess.check_call(cmd)

    # assume pelican writes dbm.nii.gz inside outdir — adapt as needed
    dbm_path = os.path.join(outdir, "dbm.nii.gz")
    if not os.path.exists(dbm_path):
        raise FileNotFoundError(f"Expected DBM at {dbm_path} (check Pelican output)")
    return dbm_path
