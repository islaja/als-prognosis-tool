"""
Image utilities for ALS SuStaIn pipeline.

Currently includes:
- Conversion of MINC files (.mnc) to NIfTI (.nii/.nii.gz) using `mnc2nii`.

Notes:
- External dependencies: requires `mnc2nii` executable in PATH.
- Supports optional compression to .nii.gz.
"""

import subprocess
import gzip
import shutil
from pathlib import Path

def minc2nii(input_path:Path, output_path:Path):
    """
    Convert a MINC (.mnc) file to NIfTI (.nii or .nii.gz) format.

    Uses the external `mnc2nii` command-line tool. If `output_path` ends with
    '.gz', the NIfTI file will be compressed automatically.

    Args:
        input_path (str): Path to the input MINC file (.mnc)
        output_path (str): Path to the output NIfTI file (.nii or .nii.gz)

    Raises:
        subprocess.CalledProcessError: If the `mnc2nii` command fails
        FileNotFoundError: If the input file does not exist
        ValueError: If the output path has an invalid extension
    """

    # Validate paths
    input_path_obj = Path(input_path)
    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input MINC file {input_path} does not exist")
    if not input_path_obj.suffix == '.mnc':
        raise ValueError("Input file must have a .mnc extension")
    if not '.nii' in output_path.suffixes:
        raise ValueError("Output file must have a .nii or .nii.gz extension")
   
    # Convert to uncompressed NIfTI
    if '.gz' in output_path.suffixes:
        # remove .gz to get temporary output path
        output_tmp = output_path.with_suffix('')
    else:
        output_tmp = output_path
    subprocess.run(["mnc2nii", "-float", "-nii", input_path, output_tmp], check=True)

    # Compress if requested
    if '.gz' in output_path.suffixes:
        with open(output_tmp, 'rb') as f_in:
            with gzip.open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        Path(output_tmp).unlink()

def nii2minc(input_path:Path, output_path:Path):
    """
    Convert a NIfTI (.nii or .nii.gz) file to MINC (.mnc) format.

    Uses the external `mnc2nii` command-line tool. If `input_path` ends with
    '.gz', the NIfTI file will be decompressed automatically.

    Args:
        input_path (str): Path to the input NIfTI file (.nii or .nii.gz)
        output_path (str): Path to the output MINC file (.mnc)

    Raises:
        subprocess.CalledProcessError: If the `nii2mnc` command fails
        FileNotFoundError: If the input file does not exist
        ValueError: If the output path has an invalid extension
    """

    # Validate paths
    if not input_path.exists():
        raise FileNotFoundError(f"Input NIfTI file {input_path} does not exist")
    if not '.nii' in input_path.suffixes:
        raise ValueError("Input file must have a .nii or .nii.gz extension")
    if output_path.suffix != '.mnc':
        raise ValueError("Output file must have a .mnc extension")

    # Decompress if needed
    if '.gz' in input_path.suffixes:
        with gzip.open(input_path, 'rb') as f_in:
            output_path_ungz = input_path.with_suffix('')
            with open(output_path_ungz, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        input_tmp = output_path_ungz
    else:
        input_tmp = input_path

    # Convert to MINC
    subprocess.run(["nii2mnc", "-float", input_tmp, output_path], check=True)
    # Clean up temporary file if created
    if '.gz' in input_path.suffixes:
        Path(input_tmp).unlink()    