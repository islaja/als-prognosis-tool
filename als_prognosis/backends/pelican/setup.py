# als_prognosis/backends/pelican/setup.py

from __future__ import annotations

import os
import shutil
import subprocess
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict
import requests
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)

# -------------------------
# Configuration defaults
# -------------------------

PEL_CAN_VERSION = "1.0"

PELICAN_REPOSITORY_ZIP_URL = "https://zenodo.org/records/17168419/files/PELICAN%20Repository.zip?download=1"

# -------------------------
# Public data object
# -------------------------

@dataclass(frozen=True)
class PelicanConfig:
    base_dir: Path
    models_dir: Path
    minc_tool_extra_dir: Path
    env: Dict[str, str]
    version: str

# -------------------------
# Public functions
# -------------------------


def ensure_pelican_ready(
    *,
    base_dir: Optional[Path] = None,
    force: bool = False,
) -> PelicanConfig:
    """
    Orchestrates the Pelican setup process, including pelican repertory download, 
    and environment construction.
    
    Args:
        base_dir (Path, optional): Custom root directory for assets. Defaults to XDG cache.
        force (bool): If True, deletes existing repertory and performs a fresh download.
        
    Returns:
        PelicanConfig: A frozen dataclass containing all verified paths and environment variables.
    """
    base_dir = base_dir or _default_base_dir()
    base_dir.mkdir(parents=True, exist_ok=True)

    models_dir = base_dir / "Models"
    minc_tool_extra_dir = base_dir / "minc-toolkit-extras"

    _ensure_repertory(models_dir, minc_tool_extra_dir, force=force)
    env = _get_neuro_env(minc_tool_extra_dir)

    pelican_cfg = PelicanConfig(
        base_dir=base_dir, 
        models_dir=models_dir, 
        minc_tool_extra_dir=minc_tool_extra_dir, 
        env=env,
        version=PEL_CAN_VERSION
        )
    
    return pelican_cfg


# -------------------------
# Internal helpers
# -------------------------

def _default_base_dir() -> Path:
    return Path("/data/pelican")

def _grant_execution_permissions(directory: Path):
    """
    Recursively sets files to Read-Only and Executable (0o555) to ensure 
    scripts can run while protecting them from accidental modification.
    
    Args:
        directory (Path): The root directory to scan for files needing permission updates.
    """
    for file_path in directory.rglob("*"):
        if file_path.is_file():
            # Bits for: Read + Execute (No Write)
            # 5 = Read (4) + Execute (1)
            # 0o555 = rx for User, Group and Others
            read_exec_mode = (
                stat.S_IRUSR | stat.S_IXUSR |  # User
                stat.S_IRGRP | stat.S_IXGRP |  # Group
                stat.S_IROTH | stat.S_IXOTH    # Others
            )
            
            # Apply the mode
            file_path.chmod(read_exec_mode)


def _ensure_repertory(models_dir: Path, minc_tools_dir: Path, *, force: bool) -> None:
    """
    Validates the presence of required models and tools, triggering 
    download and extraction if they are missing or if a refresh is forced.
    
    Args:
        models_dir (Path): Path where the DBM models should reside.
        minc_tools_dir (Path): Path where supplemental MINC scripts should reside.
        force (bool): If True, overwrites existing local data.
    """
    if models_dir.exists() and minc_tools_dir.exists() and not force:
        return
    if models_dir.exists():
        shutil.rmtree(models_dir)
        shutil.rmtree(minc_tools_dir)

    label = "Pelican_repository"
    zip_path = models_dir.parent / f"{label}.zip"

    _download_file(url=PELICAN_REPOSITORY_ZIP_URL, destination=zip_path, label=label)
    _extract_zip(zip_path, models_dir.parent, label=label)
    zip_path.unlink()

    # Grant permission to executables in models and minc_tools_dir folder
    _grant_execution_permissions(models_dir)
    _grant_execution_permissions(minc_tools_dir)


def _download_file(*, url: str, destination: Path, label: str) -> None:
    """
    Downloads a remote file using a system-level curl command.
    
    Args:
        url (str): The web address of the file to download.
        destination (Path): The local path where the file should be saved.
        label (str): A descriptive name used for console logging.
    """
    logger.info(f"     ↳ 📥 Downloading {label}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # stream=True allows us to iterate over the chunks of the file
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get total file size from headers (fallback to 0 if not provided)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    with open(destination, 'wb') as f:
        with tqdm(
            total=total_size, 
            unit='iB', 
            unit_scale=True, 
            desc=f"📥 {label}",
            leave=True
        ) as pbar:
            for data in response.iter_content(block_size):
                f.write(data)
                pbar.update(len(data))


def _extract_zip(zip_path: Path, target_dir: Path, label: str) -> None:
    """
    Unpacks a ZIP archive into the specified directory using system utilities.
    
    Args:
        zip_path (Path): Path to the source compressed file.
        target_dir (Path): Directory where contents should be extracted.
    """
    logger.info(f"     ↳ 📦 Extracting {label} ...") 
    subprocess.run(["unzip", "-q", str(zip_path), "-d", str(target_dir)], check=True)
    logger.info(f"     ↳ {label} successfully extracted.")
    
def _get_pelican_python_bin() -> str:
    """
    Searches the system for the required 'pelican_env' binary directory, 
    checking Docker environments and standard local Conda paths.
    
    Returns:
        str: The absolute path to the environment's bin directory.
        
    Raises:
        RuntimeError: If the environment cannot be located after searching all fallbacks.
    """
    
    # Check for an Environment Variable override first (High flexibility)
    env_path = os.getenv("PELICAN_PYTHON_BIN")
    if env_path and os.path.exists(env_path):
       return env_path

    # Check for Docker
    docker_bin_path = "/opt/conda/envs/pelican_env/bin"
    if os.path.exists("/.dockerenv"):
        return docker_bin_path

    # Local Development Fallbacks 
    # Look for miniconda3, anaconda3, or .conda in the user's home directory
    home = Path.home()
    possible_conda_roots = [
        home / "miniconda3",
        home / "anaconda3",
        home / "opt/anaconda3",  
        home / ".conda"
    ]

    for root in possible_conda_roots:
        bin_path = root / "envs/pelican_env/bin"
        if bin_path.exists():
            return str(bin_path)

    # Raise if not round
    possible_conda_roots = [str(r / "envs/pelican_env/bin") for r in possible_conda_roots]
    possible_roots = [env_path, docker_bin_path] + possible_conda_roots
    raise RuntimeError(
        f"Pelican environment bin not found. \n"
        f"Searched in: {possible_roots} and found nothing. \n"
    )


def _get_neuro_env(minc_tool_extra_dir: Path):
    """
    Constructs a localized system environment dictionary by injecting MINC 
    toolkit paths and the Pelican Python environment into the system PATH.
    
    Args:
        minc_tool_extra_dir (Path): Path to supplemental MINC helper scripts.
        
    Returns:
        Dict[str, str]: A modified copy of os.environ suitable for subprocess execution.
    """
    env = os.environ.copy()
    
    # Standard MINC/ANTs paths
    minc_path = os.getenv("MINC_TOOLKIT", "/opt/minc/1.9.18")
    env["MINC_TOOLKIT"] = minc_path
    env["ANTSPATH"] = f"{minc_path}/bin"

    # Get pelican python environment
    pelican_python_bin = _get_pelican_python_bin()
    
    # Get the current system path first
    existing_path = env.get("PATH", "")

    # Build your new additions
    additions = [str(minc_tool_extra_dir), pelican_python_bin, f"{minc_path}/bin"]

    # Join them, putting your extras at the VERY front
    env["PATH"] = ":".join(additions) + ":" + existing_path
    
    # Standard library path
    env["LD_LIBRARY_PATH"] = f"{minc_path}/lib:{env.get('LD_LIBRARY_PATH', '')}"
    
    return env
