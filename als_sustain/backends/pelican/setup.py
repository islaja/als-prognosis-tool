# als_sustain/backends/pelican/setup.py

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# -------------------------
# Configuration defaults
# -------------------------

PEL_CAN_VERSION = "1.0"

PELICAN_SIF_URL = "https://example.org/PELICAN_minc_ants_anaconda.sif"
PELICAN_MODELS_ZIP_URL = "https://example.org/PELICAN_Repository.zip"

# -------------------------
# Public data object
# -------------------------

@dataclass(frozen=True)
class PelicanConfig:
    sif_path: Path
    models_dir: Path
    version: str

# -------------------------
# Public functions
# -------------------------

def ensure_pelican_ready(
    *,
    base_dir: Optional[Path] = None,
    force: bool = False,
    interactive: bool = True
) -> PelicanConfig:
    """
    Ensure Pelican container and models exist locally and are valid.
    Safe to call multiple times.
    """
    base_dir = base_dir or _default_base_dir()
    base_dir.mkdir(parents=True, exist_ok=True)

    sif_path = base_dir / "pelican.sif"
    models_dir = base_dir / "models"

    if not force and _is_pelican_ready(sif_path, models_dir):
        return PelicanConfig(sif_path=sif_path, models_dir=models_dir, version=PEL_CAN_VERSION)

    if interactive:
        _print_banner()

    _ensure_container(sif_path, force=force)
    _ensure_models(models_dir, force=force)

    return PelicanConfig(sif_path=sif_path, models_dir=models_dir, version=PEL_CAN_VERSION)


# -------------------------
# Internal helpers
# -------------------------

def _default_base_dir() -> Path:
    env = os.environ.get("ALS_SUSTAIN_PELICAN_DIR")
    if env:
        return Path(env).expanduser()
    # XDG fallback
    xdg_cache = os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    return Path(xdg_cache) / "als_sustain" / "pelican"


def _is_pelican_ready(sif_path: Path, models_dir: Path) -> bool:
    return sif_path.exists() and models_dir.exists()


def _ensure_container(sif_path: Path, *, force: bool) -> None:
    if sif_path.exists() and not force:
        return
    if sif_path.exists():
        sif_path.unlink()
    _download_file(url=PELICAN_SIF_URL, destination=sif_path, label="Pelican container")


def _ensure_models(models_dir: Path, *, force: bool) -> None:
    if models_dir.exists() and not force:
        return
    if models_dir.exists():
        shutil.rmtree(models_dir)

    zip_path = models_dir.with_suffix(".zip")
    _download_file(url=PELICAN_MODELS_ZIP_URL, destination=zip_path, label="Pelican models")
    _extract_zip(zip_path, models_dir)
    zip_path.unlink()


def _download_file(*, url: str, destination: Path, label: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[↓] Downloading {label}")
    subprocess.run(["curl", "-L", url, "-o", str(destination)], check=True)


def _extract_zip(zip_path: Path, target_dir: Path) -> None:
    print("[⛏] Extracting models")
    subprocess.run(["unzip", "-q", str(zip_path), "-d", str(target_dir)], check=True)
    

def _print_banner() -> None:
    print("=" * 60)
    print("Setting up Pelican DBM backend")
    print("=" * 60)
