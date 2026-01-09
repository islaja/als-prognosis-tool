
import pandas as pd
import numpy as np
from pathlib import Path
def compute_roi_dummy(maps_nifti_path: Path, atlas_nifti_path: Path) -> pd.Series:
    return pd.Series({"roi_1": 1.23, "roi_2": 4.56, "roi_3": 7.89})


