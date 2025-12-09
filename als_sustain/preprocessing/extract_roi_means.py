import nibabel as nib
import numpy as np
from typing import Dict

def compute_roi_means(img_path: str, atlas_img_path: str) -> Dict[str, float]:
    """Compute ROI mean values from a brain volume.

    Returns dict mapping ROI integer (as str) to mean value.
    If you prefer region names, supply an atlas where labels map to names.
    """
    img = nib.load(img_path)
    brain_val = img.get_fdata()

    atlas_img = nib.load(atlas_img_path)
    atlas = atlas_img.get_fdata()

    roi_values = {}
    labels = np.unique(atlas)
    labels = labels[labels != 0]  # skip background

    for label in labels:
        mask = atlas == label
        vals = brain_val[mask]
        if vals.size == 0:
            roi_values[str(int(label))] = float('nan')
        else:
            roi_values[str(int(label))] = float(np.mean(vals))

    return roi_values
