from pathlib import Path
import os
import logging
import re
from typing import Dict
import numpy as np
import pandas as pd
import nibabel as nib
from typing import Optional

logger = logging.getLogger(__name__)


def normalize_roi_name(name: str) -> tuple[str, str]:
    """
    Returns (base_name, side) where side is 'L', 'R', or 'W'
    """
    name = name.strip()
    LEFT_PATTERN  = r'(_l$| left$|^lh_|^l_)'
    RIGHT_PATTERN = r'(_r$| right$|^rh_|^r_)'

    if re.search(LEFT_PATTERN, name, re.IGNORECASE):
        base = re.sub(LEFT_PATTERN, '', name, flags=re.IGNORECASE)
        return base, "l"

    if re.search(RIGHT_PATTERN, name, re.IGNORECASE):
        base = re.sub(RIGHT_PATTERN, '', name, flags=re.IGNORECASE)
        return base, "r"

    return name, "w"


def compute_lr_whole_averages_for_generic_atlas(
    *,
    atlas_info: pd.DataFrame,
    atlas: np.ndarray,
    subj_maps: np.ndarray,
    atlas_name: str,
    b_only_left_right_avr: bool = True,
) -> dict:
    """
    Compute Left, Right, and Whole averages from an atlas
    that only has columns: ID, Name
    """

    results = {}
    roi_map = {}

    for _, row in atlas_info.iterrows():
        roi_id = int(row["ID"])
        base_name, side = normalize_roi_name(str(row["Name"]))
        base_name = base_name.replace(" ", "")
        roi_map.setdefault(base_name, {})[side] = roi_id
        
    # --------------------------------------------------
    # Compute values
    # --------------------------------------------------
   
    for base_name, sides in roi_map.items():
        mask_L = mask_R = mask_W = None
        # --- Whole ---
        if "w" in sides:
            mask_W = (atlas == sides["w"])
        # --- Left / Right ---
        else:
            if "l" in sides:
                mask_L = (atlas == sides["l"])
            if "r" in sides:
                mask_R = (atlas == sides["r"])  

            if mask_L is not None and mask_R is not None:
                mask_W = mask_L | mask_R
            elif mask_L is not None:
                mask_W = mask_L
            else: # mask_R is not None:
                mask_W = mask_R

        # --- Left / Right ---
        if not b_only_left_right_avr:
            val_L = np.mean(subj_maps[mask_L]) if mask_L is not None and np.any(mask_L) else np.nan
            val_R = np.mean(subj_maps[mask_R]) if mask_R is not None and np.any(mask_R) else np.nan
            results[f"dbm_{atlas_name}_l_{base_name}"] = round(val_L, 3) if not np.isnan(val_L) else np.nan
            results[f"dbm_{atlas_name}_r_{base_name}"] = round(val_R, 3) if not np.isnan(val_R) else np.nan

        # --- Whole ---
        val_W = np.mean(subj_maps[mask_W]) if mask_W is not None and np.any(mask_W) else np.nan
        results[f"dbm_{atlas_name}_w_{base_name}"] = round(val_W, 3) if not np.isnan(val_W) else np.nan

    return results

def compute_roi(
        root_dir: Path,
        img_resources: Dict,
        atlas_name: str,
        input_maps_path: Path,
        debug_dir: Optional[Path] = None,
        remove_sulci: bool = True,
        max_prob_sulci: float = 0.2,
        b_only_left_right_avr: bool = True,
        
) -> pd.Series:
    """Compute regional measures for a single image (.nii.gz) at `input_maps_path`.
    """

    # Prepare atlas
    atlas_path =  (root_dir / img_resources["paths"]["atlases"][atlas_name]).resolve()
    atlas_img = nib.load(atlas_path)
    atlas = atlas_img.get_fdata().astype(int)

    atlas_info_path = (root_dir / img_resources["paths"]["atlases_info"][atlas_name]).resolve()
    atlas_info = pd.read_csv(atlas_info_path)

    # Prepare brain mask
    brain_mask_path = (root_dir / img_resources["paths"]["masks"]["brain"]).resolve()
    brain_mask_img = nib.load(brain_mask_path)
    brain_mask = brain_mask_img.get_fdata().astype(bool)

    # In case the atlas and brain mask are not well aligned
    atlas = atlas * brain_mask
 
    # Optional: remove sulci using a CSF probability map while keeping ventricles
    if remove_sulci:
        # query the repo-level resources config
        csf_prob_file = (root_dir / img_resources["paths"]["masks"]["csf_prob"]).resolve()
        csf_img = nib.load(csf_prob_file).get_fdata()

        allen_atlas_path = (root_dir / img_resources["paths"]["atlases"]["allen"]).resolve()
        allen_atlas_img = nib.load(allen_atlas_path)
        allen_atlas = allen_atlas_img.get_fdata()

        not_csf_bin = csf_img < max_prob_sulci
        ventricles_bin = np.isin(allen_atlas, img_resources["indices"]["ventricles"])
        not_sulci_bin = (not_csf_bin | ventricles_bin)

        # update atlas to exclude sulci voxels
        atlas = atlas * not_sulci_bin
        
        # optionally write intermediate mask/atlas files into outputs
        if debug_dir is not None:
            # Save the atlas mask after sulci removal.
            debug_atlas_wo_sulci_path = str((debug_dir / f"{atlas_name}_wo_sulci.nii.gz").resolve())
            atlas_img = nib.Nifti1Image(atlas.astype('int'), atlas_img.affine, atlas_img.header)
            nib.save(atlas_img, debug_atlas_wo_sulci_path)
            logger.info(f'Wrote sulci-removed atlas {debug_dir}')
    
    subj_maps = nib.load(input_maps_path).get_fdata()

    # prepare results dict for this image
    results = {}

    # Depending on atlas type, compute different ROI lists
    if atlas_name == "allen":
        # Allen atlas: id_141 contains ROI ids and acronym holds short names
        for _, row in atlas_info.iterrows():
            roi_id = int(row['id_141'])
            acronym = str(row['acronym']).replace(" ", "")
            mask_L = (atlas == roi_id)
            mask_R = (atlas == (roi_id + 1000))
            mask_W = mask_L | mask_R

            # If we also want the left and right separated
            if not b_only_left_right_avr:
                val_L = np.mean(subj_maps[mask_L]) if np.any(mask_L) else np.nan
                val_R = np.mean(subj_maps[mask_R]) if np.any(mask_R) else np.nan    
                results[f'dbm_{atlas_name}_l_{acronym}'] = round(val_L, 3) if not np.isnan(val_L) else np.nan
                results[f'dbm_{atlas_name}_r_{acronym}'] = round(val_R, 3) if not np.isnan(val_R) else np.nan
            
            val_W = np.mean(subj_maps[mask_W]) if np.any(mask_W) else np.nan
            results[f'dbm_{atlas_name}_w_{acronym}'] = round(val_W, 3) if not np.isnan(val_W) else np.nan

    elif atlas_name == "cerebra": 
        # CerebrA atlas: RH Label and LH Label hold numeric IDs per ROI
        for _, row in atlas_info.iterrows():
            R_roi_id = int(row['RH Label'])
            L_roi_id = int(row['LH Label'])
            label_name = str(row['Label Name']).replace(" ", "")
            mask_L = (atlas == L_roi_id)
            mask_R = (atlas == R_roi_id)
            mask_W = mask_L | mask_R

            # If we also want the left and right separated
            if not b_only_left_right_avr:
                val_L = np.mean(subj_maps[mask_L]) if np.any(mask_L) else np.nan
                val_R = np.mean(subj_maps[mask_R]) if np.any(mask_R) else np.nan               
                results[f'dbm_{atlas_name}_l_{label_name}'] = round(val_L, 3) if not np.isnan(val_L) else np.nan
                results[f'dbm_{atlas_name}_r_{label_name}'] = round(val_R, 3) if not np.isnan(val_R) else np.nan
            
            val_W = np.mean(subj_maps[mask_W]) if np.any(mask_W) else np.nan
            results[f'dbm_{atlas_name}_w_{label_name}'] = round(val_W, 3) if not np.isnan(val_W) else np.nan

    else:
        results = compute_lr_whole_averages_for_generic_atlas(
            atlas_info=atlas_info,
            atlas=atlas,
            subj_maps=subj_maps,
            atlas_name=atlas_name,
            b_only_left_right_avr=b_only_left_right_avr,
        )

    out_df = pd.Series(results)
    # Make columns' name all lower case
    out_df.index = out_df.index.str.lower()
    return out_df

