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
    b_only_left_right_avr: bool = False,
) -> dict:
    """
    Compute Left, Right, and Whole averages from an atlas
    that only has columns: ID, Name
    """

    results = {}

    # --------------------------------------------------
    # Build ROI lookup: base_name -> {L: id, R: id, W: id}
    # --------------------------------------------------
    roi_map = {}

    for _, row in atlas_info.iterrows():
        roi_id = int(row["ID"])
        base_name, side = normalize_roi_name(str(row["Name"]))

        base_name = base_name.replace(" ", "")

        roi_map.setdefault(base_name, {})[side] = roi_id

    # --------------------------------------------------
    # Compute values
    # --------------------------------------------------
    print(roi_map)
   
    for base_name, sides in roi_map.items():
        mask_L = mask_R = None

        if "l" in sides:
            mask_L = atlas == sides["l"]

        if "r" in sides:
            mask_R = atlas == sides["r"]    
        if mask_L is not None and mask_R is not None:
            mask_W = mask_L | mask_R
        elif mask_L is not None:
            mask_W = mask_L
        elif mask_R is not None:
            mask_W = mask_R
        else:
            continue

        # --- Left / Right ---
        if not b_only_left_right_avr:
            if mask_L is not None and np.any(mask_L):
                val_L = float(np.mean(subj_maps[mask_L]))
            else:
                val_L = np.nan

            if mask_R is not None and np.any(mask_R):
                val_R = float(np.mean(subj_maps[mask_R]))
            else:
                val_R = np.nan

            results[f"dbm_{atlas_name}_l_{base_name}"] = (
                round(val_L, 3) if not np.isnan(val_L) else np.nan
            )
            results[f"dbm_{atlas_name}_r_{base_name}"] = (
                round(val_R, 3) if not np.isnan(val_R) else np.nan
            )

        # --- Whole ---
        if np.any(mask_W):
            val_W = float(np.mean(subj_maps[mask_W]))
        else:
            val_W = np.nan

        results[f"dbm_{atlas_name}_w_{base_name}"] = (
            round(val_W, 3) if not np.isnan(val_W) else np.nan
        )

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

        ventricles_idx  = img_resources["indices"]["ventricles"]
        allen_atlas_path = (root_dir / img_resources["paths"]["atlases"]["allen"]).resolve()
        allen_atlas = nib.load(allen_atlas_path).get_fdata()

        not_csf_bin = csf_img < max_prob_sulci
        ventricles_bin = np.isin(allen_atlas, ventricles_idx)

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
        for roi_count in range(len(atlas_info['id_141'])):
            roi_id = int(atlas_info['id_141'].iloc[roi_count])
            acronym = str(atlas_info['acronym'].iloc[roi_count])
            mask_L = atlas == roi_id
            mask_R = atlas == (roi_id + 1000)
            mask_W = mask_L | mask_R

            # If we also want the left and right separated
            if not b_only_left_right_avr:
                val_L = float(np.mean(subj_maps[mask_L])) if np.any(mask_L) else np.nan
                val_R = float(np.mean(subj_maps[mask_R])) if np.any(mask_R) else np.nan

                col_L = f'dbm_{atlas_name}_l_{acronym}'
                col_R = f'dbm_{atlas_name}_r_{acronym}'
            
                results[col_L] = round(val_L, 3) if not np.isnan(val_L) else np.nan
                results[col_R] = round(val_R, 3) if not np.isnan(val_R) else np.nan
            
            val_W = float(np.mean(subj_maps[mask_W])) if np.any(mask_W) else np.nan
            col_W = f'dbm_{atlas_name}_w_{acronym}'
            results[col_W] = round(val_W, 3) if not np.isnan(val_W) else np.nan

    elif atlas_name == "cerebra": 
        # CerebrA atlas: RH Label and LH Label hold numeric IDs per ROI
        for roi_count in range(len(atlas_info['RH Label'])):
            R_roi_id = int(atlas_info['RH Label'].iloc[roi_count])
            L_roi_id = int(atlas_info['LH Label'].iloc[roi_count])
            label_name = str(atlas_info['Label Name'].iloc[roi_count])
            mask_L = atlas == L_roi_id
            mask_R = atlas == R_roi_id
            mask_W = mask_L | mask_R

            # If we also want the left and right separated
            if not b_only_left_right_avr:
                val_L = float(np.mean(subj_maps[mask_L])) if np.any(mask_L) else np.nan
                val_R = float(np.mean(subj_maps[mask_R])) if np.any(mask_R) else np.nan
                col_L = f'dbm_{atlas_name}_l_{label_name.replace(" ","")}'
                col_R = f'dbm_{atlas_name}_r_{label_name.replace(" ","")}'
                results[col_L] = round(val_L, 3) if not np.isnan(val_L) else np.nan
                results[col_R] = round(val_R, 3) if not np.isnan(val_R) else np.nan
            
            val_W = float(np.mean(subj_maps[mask_W])) if np.any(mask_W) else np.nan
            col_W = f'dbm_{atlas_name}_w_{label_name.replace(" ","")}'
            results[col_W] = round(val_W, 3) if not np.isnan(val_W) else np.nan

    else:
        results = compute_lr_whole_averages_for_generic_atlas(
            atlas_info=atlas_info,
            atlas=atlas,
            subj_maps=subj_maps,
            atlas_name=atlas_name,
            b_only_left_right_avr=b_only_left_right_avr,
        )

        """# Generic atlas: use ID and Name columns
        # Look for pairs of left/right ROIs if applicable (those with the same name for starting with "L_" and "R_")
        L_R_pairs = atlas_info['ID'][atlas_info['Name'].str.startswith('L_')].map(
            lambda x: (x, x + 1)).to_dict()
        for roi_count in range(len(atlas_info['ID'])):
            roi_id = int(atlas_info['ID'].iloc[roi_count])
            name = str(atlas_info['Name'].iloc[roi_count]).replace(' ', '')
            mask = atlas == roi_id
            
            val = float(np.mean(subj_maps[mask])) if np.any(mask) else np.nan
            col = f'dbm_{atlas_name}_w_{name}'
            results[col] = round(val, 3) if not np.isnan(val) else np.nan
    """
    out_df = pd.Series(results)
    return out_df

