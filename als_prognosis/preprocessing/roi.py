"""
ROI extraction and atlas-based computations.

This module provides functions to:
- Normalize ROI names across atlases (`normalize_roi_name`)
- Compute left, right, and whole averages from atlas ROI maps
- Compute regional measures for a single subject image using various atlases

Functions are designed to operate on numpy arrays and pandas dataframes for
flexible pipeline integration. Debug outputs can optionally be written for intermediate steps.
"""

from pathlib import Path
import logging
import re
from typing import Dict, List
import numpy as np
import pandas as pd
import nibabel as nib
from typing import Optional

logger = logging.getLogger(__name__)


def normalize_roi_name(name: str) -> tuple[str, str]:
    """
    Standardize ROI nomenclature and extract anatomical laterality.

    Identifies directional prefixes or suffixes (e.g., 'lh_', '_R', ' left') to 
    categorize the ROI and returns a clean base name for easier cross-atlas matching.

    Args:
        name: The raw string identifier for the ROI from the atlas metadata.

    Returns:
        A tuple containing (base_name, side), where side is 'l' (left), 
        'r' (right), or 'w' (whole/bilateral).
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

def compute_roi_averages_for_generic_atlas(
    *,
    atlas_info: pd.DataFrame,
    atlas: np.ndarray,
    subj_maps: np.ndarray,
    atlas_name: str,
    include_sides: bool = True,
    result_prefix: str = "",
) -> dict:
    """
    Compute left, right, and whole ROI averages for a generic labeled atlas.

    Groups ROI components by normalized name and calculates the mean signal from 
    the subject map. Output keys are prefixed to identify the data modality.

    Args:
        atlas_info: ROI metadata containing 'ID' and 'Name' columns.
        atlas: Integer-labeled atlas array.
        subj_maps: Subject data array (e.g., DBM, VBM).
        atlas_name: Name of the atlas for key formatting.
        include_sides: Whether to compute separate hemispheric averages.
        result_prefix: String prefix for all output keys (e.g., 'dbm_').

    Returns:
        Dictionary with keys: '{result_prefix}{atlas_name}_{side}_{ROI}'.
    """

    results = {}
    roi_map = {}

    for _, row in atlas_info.iterrows():
        roi_id = int(row["ID"])
        base_name, side = normalize_roi_name(str(row["Name"]))
        base_name = base_name.replace(" ", "")
        roi_map.setdefault(base_name, {})[side] = roi_id
        
    # Compute values
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
        if include_sides:
            val_L = np.mean(subj_maps[mask_L]) if mask_L is not None and np.any(mask_L) else np.nan
            val_R = np.mean(subj_maps[mask_R]) if mask_R is not None and np.any(mask_R) else np.nan
            results[f"{result_prefix}{atlas_name}_l_{base_name}"] = val_L if not np.isnan(val_L) else np.nan
            results[f"{result_prefix}{atlas_name}_r_{base_name}"] = val_R if not np.isnan(val_R) else np.nan

        # --- Whole ---
        val_W = np.mean(subj_maps[mask_W]) if mask_W is not None and np.any(mask_W) else np.nan
        results[f"{result_prefix}{atlas_name}_w_{base_name}"] = val_W if not np.isnan(val_W) else np.nan

    return results

def compute_roi_all_atlas(atlases_name: List[str], 
                          root_dir: Path, 
                          subject_meta: pd.Series,
                          input_maps_path: Path,
                          img_resources: Dict, 
                          out_dir: Path, 
                          remove_sulci: bool = True,
                          csf_threshold: Optional[float] = 0.2,
                          result_prefix: str = "",
                          debug_dir: Optional[Path] = None,
                          include_sides=False,
                          ) -> pd.Series:
    """
    Main orchestration function for multi-atlas ROI extraction.

    Handles brain masking, optional CSF/sulci removal, and iterates through 
    all requested atlases to produce a consolidated subject Series.

    Args:
        atlases_name: List of atlases to process.
        root_dir: Project root directory.
        subject_meta: Series containing subject metadata.
        input_maps_path: Path to the subject's input NIfTI image.
        img_resources: Resource configuration dictionary.
        out_dir: Directory for output CSV files.
        remove_sulci: Boolean to enable sulcal masking.
        csf_threshold: Threshold for CSF-based sulcal removal.
        result_prefix: Modality prefix for column names (e.g., 'dbm_').
        debug_dir: Optional path to save intermediate mask files.
        include_sides: Whether to include 'l' and 'r' columns.

    Returns:
        A combined Pandas Series of all extracted ROI means.

    Raises:
        ValueError: If remove_sulci is True but csf_threshold is None.
    """

    include_sides_suffix=""
    if include_sides:
        include_sides_suffix = "_sides_included"
    
    # Load brain mask 
    brain_mask_path = (root_dir / img_resources["paths"]["template"]["brain_mask"]).resolve()
    brain_mask = nib.load(brain_mask_path).get_fdata().astype(bool)

    # If remove_sulci, load CSF prob mask and allen_atlas to get ventricles
    if remove_sulci:
        if csf_threshold == None:
            raise ValueError("When remove_sulci is True, csf_threshold should have a value.")
        
        prob_pct = int(csf_threshold * 100)
        sulci_suffix = f"_nosulci-p{prob_pct}"
        
        # Create the mask of non_sulci 
        csf_prob_file = (root_dir / img_resources["paths"]["template"]["csf_prob"]).resolve()
        
        csf_img = nib.load(csf_prob_file).get_fdata()
        
        # Use Allen atlas to retrive the ventricular
        allen_atlas_path = (root_dir / img_resources["paths"]["atlases"]["allen"]).resolve()
        allen_atlas_img = nib.load(allen_atlas_path)
        allen_atlas = allen_atlas_img.get_fdata()

        not_csf_bin = csf_img < csf_threshold
        ventricles_bin = np.isin(allen_atlas, img_resources["allen_ventricles_indices"])
        not_sulci_bin = (not_csf_bin | ventricles_bin)
    else:
        sulci_suffix = "_withsulci"

    # Load the subject 3D maps
    subj_maps = nib.load(input_maps_path).get_fdata(dtype=np.float32)

    all_atlas_roi_vals = []
    combined_atlas_roi_vals = pd.Series(dtype=float)  # Initialize as empty Series
    for atlas_name in atlases_name:
        # Prepare atlas
        atlas_path =  (root_dir / img_resources["paths"]["atlases"][atlas_name]).resolve()
        atlas_img = nib.load(atlas_path)
        atlas = atlas_img.get_fdata().astype(int)

        atlas_info_path = (root_dir / img_resources["paths"]["atlases_info"][atlas_name]).resolve()
        atlas_info = pd.read_csv(atlas_info_path)
        
        # In case the atlas and brain mask are not well aligned
        atlas = atlas * brain_mask
        
        if remove_sulci:
            # update atlas to exclude sulci voxels
            atlas = atlas * not_sulci_bin
            # optionally write intermediate mask/atlas files into outputs
            if debug_dir is not None:
                # Save the atlas mask after sulci removal.
                debug_atlas_wo_sulci_path = str((debug_dir / f"{atlas_name}_wo_sulci.nii.gz").resolve())
                atlas_img = nib.Nifti1Image(atlas.astype('int'), atlas_img.affine, atlas_img.header)
                nib.save(atlas_img, debug_atlas_wo_sulci_path)
                logger.info(f'Wrote sulci-removed atlas {debug_dir}')

        csv_path = out_dir / f"roi_means_{atlas_name}{sulci_suffix}{include_sides_suffix}.csv"
        if not csv_path.exists():
            
            subj_roi_avr_dict = compute_roi_single_subject(
                subj_maps=subj_maps, 
                atlas=atlas,
                atlas_info=atlas_info,
                atlas_name=atlas_name,
                include_sides=include_sides,
                result_prefix=result_prefix,
                )
            subj_roi_avr = pd.Series(subj_roi_avr_dict)
            meta_and_atlas_roi = pd.concat([subject_meta, subj_roi_avr])
            # Round only the numeric part of the series first
            rounded_vals = meta_and_atlas_roi.apply(
                lambda x: round(x, 3) if isinstance(x, (int, float)) else x
            )
            # Now transpose and save
            display_path = f"{csv_path.parent.name}/{csv_path.name}"
            logger.info(f"     ↳ 📊 Saving : {display_path}")
            rounded_vals.to_frame().T.to_csv(csv_path, index=False)
        else:
            meta_and_atlas_roi = pd.read_csv(csv_path).iloc[0]
            # remove subject metadata
            subj_roi_avr = meta_and_atlas_roi.drop(subject_meta.index, errors='ignore')
        
        all_atlas_roi_vals.append(subj_roi_avr)
        
    combined_atlas_roi_vals = pd.concat(all_atlas_roi_vals)
    meta_and_combined_atlas_roi_vals = pd.concat([subject_meta, combined_atlas_roi_vals])
    csv_path = out_dir / f"roi_means_all_atlas.csv"
    logger.debug(f"Saving combined ROI means to {csv_path}")
    # Round only the numeric part of the series first
    rounded_vals = meta_and_combined_atlas_roi_vals.apply(
        lambda x: round(x, 3) if isinstance(x, (int, float)) else x
    )
    # Now transpose and save
    display_path = f"{csv_path.parent.name}/{csv_path.name}"
    logger.info(f"     ↳ 📊 Saving : {display_path}")
    rounded_vals.to_frame().T.to_csv(csv_path, index=False)

    # Make columns' name lower case, except for those that turns out to be the same when lowered.
    lower_counts = combined_atlas_roi_vals.index.str.lower().value_counts()
    duplicates = lower_counts[lower_counts > 1].index
    combined_atlas_roi_vals.index = [
        idx.lower() if idx.lower() not in duplicates else idx 
        for idx in combined_atlas_roi_vals.index
    ]

    del subj_maps
    return combined_atlas_roi_vals


def compute_lr_w_means(
    subj_maps: np.ndarray,
    atlas: np.ndarray,
    roi_id_l: Optional[int],
    roi_id_r: Optional[int],
    atlas_name: str,
    base_name: str,
    include_sides: bool,
    result_prefix: str = "",
) -> Dict[str, float]:
    """
    Calculate hemispheric and bilateral means for a specific ROI.

    Provides core logic for creating masks from labels and computing means 
    with a consistent naming convention.

    Args:
        subj_maps: Subject data array.
        atlas: Atlas label array.
        roi_id_l: Numeric ID for left ROI.
        roi_id_r: Numeric ID for right ROI.
        atlas_name: Name of atlas for key formatting.
        base_name: Cleaned ROI name.
        include_sides: If True, includes individual L/R results.
        result_prefix: Modality prefix for output keys.

    Returns:
        Dictionary of ROI mean values.
    """

    results = {}

    mask_l = atlas == roi_id_l if roi_id_l is not None else None
    mask_r = atlas == roi_id_r if roi_id_r is not None else None

    mask_w = None
    if mask_l is not None and mask_r is not None:
        mask_w = mask_l | mask_r
    elif mask_l is not None:
        mask_w = mask_l
    elif mask_r is not None:
        mask_w = mask_r

    if include_sides:
        if mask_l is not None and np.any(mask_l):
            results[f"{result_prefix}{atlas_name}_l_{base_name}"] = np.mean(subj_maps[mask_l])
        if mask_r is not None and np.any(mask_r):
            results[f"{result_prefix}{atlas_name}_r_{base_name}"] = np.mean(subj_maps[mask_r])

    if mask_w is not None and np.any(mask_w):
        results[f"{result_prefix}{atlas_name}_w_{base_name}"] = np.mean(subj_maps[mask_w])

    return results

def compute_roi_single_subject(
        subj_maps: np.ndarray,
        atlas: np.ndarray,
        atlas_info: pd.DataFrame,
        atlas_name: str,
        include_sides: bool = False,
        result_prefix: str = "",
) -> Dict:
    """
    Route subject data to atlas-specific extraction workers.

    Supports specialized logic for Allen and CerebrA indexing while 
    using a generic path for other standard atlases.

    Args:
        subj_maps: Subject data array.
        atlas: The (potentially masked) atlas array.
        atlas_info: Metadata table for the atlas.
        atlas_name: Key identifying the atlas type.
        include_sides: Whether to include hemispheric averages.
        result_prefix: Modality prefix for output keys.

    Returns:
        Dictionary of all extracted ROI means for the specified atlas.
    """

    results = {}

    # ALLEN ATLAS
    if atlas_name == "allen":
        for _, roi in atlas_info.iterrows():
            base_name = str(roi["acronym"]).replace(" ", "")

            roi_id_l = int(roi["id_141"])
            roi_id_r = roi_id_l + 1000

            results.update(
                compute_lr_w_means(
                    subj_maps=subj_maps,
                    atlas=atlas,
                    roi_id_l=roi_id_l,
                    roi_id_r=roi_id_r,
                    atlas_name=atlas_name,
                    base_name=base_name,
                    include_sides=include_sides,
                    result_prefix=result_prefix,
                )
            )

    # CEREBRA ATLAS
    elif atlas_name == "cerebra":
        for _, roi in atlas_info.iterrows():
            base_name = str(roi["Label Name"]).replace(" ", "")

            roi_id_l = int(roi["LH Label"])
            roi_id_r = int(roi["RH Label"])

            results.update(
                compute_lr_w_means(
                    subj_maps=subj_maps,
                    atlas=atlas,
                    roi_id_l=roi_id_l,
                    roi_id_r=roi_id_r,
                    atlas_name=atlas_name,
                    base_name=base_name,
                    include_sides=include_sides,
                    result_prefix=result_prefix,
                )
            )

    # GENERIC ATLASES
    else:
        results = compute_roi_averages_for_generic_atlas(
            atlas_info=atlas_info,
            atlas=atlas,
            subj_maps=subj_maps,
            atlas_name=atlas_name,
            include_sides=include_sides,
            result_prefix=result_prefix,
            )
    
    return results
    
