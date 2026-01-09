"""Extract regional DBM values per-atlas for a dataset.

This module provides a near-line-by-line Python translation of the provided
Matlab `extract_regional_DBM.m` logic with a few improvements:
- reads dataset CSV with columns 'Filename' and 'VisitLabel'
- supports Allen / CerebrA / generic atlases using an atlas-info CSV
- computes left (L), right (R) and whole-region (w) means for each ROI
- writes an output CSV into the outputs folder

The function is intentionally conservative: it handles missing files, logs
warnings and writes NaN when a region contains no voxels.
"""
from typing import Optional
import os
import logging
import numpy as np
import pandas as pd
import nibabel as nib

from ..utils import minc2nii

logger = logging.getLogger(__name__)

from ..utils import get_resource_val

def compute_roi(img_path: str,
                         atlas_name: str,
                         atlas_filepath: str,
                         atlas_info_filepath: str,
                         remove_sulci: bool = False,
                         max_prob_sulci: float = 0.5,
                         outputs_path: str = './outputs',
                         base_dir: str = '.') -> str:
    """Compute regional measures for a single image (.mnc or .nii) at `img_path`.

    This function consults a repository-level resources config at
    `als_sustain/config/resources.yaml` to locate the brain mask and CSF
    probability map used for sulci removal (if requested). These resource
    filenames are shared across models and are not provided per-call.

    Parameters
    ----------
    img_path
        Path to the DBM image file (NIfTI or MINC)
    atlas_name
        A short name for the atlas used (used in column names)
    atlas_filepath
        Path to atlas image (NIfTI preferred, .mnc will be converted)
    atlas_info_filepath
        CSV file describing atlas labels (ID, Name or specialized columns)
    remove_sulci
        If True, attempt to remove sulci using the standard repository
        resources config (brain mask & CSF probability map)
    max_prob_sulci
        Threshold for sulci removal if used
    outputs_path
        Directory where the single-image output CSV will be written
    base_dir
        Base directory to join relative paths

    Returns
    -------
    output_csv_path
        Path to the CSV file written (single-row CSV with regional measures)
    """

    if not os.path.exists(img_path):
        raise ValueError("input image path does not exist: " + img_path)

    os.makedirs(outputs_path, exist_ok=True)

    atlas_img_path = (atlas_filepath if os.path.isabs(atlas_filepath)
                      else os.path.join(base_dir, atlas_filepath))
    atlas_img = nib.load(atlas_img_path)
    atlas = atlas_img.get_fdata()

    # Optional: remove sulci using a CSF probability map while keeping ventricles
    if remove_sulci:
        # query the repo-level resources config
        brain_mask_file = get_resource_val('brain_mask', base_dir)
        csf_prob_file = get_resource_val('csf_prob', base_dir)
        ventricles_idx = get_resource_val('ventricles_idx', base_dir) or [119, 1119, 57, 1057, 58, 1058, 59, 1059, 60, 1060, 61, 1061, 141, 1141, 129, 1129, 130, 1130]

        # fallback: try some default filenames inside base_dir if config not provided or path doesn't exist
        def _find_file(base_name: str):
            candidates = [base_name, base_name + '.nii.gz', base_name + '.nii', base_name + '.mnc']
            for p in candidates:
                p_full = p if os.path.isabs(p) else os.path.join(base_dir, p)
                if os.path.exists(p_full):
                    return p_full
            return None

        if not brain_mask_file or not os.path.exists(brain_mask_file):
            brain_mask_file = _find_file('mni_icbm152_t1_tal_nlin_sym_09c_mask')
        if not csf_prob_file or not os.path.exists(csf_prob_file):
            csf_prob_file = _find_file('mni_icbm152_csf_tal_nlin_sym_09c')

        if brain_mask_file is None or csf_prob_file is None:
            logger.warning('Sulci removal requested, but brain mask or CSF probability map not found; skipping sulci removal')
        else:
            bm_img = nib.load(brain_mask_file)
            brain_mask = bm_img.get_fdata().astype(bool)

            csf_img = nib.load(csf_prob_file)
            csf = csf_img.get_fdata()

            not_csf_bin = csf < max_prob_sulci
            ventricles_bin = np.isin(atlas, ventricles_idx)

            not_sulci_bin = (not_csf_bin | ventricles_bin)
            # update brain mask and atlas to exclude sulci voxels
            brain_mask = brain_mask & not_sulci_bin
            atlas = atlas * brain_mask

            # optionally write intermediate mask/atlas files into outputs
            try:
                atlas_out = os.path.join(outputs_path, 'atlas_woSulci.nii.gz')
                mask_out = os.path.join(outputs_path, 'mni_icbm152_t1_tal_nlin_sym_09c_mask_woSulci.nii.gz')
                nib.save(nib.Nifti1Image(atlas.astype(atlas_img.get_fdata().dtype), atlas_img.affine), atlas_out)
                nib.save(nib.Nifti1Image(brain_mask.astype(np.uint8), bm_img.affine), mask_out)
                logger.info(f'Wrote sulci-removed atlas and mask to {outputs_path}')
            except Exception:
                logger.debug('Could not write intermediate sulci-removed files (permissions or nibabel issue)')

    atlas_info = pd.read_csv(atlas_info_filepath)
   
    # Convert .mnc to nii if needed
    if img_path.endswith('.mnc'):
        nii_path = os.path.splitext(img_path)[0] + '.nii.gz'
        try:
            minc2nii(img_path, nii_path)
            img_path = nii_path
        except FileNotFoundError:
            logger.error('mnc2nii not available in PATH; cannot convert .mnc to .nii')
            raise

    subj_img = nib.load(img_path)
    subj_dbm = subj_img.get_fdata()

    # prepare results dict for this image
    results = {}

    # Depending on atlas type, compute different ROI lists
    if 'id_141' in atlas_info.columns and 'acronym' in atlas_info.columns:
        # Allen atlas: id_141 contains ROI ids and acronym holds short names
        for roi_count in range(len(atlas_info['id_141'])):
            roi_id = int(atlas_info['id_141'].iloc[roi_count])
            acronym = str(atlas_info['acronym'].iloc[roi_count])
            mask_L = atlas == roi_id
            mask_R = atlas == (roi_id + 1000)
            mask_W = mask_L | mask_R

            val_L = float(np.mean(subj_dbm[mask_L])) if np.any(mask_L) else np.nan
            val_R = float(np.mean(subj_dbm[mask_R])) if np.any(mask_R) else np.nan
            val_W = float(np.mean(subj_dbm[mask_W])) if np.any(mask_W) else np.nan

            col_L = f'DBM_{atlas_name}_L_{acronym}'
            col_R = f'DBM_{atlas_name}_R_{acronym}'
            col_W = f'DBM_{atlas_name}_w_{acronym}'
            results[col_L] = round(val_L, 3) if not np.isnan(val_L) else np.nan
            results[col_R] = round(val_R, 3) if not np.isnan(val_R) else np.nan
            results[col_W] = round(val_W, 3) if not np.isnan(val_W) else np.nan

    elif 'RHLabel' in atlas_info.columns and 'LHLabel' in atlas_info.columns:
        # CerebrA atlas: RHLabel and LHLabel hold numeric IDs per ROI
        for roi_count in range(len(atlas_info['RHLabel'])):
            R_roi_id = int(atlas_info['RHLabel'].iloc[roi_count])
            L_roi_id = int(atlas_info['LHLabel'].iloc[roi_count])
            label_name = str(atlas_info['LabelName'].iloc[roi_count])
            mask_L = atlas == L_roi_id
            mask_R = atlas == R_roi_id
            mask_W = mask_L | mask_R

            val_L = float(np.mean(subj_dbm[mask_L])) if np.any(mask_L) else np.nan
            val_R = float(np.mean(subj_dbm[mask_R])) if np.any(mask_R) else np.nan
            val_W = float(np.mean(subj_dbm[mask_W])) if np.any(mask_W) else np.nan

            col_L = f'DBM_{atlas_name}_L_{label_name.replace(" ","")}'
            col_R = f'DBM_{atlas_name}_R_{label_name.replace(" ","")}'
            col_W = f'DBM_{atlas_name}_w_{label_name.replace(" ","")}'
            results[col_L] = round(val_L, 3) if not np.isnan(val_L) else np.nan
            results[col_R] = round(val_R, 3) if not np.isnan(val_R) else np.nan
            results[col_W] = round(val_W, 3) if not np.isnan(val_W) else np.nan

    else:
        # Generic atlas: use ID and Name columns
        for roi_count in range(len(atlas_info['ID'])):
            roi_id = int(atlas_info['ID'].iloc[roi_count])
            name = str(atlas_info['Name'].iloc[roi_count]).replace(' ', '')
            mask = atlas == roi_id
            val = float(np.mean(subj_dbm[mask])) if np.any(mask) else np.nan
            col = f'DBM_{atlas_name}_w_{name}'
            results[col] = round(val, 3) if not np.isnan(val) else np.nan
    # Write single-row CSV with computed regional measures
    dataset_filename = os.path.splitext(os.path.basename(img_path))[0]
    out_csv = os.path.join(outputs_path, f"{dataset_filename}_DBM_{atlas_name}.csv")
    out_df = pd.DataFrame([results])
    out_df.to_csv(out_csv, index=False)
    logger.info(f'Wrote regional DBM CSV: {out_csv}')
    return out_csv


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extract regional DBM CSVs from subject CSV and atlas')
    parser.add_argument('img_path', help='Path to a DBM image (.nii/.mnc)')
    parser.add_argument('atlas_name')
    parser.add_argument('atlas_filepath')
    parser.add_argument('atlas_info_csv')
    parser.add_argument('--remove-sulci', action='store_true', help='Apply sulci removal using repository resources')
    parser.add_argument('--outputs', default='./outputs')
    args = parser.parse_args()
    extract_regional_dbm(args.img_path, args.atlas_name, args.atlas_filepath, args.atlas_info_csv,
                         remove_sulci=args.remove_sulci, outputs_path=args.outputs)
