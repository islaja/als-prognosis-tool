"""
Integration tests for ROI extraction against real atlas resources.

These tests validate that:
- ROI extraction runs end-to-end against the real, production atlas/template
  resources and model descriptor
- No NaN values are produced in the resulting ROI features

Note:
- The subject image is synthetic (generated at test time on the same voxel
  grid as the real brain-mask template). compute_roi_all_atlas never reads
  affine/orientation metadata and only computes means over label masks, so
  only the image's shape and finiteness matter for this test; the atlases,
  `config/resources.yaml`, and the model descriptor remain the real,
  production configuration.
- It is therefore closer to an integration test than a pure unit test.
"""

from pathlib import Path
import logging
import nibabel as nib
import numpy as np
import pandas as pd
import yaml
from als_prognosis.preprocessing.roi import compute_roi_all_atlas

logger = logging.getLogger(__name__)

# Project root directory (model registry root)
ROOT_DIR = Path(__file__).resolve().parent.parent

def test_compute_roi_real_data(tmp_path):
    """
    Run ROI extraction against the real atlases with a synthetic subject
    image and verify valid outputs.

    This test:
    - Loads atlas configuration from the real model descriptor
    - Generates a synthetic subject image matching the real template's shape
    - Runs ROI extraction for each configured atlas
    - Asserts that the resulting features are non-empty and contain no NaN values
    """
    # -------------------------------------------------------------------------
    # Arrange
    # -------------------------------------------------------------------------
    metadata = pd.Series({"ID": "P001", "Visit": "V1"})

    # Load real, production resources configuration
    resources_yaml = ROOT_DIR / "config" / "resources.yaml"
    with resources_yaml.open() as f:
        resources = yaml.safe_load(f)

    # Load real, production model descriptor to determine which atlases are used
    model_yaml = ROOT_DIR / "config" / "models" / "CALSNIC_sustain_14_reg_dbm_wscore.yaml"
    with model_yaml.open() as f:
        model_config = yaml.safe_load(f)

    atlas_names = [
        step["atlases"]
        for step in model_config["processing_chain"]
        if step["step"] == "roi_extraction"
    ][0]

    remove_sulci = [
        step["remove_sulci"]
        for step in model_config["processing_chain"]
        if step["step"] == "roi_extraction"
    ][0]

    csf_threshold = [
        step["csf_threshold"]
        for step in model_config["processing_chain"]
        if step["step"] == "roi_extraction"
    ][0]

    # Build a synthetic subject image on the same voxel grid as the real,
    # tracked brain-mask template. compute_roi_all_atlas only computes means
    # over label masks and never reads affine metadata, so the fixture only
    # needs to match the atlas/template shape and contain finite values.
    brain_mask_path = (ROOT_DIR / resources["paths"]["template"]["brain_mask"]).resolve()
    brain_mask_img = nib.load(str(brain_mask_path))

    synthetic_data = np.ones(brain_mask_img.shape, dtype=np.float32)
    synthetic_img = nib.Nifti1Image(synthetic_data, affine=brain_mask_img.affine)

    synthetic_dbm_path = tmp_path / "synthetic_dbm.nii.gz"
    nib.save(synthetic_img, str(synthetic_dbm_path))

    # -------------------------------------------------------------------------
    # Act
    # -------------------------------------------------------------------------
    combined_atlas_roi_vals = compute_roi_all_atlas(
            atlases_name=atlas_names,
            root_dir=ROOT_DIR,
            subject_meta=metadata,
            input_maps_path=synthetic_dbm_path,
            img_resources=resources,
            out_dir=tmp_path,
            remove_sulci=remove_sulci,
            csf_threshold=csf_threshold,
            include_sides=False
            )

    # -------------------------------------------------------------------------
    # Assert
    # -------------------------------------------------------------------------
    assert isinstance(combined_atlas_roi_vals, pd.Series)
    assert len(combined_atlas_roi_vals) > 0, "No ROI features were computed"
    assert combined_atlas_roi_vals.notnull().all(), \
        f"Found {combined_atlas_roi_vals.isnull().sum()} missing values in the ROI series!"
