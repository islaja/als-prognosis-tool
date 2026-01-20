"""
Integration tests for ROI extraction on real DBM data.

These tests validate that:
- ROI extraction runs end-to-end on real atlas and image data
- No NaN values are produced in the resulting ROI features

Note:
- This test depends on container-internal resources defined in
  `config/resources.yaml`.
- It is therefore closer to an integration test than a pure unit test.
"""

from pathlib import Path
import logging
import pandas as pd
import yaml
from als_sustain.preprocessing.roi import compute_roi

logger = logging.getLogger(__name__)

# Project root directory (model registry root)
ROOT_DIR = Path(__file__).resolve().parent.parent

def test_compute_roi_real_data():
    """
    Run ROI extraction on a real DBM image and verify valid outputs.

    This test:
    - Loads atlas configuration from the model descriptor
    - Runs ROI extraction for each configured atlas
    - Writes per-atlas and combined ROI CSVs for inspection
    - Asserts that the resulting features contain no NaN values
    """
    # -------------------------------------------------------------------------
    # Arrange
    # -------------------------------------------------------------------------
    test_dir = Path(__file__).resolve().parent
    dbm_path = (test_dir / "data" / "DBM_P001_V1_for_test.nii.gz").resolve()
    metadata = pd.Series({"ID": "P001", "Visit": "V1"})
    
    debug_dir = test_dir / "compute_roi_debug"
    debug_dir.mkdir(exist_ok=True)

    # Load container resources configuration
    resources_yaml = Path("config/resources.yaml")
    with resources_yaml.open() as f:
        resources = yaml.safe_load(f)

    # Load model descriptor to determine which atlases are used
    model_yaml = ROOT_DIR / "config" / "models" /"CALSNIC_sustain_14_reg_dbm_wscore.yaml"
    with model_yaml.open() as f:
        model_config = yaml.safe_load(f)

    atlas_names = [
        step["atlases"]
        for step in model_config["processing_chain"]
        if step["step"] == "roi_extraction"
    ][0]

    # -------------------------------------------------------------------------
    # Act
    # ------------------------------------------------------------------------- 
    all_atlas_roi_vals = []
    for atlas_name in atlas_names:
        roi_vals = compute_roi(
            root_dir=ROOT_DIR,
            img_resources=resources,
            atlas_name=atlas_name,
            input_maps_path=dbm_path,
            debug_dir=debug_dir,
            )
        
        indiv_atlas_vals_to_save = pd.concat([metadata, roi_vals])
        csv_path = test_dir / f"roi_means_{atlas_name}.csv"
        indiv_atlas_vals_to_save.to_frame().T.to_csv(csv_path, index=False)
        all_atlas_roi_vals.append(roi_vals)
    
    combined_atlas_roi_vals = pd.concat(all_atlas_roi_vals)
    all_atlas_vals_to_save = pd.concat([metadata, combined_atlas_roi_vals])
    csv_path = test_dir / f"roi_means_all_atlas.csv"
    logger.debug(f"Saving combined ROI means to {csv_path}")
    all_atlas_vals_to_save.to_frame().T.to_csv(csv_path, index=False)

    # -------------------------------------------------------------------------
    # Assert
    # -------------------------------------------------------------------------
    assert isinstance(roi_vals, pd.Series)
    assert not roi_vals.isnull().values.any() 