from pathlib import Path
import pandas as pd
import yaml
import logging
from als_sustain.preprocessing.roi import compute_roi # Example path
from als_sustain.utils.image import minc2nii

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent

def test_compute_roi_real_data():
    # --- Arrange ---
    # Get path relative to the test file
    test_dir = Path(__file__).resolve().parent
    dbm_path = (test_dir / "data" / "DBM_P001_V1_for_test.nii.gz").resolve()
    metadata = pd.Series({"ID": "P001", "Visit": "V1"})
    
    debug_dir = test_dir / "compute_roi_debug"
    debug_dir.mkdir(exist_ok=True)

    # Load image resources yaml 
    resources_yaml = Path("config/resources.yaml")
    with resources_yaml.open() as f:
        resources = yaml.safe_load(f)

    model_yaml = ROOT_DIR / "config" / "models" /"CALSNIC_sustain_14_reg_dbm_wscore.yaml"

    with model_yaml.open() as f:
        config = yaml.safe_load(f)

    
    atlas_names = [s['atlases'] for s in config['processing_chain'] if s['step'] == 'roi_extraction'][0]
    
    all_atlas_roi_vals = []
    for atlas_name in atlas_names:
        roi_vals=pd.Series([])
        # --- Act ---
        results = compute_roi(
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

    # --- Assert ---
    assert isinstance(results, pd.Series)
    assert not results.isnull().values.any() # Ensure no NaNs