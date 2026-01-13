import pandas as pd
import yaml
from pathlib import Path

from als_sustain.pipeline import run_pipeline as rp
from als_sustain.preprocessing import roi_dummy
from als_sustain.preprocessing import wscores 
from als_sustain.preprocessing import pelican_runner

ROOT_DIR = Path(__file__).resolve().parent.parent

def test_run_pipeline_steps_monkeypatched(tmp_path, monkeypatch):
    """
    Integration-style test for run_for_row().
    Heavy external steps are monkeypatched, but descriptor loading
    and pipeline orchestration are real.
    """

    # ------------------------------------------------------------------
    # Arrange: Setup everything the function needs
    # ------------------------------------------------------------------

    # 1. Setup directories
    base_dir = tmp_path / "base"
    models_dir = base_dir / "config" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    outdir = tmp_path / "outdir"
    outdir.mkdir()

    # 2. Load yaml for real model descriptor, but modify to point to fake model files and selected features
    real_yaml = Path(ROOT_DIR / "config" / "models" / "CALSNIC_sustain_14_reg_dbm_wscore.yaml").resolve()
    with real_yaml.open() as f:
        desc = yaml.safe_load(f)
    
    desc["model_metadata"]["model_id"] = "test_model"
    desc["model_metadata"]["model_file"] = "models/dummy.pkl"
    desc["model_metadata"]["model_meta_file"] = "models/dummy_meta.pkl"

    for step in desc["processing_chain"]:
        if step["step"] == "feature_selection":
            step["selected_list"] = ["roi_1_wscore"]
        if step["step"] == "w_score_normalization":
            step["model_artifact"] = "models/dummy_meta.pkl"

    desc_path = models_dir / "test_model.yaml"
    with desc_path.open("w") as f:
        yaml.safe_dump(desc, f)

    model_id = desc["model_metadata"]["model_id"] 
    
    # 3. fake input row
    t1_path = tmp_path / "subj_t1.nii.gz"
    t1_path.write_text("fake")

    row = {
        "ID": "S01",
        "Visit": "V1",
        "Path": str(t1_path),
    }

    # 4. resources 
    # load yaml for real resources descriptor
    real_resource_yaml = Path(ROOT_DIR / "config" / "resources.yaml").resolve()

    with real_resource_yaml.open() as f:
        resources = yaml.safe_load(f)

    # 5. Set up Monkeypatches
    monkeypatch.setattr(
        pelican_runner,
        "run_pelican",
        lambda t1_path, subject_outdir: str(tmp_path / "fake_dbm.nii.gz"),
    )

    monkeypatch.setattr(
        roi_dummy,
        "compute_roi_dummy",
        lambda maps_nifti_path, atlas_nifti_path: pd.Series({
            "roi_1": 1.23,
            "roi_2": 4.56,
            "roi_3": 7.89
        }),
    )

    monkeypatch.setattr(
        wscores,
        "compute_wscores",
        lambda features, wscore_hc_model_path: pd.Series({
            "roi_1_wscore": 0.123,
            "roi_2_wscore": 0.456,
            "roi_3_wscore": 0.789
        }),
    )

    monkeypatch.setattr(
        rp,
        "predict_step",
        lambda ctx: {
            **ctx,
            "prediction": pd.Series({"subtype": "X"}),
        },
    )

    # ------------------------------------------------------------------
    # Act
    # ------------------------------------------------------------------
    result = rp.run_for_row(
        row=row,
        model_id=model_id,
        desc=desc,
        outdir=outdir,
        resources=resources,
        root_dir=ROOT_DIR,
        input_type="t1w_maps",
    )


    # ------------------------------------------------------------------
    # Assert
    # ------------------------------------------------------------------
    
    expected_csv = outdir / "S01_V1" / "roi_means_all_atlas.csv"

    assert expected_csv.exists(), f"CSV not found at {expected_csv}"

    df_check = pd.read_csv(expected_csv)
    assert "ID" in df_check.columns
    assert "Visit" in df_check.columns
    assert df_check["ID"].iloc[0] == "S01"
    
    assert result["ID"] == "S01"
    assert result["Visit"] == "V1"
    assert result["model_id"] == "test_model"

    expected_pred = pd.Series({"subtype": "X"})
    pd.testing.assert_series_equal(result["sustain_prediction"], expected_pred)
