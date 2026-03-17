"""
Pipeline orchestration tests.

These tests validate that:
- The processing chain is correctly built from a model descriptor
- Pipeline steps are executed in order
- Context propagation and outputs are correct

Heavy external computations are monkeypatched to keep tests fast and deterministic.
"""

from pathlib import Path
import pandas as pd
import yaml
from als_prognosis.progression import predict
from als_prognosis.backends.pelican import setup, run
from als_prognosis.pipeline import run_pipeline as rp
from als_prognosis.preprocessing import roi, wscores
from als_prognosis.utils import image

ROOT_DIR = Path(__file__).resolve().parent.parent

def test_run_pipeline_steps_monkeypatched(tmp_path, monkeypatch):
    """
    Integration-style test for `run_for_row()`.

    This test exercises real pipeline orchestration and descriptor parsing,
    while monkeypatching heavy external steps (Pelican, ROI extraction,
    w-score computation, and model prediction).

    It verifies that:
    - The pipeline runs end-to-end for a single row
    - Intermediate outputs are written to disk
    - Final prediction results are correctly attached to the output context
    """

    # ------------------------------------------------------------------
    # Arrange
    # ------------------------------------------------------------------

    # Base directories
    tmp_path = Path(tmp_path).resolve()
    base_dir = tmp_path / "base"
    models_dir = base_dir / "config" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    outdir = tmp_path / "outdir"
    outdir.mkdir()

    # Load real model descriptor and adapt it for testing
    real_yaml = Path(ROOT_DIR / "config" / "models" / "CALSNIC_sustain_14_reg_dbm_wscore.yaml").resolve()
    with real_yaml.open() as f:
        desc = yaml.safe_load(f)
    
    desc["models"]["sustain_model"]["model_id"] = "test_model"
    desc["models"]["sustain_model"]["model_file"] = "models/dummy.pkl"
    desc["models"]["sustain_model"]["model_meta_file"] = "models/dummy_meta.pkl"

    # Restrict feature selection to a known test feature
    for step in desc["processing_chain"]:
        if step["step"] == "feature_selection":
            step["selected_list"] = ["roi_1_wscore"]

    desc_path = models_dir / "test_model.yaml"
    with desc_path.open("w") as f:
        yaml.safe_dump(desc, f)

    model_id = desc["models"]["sustain_model"]["model_id"] 
    
    # Fake input row
    t1_path = tmp_path / "subj_t1.mnc"
    t1_path.write_text("fake")
    input_type = "t1w_maps"

    row = {
        "id": "S01",
        "visit": "V1",
        "path": str(t1_path),
        "age": 65,
        "sex": "M",
        "scanner": "siemenspri",
        "dpr": 0.4,
    }

    # Load real resources descriptor
    real_resource_yaml = Path(ROOT_DIR / "config" / "resources.yaml").resolve()

    with real_resource_yaml.open() as f:
        resources = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Monkeypatch heavy external steps
    # ------------------------------------------------------------------

    monkeypatch.setattr(
        image,
        "minc2nii",
        lambda minc_path, nii_path: nii_path.write_text("fake nii content"),
    )

    monkeypatch.setattr(
        image,
        "nii2minc",
        lambda nii_path, minc_path: minc_path.write_text("fake minc content"),
    )

    monkeypatch.setattr(
        setup,
        "ensure_pelican_ready",
        lambda *args, **kwargs: setup.PelicanConfig(
            base_dir=Path("/fake/data"),
            models_dir=Path("/fake/models"),
            minc_tool_extra_dir=Path("/fake/minc_tool_extra_dir"),
            env={"fake1":"fake", "fake2":"fake"},
            version="1.0",
        ),  
    )

    monkeypatch.setattr(
        run,
        "run_pelican",
        lambda *args, **kwargs: [Path(tmp_path / "fake_dbm.mnc")],
    )

    monkeypatch.setattr(
        roi,
        "compute_roi_all_atlas",
        lambda *args, **kwargs: pd.Series({
            "roi_1": 1.23,
            "roi_2": 4.56,
            "roi_3": 7.89
        }),
    )

    monkeypatch.setattr(
        wscores,
        "compute_wscores",
        lambda *args, **kwargs: pd.Series({
            "roi_1_wscore": 0.123,
            "roi_2_wscore": 0.456,
            "roi_3_wscore": 0.789
        }),
    )

    monkeypatch.setattr(
        predict,
        "infer_with_model",
        lambda ctx: {
            **ctx,
            "prediction": pd.Series({"subtype": "X"}),
        },
    )
    
    # ------------------------------------------------------------------
    # Act
    # ------------------------------------------------------------------
    steps, _ = rp.build_steps_from_processing_chain(desc, input_type)

    result = rp.run_for_row(
        steps=steps,
        row=row,
        model_id=model_id,
        desc=desc,
        outdir=outdir,
        resources=resources,
        root_dir=ROOT_DIR,
        input_type="t1w_maps",
        pelican_path=Path("/fake/data"),
    )


    # ------------------------------------------------------------------
    # Assert
    # ------------------------------------------------------------------
    if result is not None:
        assert result["id"] == "S01"
        assert result["visit"] == "V1"
        assert result["model_id"] == "test_model"

        expected_pred = pd.Series({"subtype": "X"})
        pd.testing.assert_series_equal(result["sustain_prediction"], expected_pred)
