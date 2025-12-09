import json
import os
import tempfile

from als_sustain.pipeline import run_pipeline as rp

def test_run_pipeline_steps_monkeypatched(tmp_path, monkeypatch):
    # Create minimal model descriptor
    base_dir = tmp_path / "base"
    os.makedirs(base_dir / "als_sustain" / "models", exist_ok=True)
    model_id = "test_model"
    desc = {
        "model_id": model_id,
        "model_file": "models/dummy_model.pkl",
        "input_type": "dbm",
        "preprocessing": {
            "requires_dbm": True,
            "requires_roi_extraction": True,
            "requires_dl_feature_extraction": False,
        },
    }
    desc_path = base_dir / "als_sustain" / "models" / f"{model_id}.json"
    with open(desc_path, "w") as f:
        json.dump(desc, f)

    # Prepare a fake T1 file path
    t1_path = tmp_path / "subj_t1.nii.gz"
    t1_path.write_text("fake")

    # Monkeypatch pipeline step functions to avoid heavy external deps
    monkeypatch.setattr(rp, "run_pelican_step", lambda ctx: {**ctx, "dbm_path": str(tmp_path / "fake_dbm.nii.gz")})
    monkeypatch.setattr(rp, "compute_roi_means_step", lambda ctx: {**ctx, "features": {**ctx.get("features", {}), "roi_1": 1.23}})
    monkeypatch.setattr(rp, "extract_dl_features_step", lambda ctx: {**ctx, "features": {**ctx.get("features", {}), "dl_1": 0.5}})
    monkeypatch.setattr(rp, "normalize_step", lambda ctx: {**ctx, "arr": [0.1], "features_used": list(ctx.get("features", {}).keys())})
    monkeypatch.setattr(rp, "predict_step", lambda ctx: {**ctx, "prediction": {"subtype": "X"}})

    row = {"ID": "S01", "Visit": "V1", "Path": str(t1_path)}
    result = rp.run_for_row(row, model_id, str(tmp_path / "workdir"), base_dir=str(base_dir))

    assert result["ID"] == "S01"
    assert result["Visit"] == "V1"
    assert result["model_id"] == model_id
    assert result["prediction"] == {"subtype": "X"}
    assert "roi_1" in result["features_used"]