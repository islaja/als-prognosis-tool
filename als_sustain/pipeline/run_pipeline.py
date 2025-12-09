"""Top-level pipeline runner.
This code reads a model descriptor JSON and routes inputs through a step/context
pipeline so preprocessing steps are composable and testable. Imports that may
require large optional dependencies (nibabel, sklearn, etc.) are done lazily
inside the step functions to keep import-time lightweight for tests.
"""
import json
import os
import logging
from typing import Dict

logger = logging.getLogger(__name__)

def load_descriptor(model_id: str, base_dir: str) -> Dict:
    path = os.path.join(base_dir, "als_sustain", "models", f"{model_id}.json")
    with open(path, "r") as f:
        desc = json.load(f)
    return desc

# --- Step implementations ----------------------------------------------------
# Each step accepts a `context` dict and returns the (modified) context.
# Context keys: row, model_id, workdir, base_dir, desc, input_path, subject_outdir,
# features, dbm_path, arr, scaler, features_used, prediction

def ensure_subject_outdir_step(context: Dict) -> Dict:
    row = context["row"]
    workdir = context["workdir"]
    pid = row["ID"]
    visit = row["Visit"]
    subject_outdir = os.path.join(workdir, f"{pid}_{visit}")
    os.makedirs(subject_outdir, exist_ok=True)
    context["subject_outdir"] = subject_outdir
    return context

def run_pelican_step(context: Dict) -> Dict:
    """Run PELICAN on a T1 input to produce a DBM path (lazy import)."""
    input_path = context["input_path"]
    subject_outdir = context["subject_outdir"]
    # lazy import to avoid heavy dependencies at module import time
    from als_sustain.preprocessing.pelican_runner import run_pelican
    dbm_path = run_pelican(input_path, subject_outdir)
    context["dbm_path"] = dbm_path
    return context

def compute_roi_means_step(context: Dict) -> Dict:
    """Compute ROI means from a DBM file using the model's atlas (lazy import)."""
    if "dbm_path" not in context:
        return context
    desc = context["desc"]
    base_dir = context["base_dir"]
    preprocessing = desc.get("preprocessing", {})
    atlas = preprocessing.get("atlas_path")
    if atlas is None:
        raise ValueError("Model descriptor requires atlas_path for DBM processing")
    # lazy import
    from als_sustain.preprocessing.extract_roi_means import compute_roi_means
    atlas_path = os.path.join(base_dir, atlas)
    roi_vals = compute_roi_means(context["dbm_path"], atlas_path)
    features = context.get("features", {})
    features.update(roi_vals)
    context["features"] = features
    return context

def extract_dl_features_step(context: Dict) -> Dict:
    """Extract deep features from a T1 if requested by the model descriptor (lazy)."""
    input_path = context["input_path"]
    desc = context["desc"]
    preprocessing = desc.get("preprocessing", {})
    if input_path.endswith(('.nii', '.nii.gz')):
        from als_sustain.preprocessing.dl_feature_extractor import extract_dl_features
        dl_model_path = preprocessing.get("dl_model_path")
        dl_feats = extract_dl_features(input_path, dl_model_path)
        features = context.get("features", {})
        features.update(dl_feats)
        context["features"] = features
    return context

def normalize_step(context: Dict) -> Dict:
    """Normalize features to an array using any provided scaler (lazy import)."""
    desc = context["desc"]
    base_dir = context["base_dir"]
    features = context.get("features", {})

    scaler = None
    scaler_file = desc.get("scaler_file")
    if scaler_file:
        scaler_path = os.path.join(base_dir, scaler_file)
        import pickle
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    # lazy import normalize_features from preprocessing.normalize
    from als_sustain.preprocessing.normalize import normalize_features
    arr, scaler, keys = normalize_features(features, scaler)
    context["arr"] = arr
    context["scaler"] = scaler
    context["features_used"] = keys
    return context

def predict_step(context: Dict) -> Dict:
    """Load model and predict using the normalized array in context (lazy import)."""
    desc = context["desc"]
    base_dir = context["base_dir"]
    model_file = os.path.join(base_dir, desc["model_file"])
    from als_sustain.inference.predict import load_model, predict_with_model
    model = load_model(model_file)
    arr = context.get("arr")
    if arr is None:
        raise ValueError("No feature array available for prediction")
    prediction = predict_with_model(model, arr)
    context["prediction"] = prediction
    return context

# --- Orchestration ----------------------------------------------------------

def build_steps_from_descriptor(desc: Dict):
    steps = []
    steps.append(ensure_subject_outdir_step)
    preprocessing = desc.get("preprocessing", {})

    if preprocessing.get("requires_dbm"):
        steps.append(run_pelican_step)
    if preprocessing.get("requires_roi_extraction"):
        steps.append(compute_roi_means_step)
    if preprocessing.get("requires_dl_feature_extraction"):
        steps.append(extract_dl_features_step)
    if preprocessing.get("requires_wscore_normalization"):
        # placeholder: implement wscore step if available
        logger.debug("Model requests wscore normalization but no step is implemented.")

    steps.append(normalize_step)
    steps.append(predict_step)
    return steps

def run_for_row(row: Dict, model_id: str, workdir: str, base_dir: str = ".") -> Dict:
    """Process a single CSV row. `row` has keys: ID,Visit,Path
    The Path can be either T1 or a features CSV depending on model descriptor.
    """
    desc = load_descriptor(model_id, base_dir)
    input_path = row["Path"].strip()

    context = {
        "row": row,
        "model_id": model_id,
        "workdir": workdir,
        "base_dir": base_dir,
        "desc": desc,
        "input_path": input_path,
        "features": {},
    }

    # If user supplied a CSV of features directly, parse them now
    if input_path.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(input_path)
        cols_lower = [c.lower() for c in df.columns]
        if set(["region", "value"]).issubset(cols_lower):
            df.columns = [c.lower() for c in df.columns]
            context["features"] = {str(r): float(v) for r, v in zip(df["region"], df["value"])}
        else:
            if df.shape[0] == 1:
                context["features"] = {c: float(df.iloc[0][c]) for c in df.columns}
            else:
                context["features"] = {c: float(df.iloc[0][c]) for c in df.columns}

    steps = build_steps_from_descriptor(desc)
    for step in steps:
        context = step(context)

    result = {
        "ID": row["ID"],
        "Visit": row["Visit"],
        "model_id": model_id,
        "prediction": context.get("prediction"),
        "features_used": context.get("features_used"),
    }
    return result

def run_batch(input_csv: str, model_id: str, workdir: str, base_dir: str = "."):
    import pandas as pd
    df = pd.read_csv(input_csv)
    results = []
    for _, row in df.iterrows():
        r = run_for_row(row.to_dict(), model_id, workdir, base_dir)
        results.append(r)
    return results