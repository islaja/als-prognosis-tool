"""Top-level pipeline runner.
This code reads a model descriptor JSON and routes inputs through a step/context
pipeline so preprocessing steps are composable and testable. Imports that may
require large optional dependencies (nibabel, sklearn, etc.) are done lazily
inside the step functions to keep import-time lightweight for tests.
"""
import json
import os
import logging
import pandas as pd
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
# features, arr, scaler, features_used, prediction

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
    context["input_path"] = dbm_path
    return context

def compute_roi_means_step(context: Dict) -> Dict:
    """Compute ROI means from a .nii file using the model's atlas (lazy import)."""
    input_path = context["input_path"]
    if input_path.endswith(('.nii', '.nii.gz')):
        desc = context["desc"]
        base_dir = context["base_dir"]
        preprocessing = desc.get("preprocessing", {})
        atlas = preprocessing.get("atlas_path")
        if atlas is None:
            raise ValueError("Model descriptor requires atlas_path for ROI extraction")
        # lazy import
        from als_sustain.preprocessing.extract_roi_means import compute_roi_means
        atlas_path = os.path.join(base_dir, atlas)
        roi_vals = compute_roi_means(context["input_path"], atlas_path)
        context["features"] = pd.Series(roi_vals)
    return context

def extract_dl_features_step(context: Dict) -> Dict:
    """Extract deep features from a brain maps if requested."""
    from als_sustain.preprocessing.dl_feature_extractor import extract_dl_features
    input_path = context["input_path"]
    if input_path.endswith(('.nii', '.nii.gz')):
        desc = context["desc"]
        preprocessing = desc.get("preprocessing", {})
        dl_model_path = preprocessing.get("dl_model_path")
        dl_feats = extract_dl_features(input_path, dl_model_path)
        context["features"] = pd.Series(dl_feats)
    return context

def extract_selected_features(context: Dict) -> Dict:
    """Extract only the features requested by the model descriptor."""

    input_path = context["input_path"]
    data = context.get("features")
    # get information from input csv if no features in context yet
    if data is None:
        raise ValueError("No features available to select from")
        
    desc = context["desc"]     
    preprocessing = desc.get("preprocessing", {})
    feature_selected = preprocessing.get("features_selected", [])
    model_id = context["model_id"]

    has_features = [f in data.index for f in feature_selected]
    if has_features == [True]*len(feature_selected):
        context["features"] = data[feature_selected]
        logger.debug(f"Loaded features {feature_selected} for model {model_id}")
    else:
        raise ValueError(f"Data does not contain all required selected features {feature_selected} for model {model_id}")

    return context

def sign_correction_step(context: Dict) -> Dict:
    """Invert sign of certain/all features (such as in w-scores) so larger == more abnormal.
    Controlled by descriptor keys under 'preprocessing':
      - 'invert_all_features': bool
      - 'invert_specific_features_list': list of feature names
    """
    desc = context["desc"]
    preprocessing = desc.get("preprocessing", {})
    features = context.get("features", pd.Series(dtype=float))

    invert_all = preprocessing.get("invert_all_features", False)
    invert_list = preprocessing.get("invert_specific_features_list", None)

    # Decide which indices to invert
    if invert_all:
        to_invert = features.index.tolist()
    elif invert_list:
        # only invert names present in features
        to_invert = [f for f in invert_list if f in features.index]
    else:
        raise ValueError("No list of features to invert provided.")

    if not to_invert:
        # nothing to do
        context["features"] = features
        return context

    # Invert sign for selected names, preserve NaNs and dtype
    for name in to_invert:
        features.loc[name] = -1 * features.loc[name]
        
    context["features"] = features
    return context

def predict_step(context: Dict) -> Dict:
    # TODO add prediction for survival model
    """Load model and predict using the normalized array in context (lazy import)."""
    from als_sustain.inference.predict import load_model, load_pickle_info, predict_with_model

    desc = context["desc"]
    base_dir = context["base_dir"]
    model_file = os.path.join(base_dir, desc["model_file"])
    model = load_model(model_file)
    meta_file = os.path.join(base_dir, desc["meta_file"])
    samples_sequence, samples_f = load_pickle_info(meta_file)
    data = context.get("features")
    if data is None:
        raise ValueError("No data available for prediction")
    prediction = predict_with_model(model, samples_sequence, samples_f, data)
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
        #steps.append(extract_dl_features_step)
        logger.debug("Model requests feature_extraction but this step isn't completely implemented.")
    if preprocessing.get("requires_wscore_normalization"):
        # placeholder: implement wscore step if available
        logger.debug("Model requests wscore normalization but no step is implemented.")
    if preprocessing.get("requires_feature_selection"):
        steps.append(extract_selected_features)
    if preprocessing.get("requires_features_inversion"):
        steps.append(sign_correction_step)

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
        "features": pd.Series(),
    } 

    if input_path.endswith('.csv'):
        data = pd.read_csv(input_path)
        if data.shape[0] != 1:
            raise ValueError("Input features CSV must have exactly one row")
        # pick first row and convert to Series
        data = data.iloc[0].squeeze() 
        context["features"] = data

    steps = build_steps_from_descriptor(desc)
    for step in steps:
        context = step(context)

    result = {
        "ID": row["ID"],
        "Visit": row["Visit"],
        "model_id": model_id,
        "features_used": context.get("features_used"),
        "prediction": context.get("prediction"),
    }
    return result

def run_batch(input_csv: str, model_id: str, workdir: str, base_dir: str = "."):
    df = pd.read_csv(input_csv)
    results = []
    for _, row in df.iterrows():
        r = run_for_row(row.to_dict(), model_id, workdir, base_dir)
        results.append(r)
    return results