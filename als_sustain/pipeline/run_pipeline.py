"""Top-level pipeline runner.
This code reads a model descriptor JSON and routes inputs through the right preprocessing steps.
"""
import json
import os
from typing import Dict

from als_sustain.preprocessing.pelican_runner import run_pelican
from als_sustain.preprocessing.dbm_processing import compute_roi_means
from als_sustain.preprocessing.dl_feature_extractor import extract_dl_features
from als_sustain.preprocessing.normalize import normalize_features
from als_sustain.inference.predict import load_model, predict_with_model
from als_sustain.utils.io import read_participant_inputs

def load_descriptor(model_id: str, base_dir: str) -> Dict:
    path = os.path.join(base_dir, "als_sustain", "models", f"{model_id}.json")
    with open(path, "r") as f:
        desc = json.load(f)
    return desc

def run_for_row(row: Dict, model_id: str, workdir: str, base_dir: str = ".") -> Dict:
    """Process a single CSV row. `row` has keys: ParticipantID, ParticipantVisit, Path
    The Path can be either T1 or a features CSV depending on model descriptor.
    Returns result dict with subtype/stage and metadata.
    """
    desc = load_descriptor(model_id, base_dir)

    input_type = desc.get("input_type")
    preprocessing = desc.get("preprocessing", {})

    # create a per-subject temporary workdir
    pid = row["ParticipantID"]
    visit = row["ParticipantVisit"]
    input_path = row["Path"].strip()

    subject_outdir = os.path.join(workdir, f"{pid}_{visit}")
    os.makedirs(subject_outdir, exist_ok=True)

    # 1) If model requires T1 and user provided T1 path
    features = {}

    if preprocessing.get("requires_t1", False) and input_path.endswith(('.nii', '.nii.gz')):
        # run pelican to get dbm
        dbm_path = run_pelican(input_path, subject_outdir)
    elif input_path.endswith('.csv') and input_type.startswith('dbm'):
        # user has direct features CSV for DBM/regional inputs
        # The CSV should be two columns: region,value or header row of regions
        import pandas as pd
        df = pd.read_csv(input_path)
        # try two formats
        cols_lower = [c.lower() for c in df.columns]
        if set(['region','value']).issubset(cols_lower):
            # normalized names -> lower-case mapping
            df.columns = [c.lower() for c in df.columns]
            features = {str(r): float(v) for r, v in zip(df['region'], df['value'])}
        else:
            # assume single-row CSV with region columns
            if df.shape[0] == 1:
                features = {c: float(df.iloc[0][c]) for c in df.columns}
            else:
                raise ValueError('Unsupported CSV format for features')
    else:
        # if there are DL features expected and input is features CSV
        if input_path.endswith('.csv') and input_type.startswith('deep'):
            import pandas as pd
            df = pd.read_csv(input_path)
            if df.shape[0] == 1:
                features = {c: float(df.iloc[0][c]) for c in df.columns}
            else:
                # if many rows, you may pick a row based on visit
                features = {c: float(df.iloc[0][c]) for c in df.columns}

    # If we have a dbm file we need to convert to ROI features
    if 'dbm_path' in locals():
        atlas = preprocessing.get('atlas_path')
        if atlas is None:
            raise ValueError('Model descriptor requires atlas_path for DBM processing')
        roi_vals = compute_roi_means(dbm_path, os.path.join(base_dir, atlas))
        features.update(roi_vals)

    # If model needs dl features and only T1 present
    if preprocessing.get('requires_dl_feature_extraction', False) and input_path.endswith(('.nii', '.nii.gz')):
        dl_model_path = preprocessing.get('dl_model_path')
        dl_feats = extract_dl_features(input_path, dl_model_path)
        features.update(dl_feats)

    # Load model and scaler if any
    model_file = os.path.join(base_dir, desc['model_file'])
    model = load_model(model_file)

    # Normalization
    scaler = None
    if 'scaler_file' in desc:
        scaler_path = os.path.join(base_dir, desc['scaler_file'])
        import pickle
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

    arr, scaler, keys = normalize_features(features, scaler)

    # prediction
    subtype_stage = predict_with_model(model, arr)

    result = {
        'ParticipantID': pid,
        'ParticipantVisit': visit,
        'model_id': model_id,
        'prediction': subtype_stage,
        'features_used': keys
    }
    return result

def run_batch(input_csv: str, model_id: str, workdir: str, base_dir: str = '.'):
    import pandas as pd
    df = pd.read_csv(input_csv)
    results = []
    for _, row in df.iterrows():
        r = run_for_row(row.to_dict(), model_id, workdir, base_dir)
        results.append(r)
    return results
