"""Top-level pipeline runner.
This code reads a model descriptor JSON and routes inputs through a step/context
pipeline so preprocessing steps are composable and testable. Imports that may
require large optional dependencies (nibabel, sklearn, etc.) are done lazily
inside the step functions to keep import-time lightweight for tests.
"""
import yaml
import logging
import pandas as pd
from typing import Dict
from pathlib import Path

logger = logging.getLogger(__name__)


def load_descriptor(*, model_id: str, base_dir: Path) -> Dict:
    """
    Load a model descriptor YAML file.

    Parameters
    ----------
    model_id : str
        Model identifier (filename without .yaml)
    base_dir : Path
        Project root directory (container-safe)

    Returns
    -------
    Dict
        Parsed model descriptor
    """
    path = base_dir / "config" / "models" / f"{model_id}.yaml"
    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(f"Model descriptor not found: {path}")

    with path.open() as f:
        desc = yaml.safe_load(f)

    if not isinstance(desc, dict):
        raise ValueError(f"Invalid model descriptor format: {path}")

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
    subject_outdir = Path(workdir) / f"{pid}_{visit}".replace(" ", "_")
    subject_outdir.mkdir(parents=True, exist_ok=True)

    context["subject_outdir"] = subject_outdir
    return context

def predict_step(context: Dict) -> Dict:
    # TODO add prediction for survival model
    """Load model and predict using the normalized array in context (lazy import)."""
    from als_sustain.inference.predict import load_model, load_pickle_info, predict_with_model
    
    desc = context["desc"]
    # Make sure the current input type is accepted by the model
    expected_input = desc.get("model_metadata", {}).get("input_accepted")
    current_type = context.get("current_type")
    if current_type not in expected_input:
        raise ValueError(
            f"Model expects input type(s) {expected_input}, "
            f"but current type is '{current_type}'"
        )
    
    base_dir = Path(context["base_dir"])
    model_file_rel = desc.get("model_metadata", {}).get("model_file")
    model_meta_rel = desc.get("model_metadata", {}).get("model_meta_file")
    
    if not model_file_rel or not model_meta_rel:
        raise ValueError("Model descriptor missing 'model_file' or 'model_meta_file'")

    model_file = base_dir / Path(model_file_rel)
    model_file = model_file.resolve()
    model = load_model(model_file)
    
    meta_file = base_dir / Path(model_meta_rel)
    meta_file = meta_file.resolve()
    samples_sequence, samples_f = load_pickle_info(meta_file)

    data = context.get("features")
    if data is None:
        raise ValueError("No data available for prediction")
    prediction = predict_with_model(model, samples_sequence, samples_f, data)
    context["prediction"] = prediction.to_dict()
    return context

# --- Orchestration ----------------------------------------------------------

def build_steps_from_processing_chain(desc: Dict, input_type: str):
    chain = desc.get("processing_chain", [])
    if not chain:
        raise ValueError("Descriptor has no 'processing_chain' to build steps from")

    start_idx = next((i for i, s in enumerate(chain) if input_type in s.get("input_accepted", [])), None)
    if start_idx is None:
        raise ValueError(f"Model does not accept input_type '{input_type}'")

    steps = []
    for step_cfg in chain[start_idx:]:
        name = step_cfg.get("step")
        if name == "dbm_generation":
            steps.append(make_run_pelican_step(step_cfg))
        elif name == "roi_extraction":
            steps.append(make_roi_extraction_step(step_cfg))
        elif name == "w_score_normalization":
            steps.append(make_wscore_step(step_cfg))
        elif name == "feature_selection":
            steps.append(make_feature_selection_step(step_cfg))
        elif name == "feature_inversion":
            steps.append(make_feature_inversion_step(step_cfg))
        else:
            raise ValueError(f"Unknown processing step in descriptor: {name}")
    return steps

def make_run_pelican_step(dfg: Dict):
    """Run PELICAN to produce DBM image from T1 input."""
    input_accepted = dfg.get("input_accepted", [])
    output_type = dfg.get("output_type")
    step_name = dfg.get("step")

    def step(context: Dict):
        current_type = context.get("current_type")
        if current_type not in input_accepted:
            raise ValueError(
                f"Step {step_name} cannot accept input type '{current_type}'. "
                f"Accepted: {input_accepted}"
            )
        t1_path = context["input_path"]
        from als_sustain.preprocessing.pelican_runner import run_pelican
        dbm_path = run_pelican(t1_path, context["subject_outdir"])
        context["input_path"] = dbm_path
        context["current_type"] = output_type
        return context
    return step

def make_roi_extraction_step(cfg: Dict):
    """Compute ROI means from a .nii file using the step's atlas """
    from als_sustain.preprocessing.extract_roi_means_dummy import extract_roi_means_dummy
    atlases = cfg.get("atlases", [])
    input_accepted = cfg.get("input_accepted", [])
    output_type = cfg.get("output_type")
    step_name = cfg.get("step")

    def step(context: Dict):
        current_type = context.get("current_type")
        if current_type not in input_accepted:
            raise ValueError(
                f"Step {step_name} cannot accept input type '{current_type}'. "
                f"Accepted: {input_accepted}"
            )

        maps_path = context.get("input_path")
        if maps_path is None:
            raise ValueError("No input_path available for ROI extraction")

        row = context["row"]
        pid = row["ID"]
        visit = row["Visit"]
        metadata = pd.Series({"ID": pid, "Visit": visit})
        logger.debug(f"Extracting ROI means for subject {pid} visit {visit} using atlases {atlases}")
        
        resources = context.get("resources", {})
        all_atlas_roi_vals = []

        for atlas_name in atlases:
            atlas_nifti = resources["paths"]["atlases"].get(atlas_name)
            roi_vals = extract_roi_means_dummy(maps_path, atlas_nifti)
            indiv_atlas_vals_to_save = pd.concat([metadata, roi_vals])
            csv_path = context.get("subject_outdir", {}) / f"roi_means_{atlas_name}.csv"
            indiv_atlas_vals_to_save.to_frame().T.to_csv(csv_path, index=False)
            all_atlas_roi_vals.append(roi_vals)
        
        combined_atlas_roi_vals = pd.concat(all_atlas_roi_vals)
        all_atlas_vals_to_save = pd.concat([metadata, combined_atlas_roi_vals])
        csv_path = context.get("subject_outdir", {}) / f"roi_means_all_atlas.csv"
        logger.debug(f"Saving combined ROI means to {csv_path}")
        all_atlas_vals_to_save.to_frame().T.to_csv(csv_path, index=False)
        
        context["features"] = combined_atlas_roi_vals
        context["current_type"] = output_type
        return context
    return step

def make_wscore_step(cfg: Dict):
    model_artifact = cfg.get("model_artifact")
    input_accepted = cfg.get("input_accepted", [])
    output_type = cfg.get("output_type")
    step_name = cfg.get("step")

    def step(context: Dict):
        current_type = context.get("current_type")
        if current_type not in input_accepted:
            raise ValueError(
                f"Step {step_name} cannot accept input type '{current_type}'. "
                f"Accepted: {input_accepted}"
            )
        features = context.get("features")
        if features is None:
            raise ValueError("No features available for w-score normalization")
        base_dir = Path(context["base_dir"])
        if model_artifact is None:
            raise ValueError("w-score step requires 'model_artifact' in the step config")
        model_path = base_dir / model_artifact
        
        from als_sustain.preprocessing.compute_wscores import compute_wscores
        ws = compute_wscores(features, wscore_hc_model_path=model_path)
        context["features"] = ws
        context["current_type"] = output_type
        return context
    return step

def make_feature_selection_step(cfg: Dict):
    selected = cfg.get("selected_list", [])
    input_accepted = cfg.get("input_accepted", [])
    output_type = cfg.get("output_type")
    step_name = cfg.get("step")

    def step(context: Dict):
        current_type = context.get("current_type")
        if current_type not in input_accepted:
            raise ValueError(
                f"Step {step_name} cannot accept input type '{current_type}'. "
                f"Accepted: {input_accepted}"
            )
        data = context.get("features")
        if data is None:
            raise ValueError("No features available to select from")
        missing = [f for f in selected if f not in data.index]
        if missing:
            raise ValueError(f"Data does not contain required selected features: {missing}")
        context["features"] = data[selected]
        context["current_type"] = output_type
        logger.debug(f"Loaded features {selected}.")
        return context
    return step

def make_feature_inversion_step(cfg: Dict):
    """Invert sign of certain/all features (such as in w-scores) so larger == more abnormal.
    Controlled by descriptor keys under 'step: feature_inversion':
      - 'invert_all': bool
      - 'invert_specific_features_list': list of feature names to invert
    """
    invert_all = cfg.get("invert_all", False)
    invert_list = cfg.get("invert_specific_features_list", [])
    input_accepted = cfg.get("input_accepted", [])
    output_type = cfg.get("output_type")
    step_name = cfg.get("step")
    
    def step(context: Dict):
        current_type = context.get("current_type")
        if current_type not in input_accepted:
            raise ValueError(
                f"Step {step_name} cannot accept input type '{current_type}'. "
                f"Accepted: {input_accepted}"
            )
          
        data = context.get("features")
        if data is None:
            raise ValueError("No features available to invert")
        if invert_all:
            to_invert = data.index.tolist()
        else:
            to_invert = [f for f in invert_list if f in data.index]
            if to_invert == []:
                raise ValueError("No list of features to invert provided.") 
        for name in to_invert:
            data.loc[name] = -1 * data.loc[name]
        context["features"] = data
        context["current_type"] = output_type
        logger.debug(f"Inverted features {to_invert}.")
        return context
    return step

def run_for_row(
        *,
        row: Dict,
        model_id: str,
        workdir: Path,
        resources: Dict,
        base_dir: Path,
        input_type: str,
    ) -> Dict:
    
    """Process a single CSV row. `row` has keys: ID,Visit,Path
    The Path can be either T1 or a features CSV depending on model descriptor.
    """

    desc = load_descriptor(model_id=model_id, base_dir=base_dir)
    input_path = row["Path"].strip()


    context = {
        "row": row,
        "current_type": input_type,
        "model_id": model_id,
        "workdir": workdir,
        "base_dir": base_dir,
        "desc": desc,
        "input_path": input_path,
        "resources": resources, 
    } 

    if input_path.endswith('.csv'):
        data = pd.read_csv(input_path)
        if data.shape[0] != 1:
            raise ValueError("Input features CSV must have exactly one row")
        # pick first row and convert to Series
        data = data.iloc[0].squeeze() 
        context["features"] = data

    # create subject output directory
    context = ensure_subject_outdir_step(context)

    steps = build_steps_from_processing_chain(desc, input_type)
    for step in steps:
        context = step(context)

    context = predict_step(context)

    result = {
        "ID": row["ID"],
        "Visit": row["Visit"],
        "model_id": model_id,
        "sustain_prediction": context.get("prediction"),
    }
    return result

def run_batch(
        *, 
        input_csv: str, 
        input_type: str,
        model_id: str, 
        workdir: Path, 
        resources: Dict, 
        base_dir: Path,
    ) -> Dict:
    
    df = pd.read_csv(input_csv)
    results = []
    for _, row in df.iterrows():
        r = run_for_row(
            row=row.to_dict(), 
            model_id=model_id, 
            workdir=workdir, 
            resources=resources, 
            base_dir=base_dir,
            input_type=input_type,
            )
        results.append(r)
    return results