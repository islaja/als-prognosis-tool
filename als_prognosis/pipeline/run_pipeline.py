"""
Top-level pipeline runner.

This module reads a model descriptor YAML and routes inputs through a composable
processing step pipeline. Each step is implemented as a function that accepts and
returns a context dictionary. 

Context Keys
------------
row : dict
    CSV row for a subject (keys: ID, Visit, Path, Age, Sex, Scanner, etc.)
current_type : str
    Type of current data in pipeline ('t1w_maps', 'regional_dbm', etc.)
model_id : str
    Identifier of the model being run
outdir : Path
    Root output directory
root_dir : Path
    Project root directory (used to resolve files)
desc : dict
    Parsed model descriptor YAML
input_path : Path
    Path to the current input file (CSV or image)
subject_outdir : Path
    Directory for per-subject outputs
resources : dict
    Loaded resources configuration (atlases, etc.)
debug_dir : Path or None
    Optional directory for debug outputs
features : pandas.Series
    Features extracted/processed through the pipeline
prediction : dict
    Model prediction results

Notes:
- 'features' is created after ROI extraction or CSV input reading.
- 'prediction' is added by `predict_step`.
- 'debug_dir' is optional; only used if debug outputs are enabled.
"""


import logging
#from multiprocessing import context ## TODO check if this is needed or if it conflicts with the context dict we are using for pipeline steps
from pathlib import Path
from typing import Dict, List, Optional

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import warnings

from als_prognosis.backends.pelican.setup import PelicanConfig, ensure_pelican_ready

logger = logging.getLogger(__name__)


def load_descriptor(*, model_id: str, root_dir: Path) -> Dict:
    """
    Load a model descriptor YAML file.

    Args:
        model_id (str): Model identifier (filename without .yaml)
        root_dir (Path): Project root directory

    Returns:
        Dict: Parsed model descriptor

    Raises:
        FileNotFoundError: If the YAML file does not exist
        ValueError: If the YAML file cannot be parsed into a dict
    """
    path = (root_dir / "config" / "models" / f"{model_id}.yaml").resolve()

    if not path.exists():
        raise FileNotFoundError(f"Model descriptor not found: {path}")

    with path.open() as f:
        desc = yaml.safe_load(f)

    if not isinstance(desc, dict):
        raise ValueError(f"Invalid model descriptor format: {path}")

    return desc

def ensure_subject_outdir_step(context: Dict) -> Dict:
    """
    Create and ensure the subject-specific output directory exists.

    Args:
        context (Dict): Pipeline context (must include 'row' and 'outdir')

    Returns:
        Dict: Updated context with 'subject_outdir'

    Raises:
        None explicitly
    """
    row = context["row"]
    outdir = context["outdir"]
    pid = row["ID"]
    visit = row["Visit"]
    subject_outdir = Path(outdir / f"{pid}_{visit}".replace(" ", "_")).resolve()
    subject_outdir.mkdir(parents=True, exist_ok=True)

    context["subject_outdir"] = subject_outdir
    return context

def build_steps_from_processing_chain(desc: Dict, input_type: str) -> tuple[list, bool]:
    """
    Build a list of pipeline step functions from the model descriptor.

    Args:
        desc (Dict): Model descriptor containing 'processing_chain'
        input_type (str): Type of input data

    Returns:
        list: Ordered list of callable step functions

    Raises:
        ValueError: If descriptor has no processing chain or input_type is not accepted
    """
    chain = desc.get("processing_chain", [])
    if not chain:
        raise ValueError("Descriptor has no 'processing_chain' to build steps from")

    start_idx = next((i for i, s in enumerate(chain) if input_type in s.get("input_accepted", [])), None)
    if start_idx is None:
        raise ValueError(f"Model does not accept input_type '{input_type}'")

    active_chain = chain[start_idx:]
    
    # Check if Pelican is actually going to be in the final list
    pelican_needed = any(s.get("step") == "dbm_generation" for s in active_chain)

    steps = []
    for step_cfg in active_chain:
        name = step_cfg.get("step")
        if name == "dbm_generation":
            steps.append(make_run_pelican_step(step_cfg))
        elif name == "roi_extraction":
            steps.append(make_roi_extraction_step(step_cfg))
        elif name == "w_score_normalization":
            steps.append(make_wscore_step(step_cfg))
        elif name == 'sustain_inference':
            steps.append(make_sustain_inference_step(step_cfg))
        elif name == 'survival_prediction':
            steps.append(make_survival_predict_step(step_cfg))
        else:
            raise ValueError(f"Unknown processing step in descriptor: {name}")
        
    return steps, pelican_needed

def make_run_pelican_step(dfg: Dict):
    """
    Create a step function that runs PELICAN to generate DBM maps from T1 input.

    Args:
        dfg (Dict): Step configuration from model descriptor, must contain:
            - input_accepted (list): List of input types this step can accept
            - output_type (str): Output type produced
            - step (str): Step name

    Returns:
        Callable: A step function that modifies the pipeline context

    Raises:
        ValueError: If current input type is not accepted by the step
    """

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

        subject_ID = context["row"]["ID"]
        visit = context["row"]["Visit"]
        subject_outdir = context["subject_outdir"]
        t1_path = context["input_path"]
        
        # convert to .mnc if necessary
        if ".nii" in t1_path.suffixes:
            from als_prognosis.utils.image import nii2minc
            t1_path_suffix = "".join(t1_path.suffixes)
            t1_mnc_path = Path(context["subject_outdir"] / t1_path.name.replace(t1_path_suffix, ".mnc")).resolve()
            if not t1_mnc_path.exists():
                nii2minc(t1_path, t1_mnc_path)
            t1_path = t1_mnc_path

        # Ensure Pelican setup only once per pipeline run
        from als_prognosis.backends.pelican.run import run_pelican
        pelican_output_path = (subject_outdir / "pelican_outputs").resolve()
        dbm_file_paths = run_pelican(
            subject_ID, 
            visit, 
            t1_path,
            pelican_output_path, 
            context.get("pelican_cfg"),
        )
        # Create a single string with each path on a new line
        paths_string = "\n".join(str(p) for p in dbm_file_paths)
        print(f"DBM map(s) created at:\n{paths_string}")

        # For now the current pipeline is treating one visit at a time, so one output.
        # TODO handle multiple visits per subject.
        dbm_file_path = dbm_file_paths[0]
        
        context["input_path"] = dbm_file_path
        context["current_type"] = output_type
        
        return context
    return step

def make_roi_extraction_step(cfg: Dict):
    """
    Create a step function to compute ROI means from input images using specified atlases.

    Args:
        cfg (Dict): Step configuration, must contain:
            - input_accepted (list)
            - output_type (str)
            - step (str)
            - atlases (list): List of atlas names to extract ROIs from

    Returns:
        Callable: A step function that updates context with extracted ROI features

    Context keys used:
        - input_path: Path to the current image or maps
        - subject_outdir: Directory to save ROI CSVs
        - resources: dict with atlas resources

    Context keys modified:
        - features: pandas Series of ROI values
        - current_type: updated to output_type

    Raises:
        ValueError: If current input type is not accepted
    """

    from als_prognosis.preprocessing.roi import compute_roi_all_atlas
    atlases = cfg.get("atlases", [])
    input_accepted = cfg.get("input_accepted", [])
    output_type = cfg.get("output_type")
    step_name = cfg.get("step")
    remove_sulci = bool(cfg.get("remove_sulci"))
    csf_threshold = cfg.get("csf_threshold", None)
    result_prefix = cfg.get("result_prefix", "")

    def step(context: Dict):
        current_type = context["current_type"]
        if current_type not in input_accepted:
            raise ValueError(
                f"Step {step_name} cannot accept input type '{current_type}'. "
                f"Accepted: {input_accepted}"
            )
        
        input_maps_path = context["input_path"]
        # Convert to .nii.gz if necessary
        if input_maps_path.suffix == '.mnc':
            from als_prognosis.utils.image import minc2nii
            maps_nifti_path = Path(context["subject_outdir"] / input_maps_path.name.replace(".mnc", ".nii.gz")).resolve()
            minc2nii(input_maps_path, maps_nifti_path)
            input_maps_path = maps_nifti_path

        row = context["row"]
        metadata = pd.Series({"ID": row["ID"], "Visit": row["Visit"]})
        logger.debug(f"Extracting ROI means for subject {row['ID']} visit {row['Visit']} using atlases {atlases}")
        
        resources = context.get("resources", {})
        
        combined_atlas_roi_vals = compute_roi_all_atlas(
            atlases_name=atlases, 
            root_dir=context["root_dir"],
            subject_meta=metadata,
            input_maps_path=input_maps_path,
            img_resources=resources,
            out_dir=context.get("subject_outdir", {}),
            remove_sulci=remove_sulci,
            csf_threshold=csf_threshold,
            result_prefix=result_prefix,
            debug_dir=context.get("debug_dir"),
            include_sides=False
            )
    
        context["features"] = combined_atlas_roi_vals
        context["current_type"] = output_type
        return context
    return step

def make_wscore_step(cfg: Dict):
    """
    Create a step function to compute w-scores for extracted features.

    Args:
        cfg (Dict): Step configuration, must contain:
            - input_accepted (list)
            - output_type (str)
            - step (str)
            - model_artifact (str): Path to w-score model file

    Returns:
        Callable: A step function that updates context with normalized w-scores

    Context keys used:
        - features: pandas Series with extracted features
        - root_dir: project root directory
        - row: dict with patient metadata (ID, Age, Sex, Scanner)
        - subject_outdir: directory to save w-score CSV

    Context keys modified:
        - features: updated with w-scores
        - current_type: updated to output_type

    Raises:
        ValueError: If features are missing or model_artifact not provided
        FileNotFoundError: If model_artifact file does not exist
    """
    model_artifact = str(cfg.get("model_artifact"))
    input_accepted = cfg.get("input_accepted", [])
    output_type = cfg.get("output_type")
    step_name = cfg.get("step")
    
    if model_artifact is None:
        raise ValueError("w-score step requires 'model_artifact' in the step config")
    
    # Outer scope variable
    cache = {}

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
        
        root_dir = Path(context["root_dir"])

        if "bundle" not in cache:
            print("Loading wscore models bundle for the first time...")
            model_path = (root_dir / model_artifact).resolve()
            with open(model_path, "rb") as f:
                cache["bundle"] = joblib.load(f)
        
        # Use the cached version
        wscore_models_bundle = cache["bundle"]
        
        # Extract patient metadata from the dictionary into a Series
        row = context["row"]
        metadata = pd.Series({
            "filename": row["ID"],
            "age": row["Age"],
            "sex": row["Sex"],
            "scanner": row["Scanner"]
        })
        not_numerical = ["filename", "sex", "scanner"]

        # Combine with the existing brain features Series
        patient_data = pd.concat([metadata, context["features"]])

        
        from als_prognosis.preprocessing.wscores import compute_wscores
        ws = compute_wscores(patient_data=patient_data, wscore_models_bundle=wscore_models_bundle, not_numerical=not_numerical)
        csv_path = context.get("subject_outdir", {}) / f"roi_wscores_all_atlas.csv"
        patient_ws_to_save = pd.concat([metadata, ws])
        # Round only the numeric part of the series first
        patient_ws_to_save = patient_ws_to_save.apply(
            lambda x: round(x, 3) if isinstance(x, (int, float)) else x
        )
        # Now transpose and save
        logger.debug(f"Saving ROI wscores to {csv_path}")
        patient_ws_to_save.to_frame().T.to_csv(csv_path, index=False)
        
        context["features"] = ws
        context["current_type"] = output_type

        return context
    return step


def make_sustain_inference_step(step_cfg: Dict):
    """
    Create a step function that runs SuStaIn to obtain stage and subtype predictions.
    
    This step handles the end-to-end model preparation:
    1. Feature Selection: Extracts and orders features according to 'regions_list'.
    2. Sign Standardization: Optionally inverts features (e.g., atrophy) to ensure 
       positive values represent pathology, while allowing exceptions (e.g., ventricles).
    3. Model Inference: Loads the SuStaIn model and meta-info to produce assignments.

    Args:
        step_cfg (Dict): Step configuration from model descriptor, must contain:
            - input_accepted (list): List of input types this step can accept.
            - output_type (str): Output type produced (e.g., 'sustain_assignments').
            - regions_list (list): The exact features and order required by the model.
            - standardize_to_positive_pathology (bool): If True, enables sign inversion.
            - exclude_from_inversion (list): Features to skip during inversion (e.g. expansion).

    Returns:
        Callable: A step function that modifies the pipeline context with 'prediction'
                  and updates 'current_type'.

    Raises:
        ValueError: If current input type is not accepted by the step.
        ValueError: If the feature data is missing from the context.
        ValueError: If required features in 'regions_list' are missing from the input.
        ValueError: If model or metadata files (.joblib/.pickle) are missing.
    """

    input_accepted = step_cfg.get("input_accepted", [])
    output_type = step_cfg.get("output_type")
    step_name = step_cfg.get("step")

    regions = step_cfg.get("regions_list", [])
    standardize_to_positive_pathology = step_cfg.get("standardize_to_positive_pathology", False)
    exclude_from_inversion = step_cfg.get("exclude_from_inversion", [])

    def step(context: Dict):
        current_type = context.get("current_type")
        if current_type not in input_accepted:
            raise ValueError(
                f"Step {step_name} cannot accept input type '{current_type}'. "
                f"Accepted: {input_accepted}"
            )
        
        from als_prognosis.progression.predict import load_pickle_info, infer_with_model
    
        desc = context["desc"]
        data = context.get("features")
        
        if data is None:
            raise ValueError("No data available for inference")

        missing = [f for f in regions if f not in data.index]
        if missing:
            raise ValueError(f"Data does not contain required selected features: {missing}")
        
        # Ensure data is ordered according to the regions list expected by the model
        data = data.loc[regions]

        # Optionally invert features to standardize directionality for SuStaIn
        if standardize_to_positive_pathology:
            for feature in data.index:
                if feature in exclude_from_inversion:
                    continue
                data.at[feature] = -1 * data.at[feature]    

        # Get the model and meta file paths from the descriptor
        model_file_rel = desc.get("models", {}).get("sustain_model", {}).get("model_file")
        model_meta_rel = desc.get("models", {}).get("sustain_model", {}).get("model_meta_file")
        
        model_file = (context["root_dir"] / Path(model_file_rel)).resolve()
        meta_file = (context["root_dir"] / Path(model_meta_rel)).resolve()

        if not meta_file.exists():
            raise ValueError(f"Sustain meta file in descriptor not found: {meta_file}")
        if not model_file.exists():
            raise ValueError(f"Sustain model file specified in descriptor not found: {model_file}")
        
        # Load the model and meta-info, then run prediction
        model = joblib.load(model_file)
        samples_sequence, samples_f = load_pickle_info(meta_file)
        inference = infer_with_model(model, samples_sequence, samples_f, data)

        context["sustain_inference"] = inference.round(3).to_dict()

        # Create the interaction term features (e.g., S1_x_stage) based on the sustain subtype and stage predictions, 
        # if needed for downstream steps like Coxnet survival prediction.
        nb_subtypes = desc.get("models", {}).get("sustain_model", {}).get("model_nb_subtype", 3)
        for subtype in range(1, nb_subtypes+1):
            inference[f"S{subtype}_x_stage"] = inference['inferred_stage'] * (inference['inferred_subtype'] == subtype).astype(int)
        
        context["features"] = inference  # Round features for any downstream use and saving
        context["current_type"] = output_type
        
        return context
    return step


def make_survival_predict_step(step_cfg: Dict):
    """
    Create a step function that runs survival prediction to obtain the ISD curve prediction.
    
    This step handles the end-to-end model preparation:
    1. Optionally, perform the sustain interaction term generation (e.g., S1_x_stage).
    2. Feature Selection: Extracts and orders features according to 'features_list'.
    3. Model Inference: Loads the coxnet survival bootstrap ensemble models and apply each of them to produce the mean predicted survival curve.

    Args:
        step_cfg (Dict): Step configuration from model descriptor, must contain:
            - input_accepted (list): List of input types this step can accept.
            - output_type (str): Output type produced (e.g., 'sustain_assignments').
            - regions_list (list): The exact features and order required by the model.
            - standardize_to_positive_pathology (bool): If True, enables sign inversion.
            - exclude_from_inversion (list): Features to skip during inversion (e.g. expansion).

    Returns:
        Callable: A step function that modifies the pipeline context with 'prediction'
                  and updates 'current_type'.

    Raises:
        ValueError: If current input type is not accepted by the step.
        ValueError: If the feature data is missing from the context.
        ValueError: If required features in 'regions_list' are missing from the input.
        ValueError: If model or metadata files (.joblib/.pickle) are missing.
    """

    input_accepted = step_cfg.get("input_accepted", [])
    output_type = step_cfg.get("output_type")
    step_name = step_cfg.get("step")

    features_list = step_cfg.get("features_list", [])
    period_window = step_cfg.get("period_window", 60)

    def step(context: Dict):
        current_type = context.get("current_type")
        if current_type not in input_accepted:
            raise ValueError(
                f"Step {step_name} cannot accept input type '{current_type}'. "
                f"Accepted: {input_accepted}"
            )
        
        from als_prognosis.survival.predict import infer_curves_from_bootstrap_models, plot_coxnet_prediction_over_references
        
        subject_outdir = context.get("subject_outdir", {}) 

        desc = context["desc"]
        subject_info = context["row"]
        data = context.get("features")
        
        if data is None:
            raise ValueError("No data available for prediction")
            
        for feature in features_list:
            if feature not in data.index:
                # look in sujbect_info as well for clinical or demographic features
                if feature in subject_info:
                    data[feature] = subject_info[feature]
                else:
                    warnings.warn(f"Missing '{feature}': Skipping survival prediction.", UserWarning)
                    context["survival_prediction"] = None
                    return context

        model_ensemble_file_rel = desc.get("models", {}).get("survival_ensemble_model", {}).get("model_file")
        model_references_file_rel = desc.get("models", {}).get("survival_ensemble_model", {}).get("reference_library")
        
        model_ensemble_file = (context["root_dir"] / Path(model_ensemble_file_rel)).resolve()
        model_references_file = (context["root_dir"] / Path(model_references_file_rel)).resolve()
        
        if not model_ensemble_file.exists():
            raise ValueError(f"Coxnet model ensemble file in descriptor not found: {model_ensemble_file}")
        if not model_references_file.exists():
            raise ValueError(f"References library file in descriptor not found: {model_references_file}")
        
        bootstrap_models = joblib.load(model_ensemble_file)
        reference_library = joblib.load(model_references_file)

        common_times = np.linspace(0, period_window, period_window * 2)
        nb_subtypes = desc.get("models", {}).get("sustain_model", {}).get("model_nb_subtype", 3)
        patient_mean_curve_result = infer_curves_from_bootstrap_models(data, bootstrap_models, common_times)
        patient_subtype = data['inferred_subtype']

        title_suffix = f"{subject_info['ID']}\n Inferred Subtype: {int(patient_subtype)}, Stage: {int(data['inferred_stage'])}"
        fig, ax = plot_coxnet_prediction_over_references(
            patient_mean_curve_result, 
            reference_library, 
            nb_subtypes, 
            title_suffix=title_suffix,
        )
        fig.show(False)
        fig.savefig(subject_outdir / f"prognosis.png", dpi=300, bbox_inches='tight')
        plt.close('all')

        pd.DataFrame({
            'time_in_months': common_times, 
            'survival_probability': patient_mean_curve_result['mean']
        }).to_csv(subject_outdir / "predicted_survival_curve.csv", index=False)
        
        context["median_survival_time"] = np.round(patient_mean_curve_result['median_survival_time'], 1).item()
        context["features"] = patient_mean_curve_result
        context["current_type"] = output_type
        
        return context
    return step


def run_for_row(
        *,
        steps: list,
        row: Dict,
        model_id: str,
        desc: Dict,
        outdir: Path,
        resources: Dict,
        root_dir: Path,
        input_type: str,
        pelican_cfg: Optional[PelicanConfig] = None,
        debug_dir: Optional[Path] = None,
    ) -> Dict: 
    """
    Run the full processing pipeline for a single subject row.

    Args:
        steps (list): List of processing step functions
        row (Dict): CSV row with keys: ID, Visit, Visit_Date, Age, Sex, Scanner, Path
        model_id (str): Model identifier
        desc (Dict): Model descriptor
        outdir (Path): Root output directory
        resources (Dict): Resources loaded from config
        root_dir (Path): Project root directory
        input_type (str): Type of input (features, DBM maps, T1)
        debug_dir (Optional[Path]): Optional debug outputs directory

    Returns:
        Dict: Prediction result for this subject

    Raises:
        FileNotFoundError: If input file does not exist
        ValueError: If input file type is invalid or features are missing
    """

    input_path = Path(row["Path"].strip()).resolve()  

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    input_suffixes = input_path.suffixes
    if 'maps' in input_type:
        # make sure the input is either a .nii/.nii.gz/.mnc file
        if not any(suf in input_suffixes for suf in ['.nii', '.nii.gz', '.mnc']):
            raise ValueError(f"Input path for input_type '{input_type}' must be a .nii/.nii.gz/.mnc file")
    else: # tabular input_type, expecting a .csv file
        if '.csv' not in input_suffixes:
            raise ValueError(f"Input path for input_type '{input_type}' must be a .csv file")
        
    context = {
        "row": row,
        "current_type": input_type,
        "model_id": model_id,
        "outdir": outdir,
        "root_dir": root_dir,
        "desc": desc,
        "input_path": input_path,
        "resources": resources,
        "pelican_cfg": pelican_cfg,
        "debug_dir": debug_dir,
    } 

    # create subject output directory
    context = ensure_subject_outdir_step(context)
    context["input_path"] = input_path
    
    if '.csv' in input_suffixes:
        data = pd.read_csv(input_path)
        if data.shape[0] != 1:
            raise ValueError("Input features CSV must have exactly one row")
        # pick first row and convert to Series
        data = data.iloc[0].squeeze() 
        context["features"] = data
 
    for step in steps:
        context = step(context)

    result = {
        "ID": row["ID"],
        "Visit": row["Visit"],
        "model_id": model_id,
        "sustain_inference": context.get("sustain_inference"),
        "predicted_median_survival_time(months)": context.get("median_survival_time"),
    }
    return result

def run_batch(
        *, 
        input_csv: str, 
        input_type: str,
        model_id: str, 
        outdir: Path, 
        resources: Dict, 
        root_dir: Path,
        debug_dir: Optional[Path] = None,
    ) -> List:
    """
    Run the pipeline for all rows in a CSV file.

    Args:
        input_csv (str): Path to CSV with participant rows
        input_type (str): Type of input data (features, DBM maps, T1)
        model_id (str): Model identifier
        outdir (Path): Root output directory
        resources (Dict): Loaded resources configuration
        root_dir (Path): Project root directory
        debug_dir (Optional[Path]): Directory for debug outputs

    Returns:
        List[Dict]: List of prediction results for each row

    Raises:
        FileNotFoundError: If CSV or inputs are missing
        ValueError: If CSV or data are invalid
    """

    df = pd.read_csv(input_csv)
    desc = load_descriptor(model_id=model_id, root_dir=root_dir)
    steps, pelican_needed = build_steps_from_processing_chain(desc, input_type)

    pelican_cfg: Optional[PelicanConfig] = None
    if pelican_needed:
        pelican_cfg = ensure_pelican_ready()

    results = []
    for _, row in df.iterrows():
        r = run_for_row(
            steps=steps,
            row=row.to_dict(), 
            model_id=model_id,
            desc=desc, 
            outdir=outdir, 
            resources=resources, 
            root_dir=root_dir,
            input_type=input_type,
            pelican_cfg=pelican_cfg,
            debug_dir=debug_dir,
            )
        results.append(r)
    return results