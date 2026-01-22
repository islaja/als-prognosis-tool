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
from pathlib import Path
from typing import Dict, List, Optional

import yaml
import pandas as pd
import joblib

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

def predict_step(context: Dict) -> Dict:
    """
    Generate predictions for the current context using the model.

    Args:
        context (Dict): Pipeline context (must include 'desc', 'features', 'root_dir', 'current_type')

    Returns:
        Dict: Context updated with 'prediction'

    Raises:
        ValueError: If features are missing or input type is invalid
    """

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
    
    model_file_rel = desc.get("model_metadata", {}).get("model_file")
    model_meta_rel = desc.get("model_metadata", {}).get("model_meta_file")
    
    if not model_file_rel or not model_meta_rel:
        raise ValueError("Model descriptor missing 'model_file' or 'model_meta_file'")

    model_file = (context["root_dir"] / Path(model_file_rel)).resolve()
    model = load_model(model_file)

    meta_file = (context["root_dir"] / Path(model_meta_rel)).resolve()
    samples_sequence, samples_f = load_pickle_info(meta_file)

    data = context.get("features")
    if data is None:
        raise ValueError("No data available for prediction")
    prediction = predict_with_model(model, samples_sequence, samples_f, data)
    context["prediction"] = prediction.to_dict()
    return context

def build_steps_from_processing_chain(desc: Dict, input_type: str) -> list:
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
        
        t1_path = context["input_path"]
        
        # convert to .mnc if necessary
        if ".nii" in t1_path.suffixes:
            from als_sustain.utils.image import nii2minc
            t1_path_suffix = "".join(t1_path.suffixes)
            t1_mnc_path = Path(context["subject_outdir"] / t1_path.name.replace(t1_path_suffix, ".mnc")).resolve()
            nii2minc(t1_path, t1_mnc_path)
            t1_path = t1_mnc_path

        from als_sustain.preprocessing.pelican_runner import run_pelican_dummy
        dbm_path = run_pelican_dummy(t1_path, context["subject_outdir"])
        context["input_path"] = dbm_path
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

    from als_sustain.preprocessing.roi import compute_roi
    atlases = cfg.get("atlases", [])
    input_accepted = cfg.get("input_accepted", [])
    output_type = cfg.get("output_type")
    step_name = cfg.get("step")

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
            from als_sustain.utils.image import minc2nii
            maps_nifti_path = Path(context["subject_outdir"] / input_maps_path.name.replace(".mnc", ".nii.gz")).resolve()
            minc2nii(input_maps_path, maps_nifti_path)
            input_maps_path = maps_nifti_path

        row = context["row"]
        pid = row["ID"]
        visit = row["Visit"]
        metadata = pd.Series({"ID": pid, "Visit": visit})
        logger.debug(f"Extracting ROI means for subject {pid} visit {visit} using atlases {atlases}")
        
        resources = context.get("resources", {})
        all_atlas_roi_vals = []

        for atlas_name in atlases:
            roi_vals = compute_roi(
                root_dir=context["root_dir"],
                img_resources=resources,
                atlas_name=atlas_name,
                input_maps_path=input_maps_path,
                debug_dir=context.get("debug_dir"),
                )
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

        # Combine with the existing brain features Series
        patient_data = pd.concat([metadata, context["features"]])

        from als_sustain.preprocessing.wscores import compute_wscores
        ws = compute_wscores(patient_data=patient_data, wscore_models_bundle=wscore_models_bundle)
        patient_ws_to_save = pd.concat([metadata, ws])
        csv_path = context.get("subject_outdir", {}) / f"roi_wscores_all_atlas.csv"
        logger.debug(f"Saving ROI wscores to {csv_path}")
        patient_ws_to_save.to_frame().T.to_csv(csv_path, index=False)
        
        context["features"] = ws
        context["current_type"] = output_type

        return context
    return step

def make_feature_selection_step(cfg: Dict):
    """
    Create a step function that selects a subset of features.

    Args:
        cfg (Dict): Step configuration, must contain:
            - input_accepted (list)
            - output_type (str)
            - step (str)
            - selected_list (list): Features to retain

    Returns:
        Callable: A step function that filters features in context

    Context keys used:
        - features: pandas Series with feature values

    Context keys modified:
        - features: reduced to selected subset
        - current_type: updated to output_type

    Raises:
        ValueError: If features are missing or required features are not present
    """

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
    """
    Create a step function to invert the sign of features so larger values indicate
    more abnormality.

    Args:
        cfg (Dict): Step configuration, must contain:
            - input_accepted (list)
            - output_type (str)
            - step (str)
            - invert_all (bool): Whether to invert all features
            - invert_specific_features_list (list): Specific features to invert

    Returns:
        Callable: A step function that updates context with inverted features

    Context keys used:
        - features: pandas Series with feature values

    Context keys modified:
        - features: with inverted signs
        - current_type: updated to output_type

    Raises:
        ValueError: If no features are available or no features specified for inversion
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
        steps: list,
        row: Dict,
        model_id: str,
        desc: Dict,
        outdir: Path,
        resources: Dict,
        root_dir: Path,
        input_type: str,
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
    steps = build_steps_from_processing_chain(desc, input_type)

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
            debug_dir=debug_dir,
            )
        results.append(r)
    return results