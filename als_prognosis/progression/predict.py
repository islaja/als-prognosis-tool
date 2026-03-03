"""
Model Loader and Subtype and Stage Inference Wrapper for ALS SuStaIn.

This module provides the core utilities for the SuStaIn (Subtype and Stage 
Inference) component of the ALS prognosis pipeline. It handles the 
serialization of Bayesian models and the extraction of latent disease 
subtypes and stages from neuroimaging or clinical features.

Key Capabilities:
    - Robust model loading (supporting pickle and joblib).
    - Extraction of Bayesian MCMC samples (the "rules" of disease progression).
    - Subtype and Stage Identification: Determines where a patient sits on 
      the disease timeline.
    - Healthy-Subject Handling: Automatically categorizes subjects at "Stage 0" 
      as having no specific subtype yet.
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

def load_model(model_path: Path) -> Any:
    """
    Load a trained SuStaIn or Survival model from a serialized file.

    Supports standard Python .pkl/.pickle files and joblib formats used for 
    large numpy-heavy estimators.

    Args:
        model_path (Path): Path to the saved model file.

    Returns:
        Any: The loaded model object (e.g., SuStaIn instance or Coxnet estimator).

    Raises:
        FileNotFoundError: If the model file does not exist.
        RuntimeError: If the file is corrupted or format is unsupported.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with model_path.open("rb") as f:
        if model_path.suffix == '.pkl' or model_path.suffix == '.pickle':
            print("Loaded model using pickle.")
            obj = pickle.load(f)
        else:
            print("Loaded model using joblib.")
            obj = joblib.load(f)  
  
    return obj

def load_pickle_info(pickle_path: Path):
    """
    Load SuStaIn metadata from a pickled file.

    Expected keys in pickle:
      - 'samples_sequence'
      - 'samples_f'

    Args:
        pickle_path (Path): Path to the pickled metadata file

    Returns:
        tuple: (samples_sequence, samples_f)

    Raises:
        ValueError: If required keys are missing or file cannot be read
    """
    try:
        pk = pd.read_pickle(pickle_path)
        samples_sequence = pk["samples_sequence"]
        samples_f = pk["samples_f"]
        return samples_sequence, samples_f
    except Exception as e:
        raise ValueError(f"Error loading pickle info from {pickle_path}: {e}")

def infer_with_model(model, samples_sequence, samples_f, data: pd.Series) -> pd.Series:
    """
    Run SuStaIn inference on a single subject.

    Notes:
    - SuStaIn requires at least two rows, so the function internally duplicates
      the subject with slight noise. The duplicate is discarded after inference.
    - Stage 0 indicates "no subtype"; subtype is set to 0 for these cases.

    Args:
        model: Trained SuStaIn model object with method
            `subtype_and_stage_individuals_newData`
        samples_sequence: model metadata (samples_sequence)
        samples_f: model metadata (samples_f)
        data (pd.Series): Feature vector for a single subject

    Returns:
        pd.Series: Inference results including:
            - inferred_subtype: assigned subtype (categorical)
            - inferred_stage: assigned stage (Int64)
            - prob_inferred_subtype: probability of assigned subtype
            - prob_inferred_stage: probability of assigned stage
            - prob_sN: probability for each subtype N

    Raises:
        ValueError: If input data is invalid
    """
    # Prepare 2-row input for SuStaIn
    output_data = pd.DataFrame(index=range(2))
    data_nparray = np.asarray(data, dtype=np.float64).reshape(1, -1)
    new_row = data_nparray[0] + 0.2*data_nparray[0]
    data_nparray_with_additional_fake_subject = np.vstack([data_nparray, new_row])
    
    N_samples = 1000
    
    ml_subtype,             \
    prob_ml_subtype,        \
    ml_stage,               \
    prob_ml_stage,          \
    prob_subtype,           \
    prob_stage,             \
    prob_subtype_stage  = model.subtype_and_stage_individuals_newData(data_nparray_with_additional_fake_subject,
                                                                    samples_sequence,
                                                                    samples_f,
                                                                    N_samples)

    # Collect outputs into DataFrame
    output_data['inferred_subtype'] = ml_subtype
    output_data['prob_inferred_subtype'] = prob_ml_subtype
    output_data['inferred_stage'] = ml_stage
    output_data['prob_inferred_stage'] = prob_ml_stage

    # Make current subtypes (0, 1, 2) -> (1, 2, 3) instead
    output_data['inferred_subtype'] = (output_data['inferred_subtype'].astype("Int64") + 1)

    # Adjust subtype for stage 0
    stage0_mask = output_data['inferred_stage'] == 0
    output_data.loc[stage0_mask, 'inferred_subtype'] = 0
    # Invalidate subtype probability for stage 0
    output_data.loc[stage0_mask, 'prob_inferred_subtype'] = 0.0
    output_data['inferred_stage'] =  output_data['inferred_stage'].astype('Int64')

    # Make inferred_subtype categorical
    col_values = output_data['inferred_subtype']
    categories = sorted(pd.Series(col_values.dropna().unique()))
    output_data['inferred_subtype'] = pd.Categorical(
                col_values,
                categories=categories,
                ordered=False
            )

    # Add probability per subtype
    for i in range(prob_subtype.shape[1]):
        output_data.loc[:,'prob_s%s'%(i+1)] = prob_subtype[:,i]

    # Return only first subject (discard fake row)
    return pd.Series(output_data.iloc[0])