"""
Model loader and inference wrapper for ALS SuStaIn pipeline.

This module provides utilities to:
- Load trained models (pickled with pickle or joblib)
- Load supplementary model info from pickled metadata
- Perform inference on a single subject (with internal handling for SuStaIn API requirements)

"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

def load_model(model_path: Path):
    """
    Load a trained model from file.

    Supports .pkl/.pickle (pickle) or other formats (joblib).

    Args:
        model_path (Path): Path to the saved model file

    Returns:
        Loaded model object

    Raises:
        FileNotFoundError: If the model file does not exist
        Exception: If loading fails
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

def predict_with_model(model, samples_sequence, samples_f, data: pd.Series) -> pd.Series:
    """
    Run SuStaIn inference on a single subject.

    Notes:
    - SuStaIn requires at least two rows, so the function internally duplicates
      the subject with slight noise. The duplicate is discarded after prediction.
    - Stage 0 indicates "no subtype"; subtype is set to 0 for these cases.

    Args:
        model: Trained SuStaIn model object with method
            `subtype_and_stage_individuals_newData`
        samples_sequence: model metadata (samples_sequence)
        samples_f: model metadata (samples_f)
        data (pd.Series): Feature vector for a single subject

    Returns:
        pd.Series: Prediction results including:
            - ml_subtype: assigned subtype (categorical)
            - ml_stage: assigned stage (Int64)
            - prob_ml_subtype: probability of assigned subtype
            - prob_ml_stage: probability of assigned stage
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
    output_data['ml_subtype'] = ml_subtype
    output_data['prob_ml_subtype'] = prob_ml_subtype
    output_data['ml_stage'] = ml_stage
    output_data['prob_ml_stage'] = prob_ml_stage

    # Make current subtypes (0, 1, 2) -> (1, 2, 3) instead
    output_data['ml_subtype'] = (output_data['ml_subtype'].astype("Int64") + 1)

    # Adjust subtype for stage 0
    stage0_mask = output_data['ml_stage'] == 0
    output_data.loc[stage0_mask, 'ml_subtype'] = 0
    # Invalidate subtype probability for stage 0
    output_data.loc[stage0_mask, 'prob_ml_subtype'] = 0.0
    output_data['ml_stage'] =  output_data['ml_stage'].astype('Int64')

    # Make ml_subtype categorical
    col_values = output_data['ml_subtype']
    categories = sorted(pd.Series(col_values.dropna().unique()))
    output_data['ml_subtype'] = pd.Categorical(
                col_values,
                categories=categories,
                ordered=False
            )

    # Add probability per subtype
    for i in range(prob_subtype.shape[1]):
        output_data.loc[:,'prob_s%s'%(i+1)] = prob_subtype[:,i]

    # Return only first subject (discard fake row)
    return pd.Series(output_data.iloc[0])