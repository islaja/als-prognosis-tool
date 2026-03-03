"""
Subtype and Stage Inference Wrapper for ALS SuStaIn.

This module provides the core utilities for the SuStaIn (Subtype and Stage 
Inference) component of the ALS prognosis pipeline. It handles the 
serialization of Bayesian models and the extraction of latent disease 
subtypes and stages from neuroimaging or clinical features.

Key Capabilities:
    - Extraction of Bayesian MCMC samples (the "rules" of disease progression).
    - Subtype and Stage Identification: Determines where a patient sits on 
      the disease timeline.
    - Healthy-Subject Handling: Automatically categorizes subjects at "Stage 0" 
      as having no specific subtype yet.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple, Any


def load_pickle_info(pickle_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract SuStaIn MCMC metadata from a pickled results file.

    This metadata contains the posterior distributions of the subtype 
    sequences and prevalence fractions necessary for new-subject inference.

    Args:
        pickle_path (Path): Path to the SuStaIn output pickle.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - samples_sequence: The inferred ordering of biomarkers.
            - samples_f: The inferred subtype proportions.

    Raises:
        ValueError: If required SuStaIn keys are missing from the file.
    """
    try:
        pk = pd.read_pickle(pickle_path)
        samples_sequence = pk["samples_sequence"]
        samples_f = pk["samples_f"]
        return samples_sequence, samples_f
    except Exception as e:
        raise ValueError(f"Required SuStaIn keys missing in {pickle_path}: {e}")

def infer_with_model(
    model: Any, 
    samples_sequence: np.ndarray, 
    samples_f: np.ndarray, 
    data: pd.Series
) -> pd.Series:
    """
    Infers latent SuStaIn subtype and stage for a single subject.

    This function maps observed patient features to the most likely latent 
    disease state. It includes internal handling for the SuStaIn API requirement 
    of multi-row inputs and normalizes stage-0 (pre-symptomatic/healthy) 
    assignments.

    Args:
        model: Trained SuStaIn model object.
        samples_sequence (np.ndarray): Posterior samples of marker sequences.
        samples_f (np.ndarray): Posterior samples of subtype fractions.
        data (pd.Series): Feature vector for a single subject.

    Returns:
        pd.Series: Inference results including:
            - inferred_subtype: Assigned subtype (S0 for no-subtype, else S1, S2...).
            - inferred_stage: Assigned disease stage (Int64).
            - prob_inferred_subtype: Confidence in the assigned subtype.
            - prob_inferred_stage: Confidence in the assigned stage.
            - prob_sN: Individual probability scores for each possible subtype.

    Note:
        SuStaIn requires at least two rows for inference; this function 
        automatically handles data augmentation and subsequent cleanup.
    """
    # Prepare 2-row input for SuStaIn API compatibility
    data_nparray = np.asarray(data, dtype=np.float64).reshape(1, -1)
    # Add a slightly jittered row to satisfy SuStaIn requirement
    jittered_row = data_nparray[0] + 0.2 * data_nparray[0]
    input_stack = np.vstack([data_nparray, jittered_row])
    
    N_samples = 1000
    
    # SuStaIn Inference call
    (ml_subtype, 
     prob_ml_subtype, 
     ml_stage, 
     prob_ml_stage, 
     prob_subtype, 
     _, 
     _) = model.subtype_and_stage_individuals_newData(
         input_stack, samples_sequence, samples_f, N_samples
     )

    # Initialize results container
    output_data = pd.DataFrame(index=[0, 1])
    # Shift subtype indexing from 0-based to 1-based 
    output_data['inferred_subtype'] = (ml_subtype.astype("int") + 1)
    output_data['prob_inferred_subtype'] = prob_ml_subtype
    output_data['inferred_stage'] = ml_stage.astype("int")
    output_data['prob_inferred_stage'] = prob_ml_stage

    # Normalization for Stage 0 (Healthy/Control-like)
    stage0_mask = output_data['inferred_stage'] == 0
    output_data.loc[stage0_mask, 'inferred_subtype'] = 0
    output_data.loc[stage0_mask, 'prob_inferred_subtype'] = 0.0

    # Categorical casting for inferred_subtype
    categories = sorted(output_data['inferred_subtype'].unique())
    output_data['inferred_subtype'] = pd.Categorical(
        output_data['inferred_subtype'], categories=categories, ordered=False
    )

    # Map individual subtype probabilities (prob_s1, prob_s2, etc.)
    for i in range(prob_subtype.shape[1]):
        output_data[f'prob_s{i+1}'] = prob_subtype[:, i]

    # Discard the augmented row and return single subject Series
    return output_data.iloc[0]