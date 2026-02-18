"""
W-score computation utilities for ALS SuStaIn.

Provides functions to compute w-scores for each ROI
based on healthy control models.
"""

import pandas as pd

def compute_wscores(patient_data : pd.Series, wscore_models_bundle: dict):
    """
    Compute w-scores for a single subject based on a healthy control model.

    Each ROI's w-score is calculated as:
        (observed_value - predicted_value) / sigma

    Args:
        patient_data (pd.Series): Patient features, including covariates like
            'Age', 'Sex', 'scanner' and ROI values.
        wscore_models_bundle (dict): Dictionary mapping ROI names to model info:
            {
                'roi_name': {'model': fitted_model, 'sigma': float},
                ...
            }

    Returns:
        pd.Series: W-scores per ROI, with ROI names as index.

    Raises:
        ValueError: If `wscore_models_bundle` is None.
    """

    if wscore_models_bundle is None:
        raise ValueError('wscore models bundle must be provided to compute wscores.')
    else:
        # Convert patient series to a 1-row DataFrame for model prediction
        patient_df = patient_data.to_frame().T

        # Add missing dummy columns if needed
        if "scanner" not in patient_df.columns:
            patient_df["scanner"] = "unknown"

        # Ensure numeric types where applicable
        patient_df = patient_df.apply(pd.to_numeric, errors='ignore')

        results = {}
        for roi, bundle in wscore_models_bundle.items():
            if roi in patient_df.columns:
                expected_val = bundle["model"].predict(patient_df).iloc[0]
                observed_val = patient_df[roi].iloc[0] 
                results[roi] = (observed_val - expected_val) / bundle["sigma"]
        
        return pd.Series(results)
    
