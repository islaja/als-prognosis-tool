import pandas as pd

def compute_wscores(patient_data : pd.Series, wscore_models_bundle: dict):
    """Compute the wscores according to the healthy control model provided.
    """

    if wscore_models_bundle is None:
        raise ValueError('wscore models bundle must be provided to compute wscores.')
    else:
        # Transform Series to a 1-row DataFrame
        # .predict() requires a DataFrame to map 'Age', 'Sex', etc. to columns
        patient_df = patient_data.to_frame().T

        # Add dummy columns if missing so the model formula doesn't crash
        if "scanner" not in patient_df.columns:
            patient_df["scanner"] = "unknown"

        # Force numerical columns to numeric types
        patient_df = patient_df.apply(pd.to_numeric, errors='ignore')

        results = {}
        for roi, bundle in wscore_models_bundle.items():
            if roi in patient_df.columns:
                # Predict returns a Series; we take the first value
                expected_val = bundle["model"].predict(patient_df).iloc[0]
                observed_val = patient_df[roi].iloc[0] 
                wscores = (observed_val - expected_val) / bundle["sigma"]
                results[roi] = round(wscores, 3) if not pd.isna(wscores) else pd.NA
        return pd.Series(results)
    
