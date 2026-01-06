import pandas as pd

def compute_wscores(features : pd.Series, wscore_hc_model = None):
    """Compute the wscores according to the healthy control model provided.
    If no model is provided, return the original features.
    """
   
    if wscore_hc_model is None:
        raise ValueError('wscore hc model must be provided to compute wscores.')
    else:
        #arr_scaled = scaler.transform(arr)
        arr_scaled = features
        return arr_scaled
