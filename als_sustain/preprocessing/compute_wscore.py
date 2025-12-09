def compute_wscores(features, wscore_hc_model = None):
    """Compute the wscores according to the healthy control model provided.
    If no model is provided, return the original features.
    """
   
    if wscore_hc_model is None:
        raise ValueError('wscore hc model must be provided to compute wscores.')
    else:
        #arr_scaled = scaler.transform(arr)
        #return arr_scaled
