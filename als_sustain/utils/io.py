import pandas as pd
import numpy as np

def make_json_safe(o):
    if isinstance(o, pd.Series):
        #return o.tolist()
        return o.to_dict()
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.float32, np.float64, np.int64, np.int32)):
        return o.item()
    if isinstance(o, dict):
        return {k: make_json_safe(v) for k, v in o.items()}
    if isinstance(o, list):
        return [make_json_safe(x) for x in o]
    return o