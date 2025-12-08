from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Dict

def normalize_features(features: Dict[str, float], scaler: StandardScaler = None):
    """Normalize a features dict to numpy array using provided scaler.
    If scaler is None, returns the raw numpy array and None scaler.
    """
    keys = sorted(features.keys())
    arr = np.array([features[k] for k in keys], dtype=float).reshape(1, -1)

    if scaler is None:
        return arr, None, keys
    else:
        arr_scaled = scaler.transform(arr)
        return arr_scaled, scaler, keys
