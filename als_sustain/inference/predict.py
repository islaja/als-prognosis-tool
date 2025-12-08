"""Model loader and inference wrapper.
This wrapper expects trained models to be pickled and to expose a method
that accepts numpy arrays and returns (subtype, stage) or similar.
Adapt to your SuStaIn implementation.
"""
import pickle
import numpy as np
from typing import Tuple, Dict

class ModelDescriptor:
    def __init__(self, descriptor: Dict):
        self.descriptor = descriptor

    @property
    def model_file(self):
        return descriptor_path_resolve(self.descriptor["model_file"])

def load_model(model_path: str):
    with open(model_path, "rb") as f:
        obj = pickle.load(f)
    return obj

def predict_with_model(model, features_array: np.ndarray) -> Tuple[str, float]:
    """Call the model's inference method. Modify to match your model API.
    Expected return: (subtype_label, stage_value)
    """
    # Replace the line below with your model's inference call
    if hasattr(model, "infer"):
        out = model.infer(features_array)
        return out
    elif hasattr(model, "predict"):
        out = model.predict(features_array)
        return out
    else:
        # fallback: if model is a scikit-learn like object
        preds = model.predict(features_array)
        return (str(preds[0]), float(0))
