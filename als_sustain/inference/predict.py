"""Model loader and inference wrapper.
This wrapper expects trained models to be pickled and to expose a method
that accepts numpy arrays and returns (subtype, stage) or similar.
Adapt to your SuStaIn implementation.
"""
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from pathlib import Path

def load_model(model_path: str):
    with open(model_path, "rb") as f:
        obj = pickle.load(f)
    return obj

def load_pickle_info(pickle_path: str):
    pk = pd.read_pickle(pickle_path)
    samples_sequence = pk["samples_sequence"]
    samples_f = pk["samples_f"]

    return samples_sequence, samples_f

def predict_with_model(model, samples_sequence, samples_f, data: np.ndarray) -> np.ndarray:
    """Call the model's inference method. Modify to match your model API.
    Expected return: (subtype_label, stage_value)
    """
    
    output_data = np.array([]).reshape(data.shape[0],0)

    N_samples = 1000
    
    ml_subtype,             \
    prob_ml_subtype,        \
    ml_stage,               \
    prob_ml_stage,          \
    prob_subtype,           \
    prob_stage,             \
    prob_subtype_stage  = model.subtype_and_stage_individuals_newData(data,
                                                                    samples_sequence,
                                                                    samples_f,
                                                                    N_samples)

    nb_subtype = len(np.unique(ml_subtype))
   
    output_data['inf_subtype'] = ml_subtype
    output_data['inf_subtype_prob'] = prob_ml_subtype
    output_data['inf_stage'] = ml_stage
    output_data['inf_stage_prob'] = prob_ml_stage

    # make current subtypes (0, 1, 2) 1 and 2, 3 instead
    #output_data.loc[:, ml_subtype_col] = (output_data[ml_subtype_col].to_numpy() + 1).astype('Int64')
    output_data['inf_subtype'] = (output_data['inf_subtype'] + 1).astype("Int64")

    # Define a mask for stage 0
    stage0_mask = output_data['inf_stage'] == 0
    # Assign subtype = 0 for stage 0 (meaning: no valid subtype)
    output_data.loc[stage0_mask, 'inf_subtype'] = 0

    # set stage as int
    output_data['inf_stage'] =  output_data['inf_stage'].astype('Int64')

    # set ml_subtype as categorical
    col_values = output_data['inf_subtype']
    categories = sorted(pd.Series(col_values.dropna().unique()))
    output_data['inf_subtype'] = pd.Categorical(
                col_values,
                categories=categories,
                ordered=False
            )

    # Invalidate subtype probability for stage 0
    output_data.loc[stage0_mask, 'inf_subtype_prob'] = 0.0
    
    # let's also add the probability for each subject of being each subtype
    for i in range(nb_subtype):
        # # TODO Why that
        # if i == 0:
        #    output_data.loc[:,'prob_s%s'%(i+1)] = 0
        output_data.loc[:,'prob_s%s'%(i+1)] = prob_subtype[:,i]

    return output_data