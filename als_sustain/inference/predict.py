"""Model loader and inference wrapper.
This wrapper expects trained models to be pickled and to expose a method
that accepts numpy arrays and returns (subtype, stage) or similar.
Adapt to your SuStaIn implementation.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

def load_model(model_path: Path):
    with model_path.open("rb") as f:
        if model_path.suffix == '.pkl' or model_path.suffix == '.pickle':
            print("Loaded model using pickle.")
            obj = pickle.load(f)
        else:
            print("Loaded model using joblib.")
            obj = joblib.load(f)  
  
    return obj

def load_pickle_info(pickle_path: Path):
    try:
        pk = pd.read_pickle(pickle_path)
        samples_sequence = pk["samples_sequence"]
        samples_f = pk["samples_f"]
        return samples_sequence, samples_f
    except Exception as e:
        raise ValueError(f"Error loading pickle info from {pickle_path}: {e}")

def predict_with_model(model, samples_sequence, samples_f, data: pd.Series) -> pd.Series:
    """Call the model's inference method. Modify to match your model API.
    Expected return: (subtype_label, stage_value)
    """
    
    #output_data = pd.Series()
    # it seems like sustain can predict on only one subject at a time. We will
    # create a fake second subject by duplicating the first one with some noise
    # and we will discard the second subject prediction later.
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


    output_data['ml_subtype'] = ml_subtype
    output_data['prob_ml_subtype'] = prob_ml_subtype
    output_data['ml_stage'] = ml_stage
    output_data['prob_ml_stage'] = prob_ml_stage

    # make current subtypes (0, 1, 2) 1 and 2, 3 instead
    #output_data.loc[:, ml_subtype_col] = (output_data[ml_subtype_col].to_numpy() + 1).astype('Int64')
    output_data['ml_subtype'] = (output_data['ml_subtype'].astype("Int64") + 1)

    # Define a mask for stage 0
    stage0_mask = output_data['ml_stage'] == 0
    # Assign subtype = 0 for stage 0 (meaning: no valid subtype)
    output_data.loc[stage0_mask, 'ml_subtype'] = 0

    # set stage as int
    output_data['ml_stage'] =  output_data['ml_stage'].astype('Int64')

    # set ml_subtype as categorical
    col_values = output_data['ml_subtype']
    categories = sorted(pd.Series(col_values.dropna().unique()))
    output_data['ml_subtype'] = pd.Categorical(
                col_values,
                categories=categories,
                ordered=False
            )

    # Invalidate subtype probability for stage 0
    output_data.loc[stage0_mask, 'prob_ml_subtype'] = 0.0
    
    # let's also add the probability for each subject of being each subtype
    for i in range(prob_subtype.shape[1]):
        # # TODO Why that
        # if i == 0:
        #    output_data.loc[:,'prob_s%s'%(i+1)] = 0
        output_data.loc[:,'prob_s%s'%(i+1)] = prob_subtype[:,i]

    # remove fake row 
    output_data = pd.Series(output_data.iloc[0])
    print(output_data)
   
    return output_data