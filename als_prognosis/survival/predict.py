"""
Survival Analysis and Visualization for ALS SuStaIn Pipeline.

This module provides an ensemble inference engine for Coxnet survival models
and visualization utilities to compare individual patient trajectories against
a reference cohort.

Main Functions:
    - infer_curves_from_bootstrap_models: Aggregates ensemble predictions.
    - plot_inferred_mean_curves: Generates the background reference landscape.
    - plot_coxnet_prediction_over_references: Overlays patient data on the cohort.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Dict, List, Optional, Optional, Tuple, Union, Any

def infer_curves_from_bootstrap_models(
    subj_data: pd.Series, 
    bootstrap_models: List[Dict[str, Any]], 
    common_times: np.ndarray
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Infers survival probabilities by aggregating a bootstrap ensemble of Coxnet models.

    This function performs inference by iterating through an ensemble of models, 
    aligning features, and interpolating survival functions to a unified time axis. 
    It calculates the mean trajectory, 95% confidence intervals, and predicts the 
    median survival time.

    Args:
        subj_data (pd.Series): Feature vector for a single subject.
        bootstrap_models (list): List of dictionaries containing 'model' (fitted 
            estimator) and 'feature_names' (list of strings).
        common_times (np.array): Unified time points (in months) for interpolation.

    Returns:
        dict: A dictionary containing:
            - 'mean': Inferred mean survival probabilities (np.array).
            - 'median_survival_time': Predicted time at 0.5 probability (float).
            - 'lower': 2.5th percentile CI bound (np.array).
            - 'upper': 97.5th percentile CI bound (np.array).
            - 'times': The input common_times (np.array).
    """

    subject_curves = []
    for entry in bootstrap_models:
        m = entry['model']
        f_names = entry['feature_names']
        
        # Align features for this specific model
        X_sub = subj_data[f_names].to_frame().T
        
        # Predict survival function
        surv_func = m.predict_survival_function(X_sub)
        
        # Interpolate to common time points
        subject_curves.append(surv_func[0](common_times))
    
    # Convert list to a 2D NumPy array (Shape: 100 iterations x 120 time points)
    subject_curves = np.array(subject_curves)

    # Average the bootstrap curves
    mean_curve = np.mean(subject_curves, axis=0)
    
    # Calculate 95% Confidence Interval using percentiles
    lower_ci = np.percentile(subject_curves, 2.5, axis=0)
    upper_ci = np.percentile(subject_curves, 97.5, axis=0)

    # Get the time at probability of survival = 0.5 (median survival time) for the mean curve
    median_survival_time = np.interp(0.5, mean_curve[::-1], common_times[::-1])

    # Return as a dictionary for easy access in your tool
    return {
        'mean': mean_curve,
        'median_survival_time': median_survival_time,
        'lower': lower_ci,
        'upper': upper_ci,
        'times': common_times
    }

def plot_inferred_mean_curves(
    reference_curves_library: Dict[str, Any], 
    b_color_by_subtype: bool = False, 
    subtype_color_map: Optional[Dict[int, Tuple[float, float, float]]] = None, 
    legend_elements: Optional[List[Line2D]] = None, 
    legend_title: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Generates a background visualization of inferred mean survival curves for a cohort.

    Used to create a reference landscape of disease progression. Curves can be 
    colored by their inferred SuStaIn subtype to visualize phenotypic variance.

    Args:
        reference_curves_library (dict): Library containing 'times' and a list of 
            'subjects' data (including 'mean_curve' and 'ml_subtype').
        b_color_by_subtype (bool): Whether to color background curves by subtype.
        subtype_color_map (dict, optional): Mapping of subtype integers to colors.
        legend_elements (list, optional): Matplotlib handles for custom legend.
        legend_title (str, optional): Title for the color legend.

    Returns:
        Tuple[Figure, Axes]: The Matplotlib figure and axes objects for the plot.
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    common_times = reference_curves_library['times']

    for subject in reference_curves_library['subjects']:
        # Extract data
        mean_y = subject['mean_curve']
        subtype = int(subject['ml_subtype'])

        # Determine color
        color = 'gray'
        if b_color_by_subtype and subtype_color_map:
            color = subtype_color_map.get(subtype, 'gray')

        ax.plot(common_times, mean_y, color=color, alpha=0.8, linewidth=1, label=None)
        
    if legend_elements is not None:
        ax.legend(handles=legend_elements, loc='upper right', title=legend_title, frameon=False)

    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, common_times.max())
    ax.set_xlabel('Months')
    ax.set_ylabel('Survival Probability')
    ax.set_title('Cohort Reference Survival Curves')

    fig.tight_layout()

    return fig, ax

def plot_coxnet_prediction_over_references( 
    patient_mean_curve_result: Dict[str, Any], 
    reference_library: Dict[str, Any], 
    nb_subtypes: int, 
    title_suffix: str = ''
) -> Tuple[Figure, Axes]:
    """
    Visualizes a patient's predicted survival trajectory against a reference cohort.

    This high-level function layers the specific patient's predicted curve and 
    median survival time over the inferred background curves of the reference 
    population. It automatically generates a color palette and legend for 
    SuStaIn subtypes.

    Args:
        patient_mean_curve_result (dict): Output from `infer_curves_from_bootstrap_models`.
        reference_library (dict): Data structure containing the cohort's reference curves.
        nb_subtypes (int): Number of SuStaIn subtypes (excluding S0).
        title_suffix (str): Additional text to append to the plot title.

    Returns:
        Tuple[Figure, Axes]: The Matplotlib figure and axes objects.
    """

    # nb subtype + S0
    nb_total_subtype = nb_subtypes+1
    base_palette = sns.color_palette('husl', nb_total_subtype)
    subtype_color_map = {i: base_palette[i] for i in range(nb_total_subtype)}
    legend_title = "CALSNIC SuStaIn Subtype"
    
    legend_elements = []
    for i in range(nb_total_subtype):
        legend_elements.append(Line2D([i], [i], color=base_palette[i], lw=2, label=f'S{i}'))

    # Plot References curves
    fig, ax = plot_inferred_mean_curves(
        reference_library, 
        b_color_by_subtype=True, 
        subtype_color_map=subtype_color_map, 
        legend_elements=legend_elements, 
        legend_title=legend_title,
    )

    # Clearer to use black for the new patient than the subtype color. 
    # new_patient_color = subtype_color_map.get(patient_subtype, 'black')
    new_patient_color = 'black'
    
    # Plot the patient's predicted curve on top of the reference curves
    ax.plot(patient_mean_curve_result['times'], patient_mean_curve_result['mean'], 
            color=new_patient_color, linewidth=2, label='New Patient')

    # Plot the mean_survival_time as a vertical dashed line
    ax.axvline(x=patient_mean_curve_result['median_survival_time'], color=new_patient_color, linestyle='--')
    ax.set_title(f"{title_suffix}\nPredicted Median Survival: {patient_mean_curve_result['median_survival_time']:.1f} months", fontsize=14)

    return fig, ax

