"""
Unit tests for infer_curves_from_bootstrap_models.

These tests validate the ensembling math over a bootstrap of survival
models, using lightweight fake models with fixed, hand-computed curves
instead of real Coxnet artifacts:
- mean survival curve is the elementwise average across bootstrap models
- lower/upper bounds are the 2.5th/97.5th percentiles across models
- median survival time is the time where the mean curve crosses 0.5,
  found by linear interpolation
"""

import numpy as np
import pandas as pd
import pytest
from als_prognosis.survival.predict import infer_curves_from_bootstrap_models


class FakeCoxnetModel:
    """Stub model returning a fixed survival curve regardless of input times."""

    def __init__(self, curve_values):
        self.curve_values = np.array(curve_values, dtype=float)

    def predict_survival_function(self, X):
        return [lambda times: self.curve_values]


def _bootstrap_models():
    return [
        {"model": FakeCoxnetModel([1.0, 0.8, 0.6, 0.4]), "feature_names": ["f1", "f2"]},
        {"model": FakeCoxnetModel([1.0, 0.9, 0.7, 0.5]), "feature_names": ["f1", "f2"]},
        {"model": FakeCoxnetModel([1.0, 0.7, 0.5, 0.3]), "feature_names": ["f1", "f2"]},
    ]


def _subj_data():
    return pd.Series({"f1": 10.0, "f2": 20.0})


def _common_times():
    return np.array([0.0, 1.0, 2.0, 3.0])


def test_infer_curves_mean_is_elementwise_average():
    result = infer_curves_from_bootstrap_models(_subj_data(), _bootstrap_models(), _common_times())

    # (1.0+1.0+1.0)/3, (0.8+0.9+0.7)/3, (0.6+0.7+0.5)/3, (0.4+0.5+0.3)/3
    assert np.allclose(result["mean"], [1.0, 0.8, 0.6, 0.4])


def test_infer_curves_confidence_interval_is_percentile_based():
    result = infer_curves_from_bootstrap_models(_subj_data(), _bootstrap_models(), _common_times())

    # Hand-computed via linear-interpolation percentile formula (numpy default):
    # t0: [1.0, 1.0, 1.0] -> 1.0 / 1.0
    # t1: sorted [0.7, 0.8, 0.9] -> low=0.7+0.05*(0.8-0.7)=0.705, high=0.8+0.95*(0.9-0.8)=0.895
    # t2: sorted [0.5, 0.6, 0.7] -> low=0.5+0.05*(0.6-0.5)=0.505, high=0.6+0.95*(0.7-0.6)=0.695
    # t3: sorted [0.3, 0.4, 0.5] -> low=0.3+0.05*(0.4-0.3)=0.305, high=0.4+0.95*(0.5-0.4)=0.495
    assert result["lower"] == pytest.approx([1.0, 0.705, 0.505, 0.305])
    assert result["upper"] == pytest.approx([1.0, 0.895, 0.695, 0.495])


def test_infer_curves_median_survival_time_via_interpolation():
    result = infer_curves_from_bootstrap_models(_subj_data(), _bootstrap_models(), _common_times())

    # mean_curve = [1.0, 0.8, 0.6, 0.4] at times [0, 1, 2, 3].
    # np.interp(0.5, [0.4, 0.6, 0.8, 1.0], [3, 2, 1, 0]):
    # 0.5 is halfway between 0.4 and 0.6 -> halfway between times 3 and 2 -> 2.5
    assert result["median_survival_time"] == pytest.approx(2.5)
