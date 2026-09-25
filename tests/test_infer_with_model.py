"""
Unit tests for infer_with_model.

These tests validate the post-processing logic applied to raw SuStaIn
outputs, using a lightweight fake model instead of a real SuStaIn model:
- 0-based to 1-based subtype re-indexing
- Stage-0 normalization of inferred_subtype / prob_inferred_subtype
- prob_sN column construction matching the number of subtypes
- Returning the patient's row, not the augmented jitter row
"""

import numpy as np
import pandas as pd
from als_prognosis.progression.predict import infer_with_model


class FakeSustainModel:
    """Stub model returning fixed outputs for both input rows (patient + jitter)."""

    def __init__(self, ml_subtype, prob_ml_subtype, ml_stage, prob_ml_stage, prob_subtype):
        self._outputs = (
            np.array(ml_subtype),
            np.array(prob_ml_subtype),
            np.array(ml_stage),
            np.array(prob_ml_stage),
            np.array(prob_subtype),
            None,
            None,
        )

    def subtype_and_stage_individuals_newData(self, input_stack, samples_sequence, samples_f, n_samples):
        return self._outputs


def _data():
    return pd.Series({"roi_1": 1.0, "roi_2": 2.0})


def test_infer_with_model_normal_case_shifts_subtype_to_one_based():
    model = FakeSustainModel(
        ml_subtype=[0, 0],
        prob_ml_subtype=[0.9, 0.9],
        ml_stage=[5, 5],
        prob_ml_stage=[0.8, 0.8],
        prob_subtype=[[0.9, 0.1], [0.9, 0.1]],
    )

    result = infer_with_model(model, None, None, _data())

    assert result["inferred_subtype"] == 1
    assert result["prob_inferred_subtype"] == 0.9
    assert result["inferred_stage"] == 5
    assert result["prob_inferred_stage"] == 0.8
    assert result["prob_s1"] == 0.9
    assert result["prob_s2"] == 0.1


def test_infer_with_model_stage_zero_forces_subtype_zero_but_keeps_stage_probability():
    model = FakeSustainModel(
        ml_subtype=[2, 2],
        prob_ml_subtype=[0.7, 0.7],
        ml_stage=[0, 0],
        prob_ml_stage=[0.6, 0.6],
        prob_subtype=[[0.1, 0.2, 0.7], [0.1, 0.2, 0.7]],
    )

    result = infer_with_model(model, None, None, _data())

    assert result["inferred_subtype"] == 0
    assert result["prob_inferred_subtype"] == 0.0
    assert result["inferred_stage"] == 0
    assert result["prob_inferred_stage"] == 0.6
    assert result["prob_s1"] == 0.1
    assert result["prob_s2"] == 0.2
    assert result["prob_s3"] == 0.7


def test_infer_with_model_prob_sN_columns_match_subtype_count():
    model = FakeSustainModel(
        ml_subtype=[1, 1],
        prob_ml_subtype=[0.5, 0.5],
        ml_stage=[3, 3],
        prob_ml_stage=[0.4, 0.4],
        prob_subtype=[[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]],
    )

    result = infer_with_model(model, None, None, _data())

    prob_s_columns = sorted(c for c in result.index if c.startswith("prob_s"))
    assert prob_s_columns == ["prob_s1", "prob_s2", "prob_s3"]
    assert result["prob_s1"] == 0.2
    assert result["prob_s2"] == 0.3
    assert result["prob_s3"] == 0.5


def test_infer_with_model_returns_patient_row_not_jitter_row():
    model = FakeSustainModel(
        ml_subtype=[0, 1],
        prob_ml_subtype=[0.9, 0.4],
        ml_stage=[5, 9],
        prob_ml_stage=[0.8, 0.3],
        prob_subtype=[[0.9, 0.1], [0.4, 0.6]],
    )

    result = infer_with_model(model, None, None, _data())

    assert result["inferred_stage"] == 5
    assert result["prob_inferred_stage"] == 0.8
    assert result["inferred_subtype"] == 1
