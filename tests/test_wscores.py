"""
Unit tests for compute_wscores.

These tests validate that:
- W-scores are computed correctly as (observed - expected) / sigma
- ROIs present in the model bundle but absent from the patient data are skipped
- A missing 'scanner' column is defaulted to 'unknown' before prediction
- A missing model bundle raises ValueError

All models are lightweight stubs; no real serialized model artifacts are used.
"""

import pandas as pd
import pytest
from als_prognosis.preprocessing.wscores import compute_wscores


class FakeModel:
    """Stub model exposing a `.predict()` returning a fixed value."""

    def __init__(self, value):
        self.value = value
        self.last_X = None

    def predict(self, X):
        self.last_X = X
        return pd.Series([self.value])


def test_compute_wscores_basic():
    patient_data = pd.Series({
        "filename": "P01",
        "age": 65,
        "sex": "M",
        "scanner": "siemens",
        "roi_1": "10.0",
        "roi_2": 5.0,
    })
    not_numerical = ["filename", "sex", "scanner"]
    bundle = {
        "roi_1": {"model": FakeModel(8.0), "sigma": 2.0},
        "roi_2": {"model": FakeModel(5.0), "sigma": 1.0},
    }

    result = compute_wscores(patient_data, bundle, not_numerical=not_numerical)

    expected = pd.Series({"roi_1": 1.0, "roi_2": 0.0})
    pd.testing.assert_series_equal(result.sort_index(), expected.sort_index())


def test_compute_wscores_skips_roi_missing_from_patient_data():
    patient_data = pd.Series({
        "filename": "P01",
        "age": 65,
        "sex": "M",
        "scanner": "siemens",
        "roi_1": 10.0,
        "roi_2": 5.0,
    })
    not_numerical = ["filename", "sex", "scanner"]
    bundle = {
        "roi_1": {"model": FakeModel(8.0), "sigma": 2.0},
        "roi_2": {"model": FakeModel(5.0), "sigma": 1.0},
        "roi_3": {"model": FakeModel(1.0), "sigma": 1.0},
    }

    result = compute_wscores(patient_data, bundle, not_numerical=not_numerical)

    assert "roi_3" not in result.index
    assert set(result.index) == {"roi_1", "roi_2"}


def test_compute_wscores_defaults_missing_scanner_to_unknown():
    patient_data = pd.Series({
        "filename": "P01",
        "age": 65,
        "sex": "M",
        "roi_1": 10.0,
    })
    not_numerical = ["filename", "sex", "scanner"]
    model = FakeModel(8.0)
    bundle = {"roi_1": {"model": model, "sigma": 2.0}}

    result = compute_wscores(patient_data, bundle, not_numerical=not_numerical)

    assert model.last_X is not None
    assert model.last_X["scanner"].iloc[0] == "unknown"
    assert result["roi_1"] == pytest.approx(1.0)


def test_compute_wscores_raises_when_bundle_is_none():
    patient_data = pd.Series({"filename": "P01", "roi_1": 10.0})

    with pytest.raises(ValueError):
        compute_wscores(patient_data, None, not_numerical=["filename"])
