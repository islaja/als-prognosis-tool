"""
Unit tests for compute_progression_rate.

These tests validate that:
- Missing DPR is computed as (48 - ALSFRS) / symptom_duration
- An existing DPR value is preserved rather than recomputed
- Zero or missing symptom duration leaves DPR as NaN (not computed)
"""

import numpy as np
import pandas as pd
from als_prognosis.pipeline.run_pipeline import compute_progression_rate


def test_compute_progression_rate_calculates_missing_dpr():
    df = pd.DataFrame({
        "id": ["P01"],
        "dpr": [np.nan],
        "alsfrs_total": [40],
        "symptom_duration_months": [2],
    })

    result = compute_progression_rate(df)

    assert result.loc[0, "dpr"] == 4.0


def test_compute_progression_rate_preserves_existing_dpr():
    df = pd.DataFrame({
        "id": ["P01"],
        "dpr": [99.0],
        "alsfrs_total": [40],
        "symptom_duration_months": [2],
    })

    result = compute_progression_rate(df)

    assert result.loc[0, "dpr"] == 99.0


def test_compute_progression_rate_leaves_dpr_missing_for_zero_or_missing_duration():
    df = pd.DataFrame({
        "id": ["P01", "P02"],
        "dpr": [np.nan, np.nan],
        "alsfrs_total": [40, 40],
        "symptom_duration_months": [0, np.nan],
    })

    result = compute_progression_rate(df)

    assert pd.isna(result.loc[0, "dpr"])
    assert pd.isna(result.loc[1, "dpr"])
