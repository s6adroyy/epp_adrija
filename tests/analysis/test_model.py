"""Tests for the main regression model."""

import numpy as np
import pandas as pd
import pytest
from epp_adrija.analysis.model import run_dd_regression


def test_run_dd_regression_coefficients():
    # create a sample dataset
    np.random.seed(42)
    data = pd.DataFrame({
        "treatment_var": np.random.choice([0, 1], size=100),
        "outcome_var": np.random.normal(size=100),
        "cov_1": np.random.normal(size=100),
        "cov_2": np.random.normal(size=100),
        "year_hgsch_entry": np.random.choice([2010, 2011], size=100),
        "State": np.random.choice(["NY", "CA"], size=100)
    })

    # define the true coefficients
    true_coefficients = {
        "Intercept": 0.0301,
        "cov_1": -0.009,
        "cov_2": 0.0105,
        "C(year_hgsch_entry)[T.2011]": 0.0456,
        "treatment_var": -0.1446,
    }

    # add noise to the outcome variable to make the coefficients slightly different from the true values
    data["outcome_var"] += np.random.normal(scale=0.1, size=100)

    # run the function
    result = run_dd_regression(data, "outcome_var", ["cov_1", "cov_2" , "treatment_var"])
    
    print("Expected keys:", list(true_coefficients.keys()))
    print("Actual keys:", list(result.params.keys()))

    # check that the coefficients match the expected values within a tolerance
    for name, expected in true_coefficients.items():
        print(name, "Expected:", expected, "Actual:", result.params[name])
        assert np.isclose(result.params[name], expected, rtol=0.3)
       