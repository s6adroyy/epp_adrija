"""Functions for fitting the regression model."""

import numpy as np
import statsmodels.formula.api as smf
from statsmodels.iolib.smpickle import load_pickle
from statsmodels.miscmodels.ordinal_model import OrderedModel


def load_model(path):
    """Load statsmodels model.

    Args:
        path (str or pathlib.Path): Path to model file.

    Returns:
        statsmodels.base.model.Results: The stored model.

    """
    return load_pickle(path)
  


def run_dd_regression(data, outcome_var, covariates):
    """Runs a difference-in-differences regression on the given data using the specified variables and clustering variable.

    Parameters:
        data (pandas.DataFrame): The data to use for the regression analysis.
        outcome_var (str): The name of the outcome variable.
        covariates (list): A list of covariate variable names to include in the regression.
        
    Returns:
        statsmodels.regression.linear_model.RegressionResultsWrapper: A summary of the regression results.

    """
    # Create the formula for the regression
    formula = f"{outcome_var} ~  {'+'.join(covariates)} + C(year_hgsch_entry)"

    # Run the regression using statsmodels
    reg = smf.ols(formula=formula, data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["State"]},
    )

    return reg

def placebo_regression (data, outcome_var, covariates):
    """Runs a difference-in-differences regression on the given data using the specified variables and clustering variable.

    Parameters:
        data (pandas.DataFrame): The data to use for the regression analysis.
        outcome_var (str): The name of the outcome variable.
        covariates (list): A list of covariate variable names to include in the regression.
        

    Returns:
        statsmodels.regression.linear_model.RegressionResultsWrapper: A summary of the regression results.

    """
    mod_log = OrderedModel(data[outcome_var],
                        data[covariates],
                        distr='logit')

    res_log = mod_log.fit(method='bfgs', disp=False)
    return res_log 

def mechanism_regression(data, outcome_vars, covariates, alpha):
    """Runs a difference-in-differences regression on the given data using the specified variables and clustering variable.

    Parameters:
        data (pandas.DataFrame): The data to use for the regression analysis.
        outcome_vars (list): A list of outcome variable names to include in the regression.
        covariates (list): A list of covariate variable names to include in the regression.
        alpha (numeric): A value for the penalty benchmark
    Returns:
        dict: A dictionary of regression results, with the outcome variable names as keys and the regression results as values.

    """
    data = data.dropna(subset=covariates)
    # Create the formula for the regression
    formula = f"{outcome_vars} ~ {'+'.join(covariates)}"

    # Run the regression using statsmodels
    reg = smf.ols(formula=formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data["State"]})

    #Check for singularity and regularize if necessary
    if reg.condition_number < 1e-8:
        X = data[covariates].values
        y = data[outcome_vars].values
        penalty = alpha * np.identity(X.shape[1])
        X_ridge = np.dot(X.T, X) + penalty
        X_inv = np.linalg.inv(X_ridge)
        coef = np.dot(np.dot(X_inv, X.T), y)
        result = reg.__class__(model=reg.model,
                                params=coef,
                                normalized_cov_params=X_inv,
                                cov_params_default=X_inv,
                                scale=reg.scale)
    else:
        result = reg

    return result


