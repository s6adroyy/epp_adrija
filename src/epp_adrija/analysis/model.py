"""Functions for fitting the regression model."""
import pandas as pd
import statsmodels.formula.api as smf

from epp_adrija.utilities import read_yaml

'''
def fit_logit_model(data, data_info, model_type):
    """Fit a logit model to data.

    Args:
        data (pandas.DataFrame): The data set.
        data_info (dict): Information on data set stored in data_info.yaml. The
            following keys can be accessed:
            - 'outcome': Name of dependent variable column in data
            - 'outcome_numerical': Name to be given to the numerical version of outcome
            - 'columns_to_drop': Names of columns that are dropped in data cleaning step
            - 'categorical_columns': Names of columns that are converted to categorical
            - 'column_rename_mapping': Old and new names of columns to be renamend,
                stored in a dictionary with design: {'old_name': 'new_name'}
            - 'url': URL to data set
        model_type (str): What model to build for the linear relationship of the logit
            model. Currently implemented:
            - 'linear': Numerical covariates enter the regression linearly, and
            categorical covariates are expanded to dummy variables.

    Returns:
        statsmodels.base.model.Results: The fitted model.

    """
    outcome_name = data_info["outcome"]
    outcome_name_numerical = data_info["outcome_numerical"]
    feature_names = list(set(data.columns) - {outcome_name, outcome_name_numerical})

    if model_type == "linear":
        # smf.logit expects the binary outcome to be numerical
        formula = f"{outcome_name_numerical} ~ " + " + ".join(feature_names)
    else:
        message = "Only 'linear' model_type is supported right now."
        raise ValueError(message)

    return smf.logit(formula, data=data).fit()


def load_model(path):
    """Load statsmodels model.

    Args:
        path (str or pathlib.Path): Path to model file.

    Returns:
        statsmodels.base.model.Results: The stored model.

    """
    return load_pickle(path)
'''

# def run_dd_regression(data, treatment_var, data_info, outcome_var, covariates, clustering_var):
def run_dd_regression(data, outcome_var, covariates):
    """Runs a difference-in-differences regression on the given data using the specified variables and clustering variable.

    Parameters:
        data (pandas.DataFrame): The data to use for the regression analysis.
        treatment_var (str): The name of the treatment variable.
        outcome_var (str): The name of the outcome variable.
        covariates (list): A list of covariate variable names to include in the regression.
        clustering_var (str): The name of the variable to use for clustering standard errors.

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


# Return the regression results
data_info = read_yaml(
    r"C:\Users\LENOVO\epp_adrija\src\epp_adrija\data_management\data_info.yaml",
)
data = pd.read_stata(
    r"C:\Users\LENOVO\epp_adrija\bld\python\data\final_df.dta",
    convert_categoricals=False,
)
abc = run_dd_regression(
    data,
    outcome_var="std_trust_var",
    covariates=[
        "Treat",
        "Age",
        "female",
        "rural",
        "East",
        "low_performing",
        "highest_educ_hh",
        "migration_backgrnd",
        "work_father",
        "work_mother",
        "reli_hh",
    ],
)
