"""Tasks running the core analyses."""


import pandas as pd
import pytask

from epp_adrija.analysis.model import run_dd_regression
from epp_adrija.config import BLD, SRC


# @pytask.mark.depends_on(
# },
@pytask.mark.depends_on(
    {
        "scripts": ["model.py"],
        "data": BLD / "python" / "data" / "final_df.dta",
        "data_info": SRC / "data_management" / "data_info.yaml",
    },
)
@pytask.mark.wip7
@pytask.mark.produces(BLD / "python" / "models" / "reg_model_gym.csv")
def task_run_dd_regression(depends_on, produces):
    """Fit a two way fixed effect regression model (Python version)."""
    data = pd.read_stata(depends_on["data"])
    reg_model_gym = run_dd_regression(
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
    results_table = reg_model_gym.summary().tables[1]

    # Convert table data to DataFrame
    reg_results_df = pd.DataFrame(results_table.data[1:], columns=results_table.data[0])
    reg_results_df.to_csv(produces)
    # with open(produces[1], 'w') as f:
    # with redirect_stdout(f):


# @pytask.mark.produces(BLD / "python" / "models" / "model.pickle")
# def task_fit_model_python(depends_on, produces):


# for group in GROUPS:

# @pytask.mark.depends_on(
# },
# @pytask.mark.task(id=group, kwargs=kwargs)
# def task_predict_python(depends_on, group, produces):
