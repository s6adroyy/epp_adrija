"""Tasks running the core analyses."""


import pandas as pd
import pytask
import csv
import pickle
from pathlib import Path
from epp_adrija.analysis.model import run_dd_regression, mechanism_regression
from epp_adrija.config import BLD, SRC

# from tabulate import tabulate

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
@pytask.mark.produces(BLD / "python" / "models" / "reg_model_gym1.csv")
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


@pytask.mark.depends_on(
    {
        "scripts": ["model.py"],
        "data": BLD / "python" / "data" / "final_df.dta",
        "data_info": SRC / "data_management" / "data_info.yaml",
    },
)
@pytask.mark.wip16
@pytask.mark.produces(BLD / "python" / "models" / "reg_model_gym2.pkl")
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
    with open(produces, "wb") as f:
        pickle.dump(reg_model_gym, f)


# @pytask.mark.produces(BLD / "python" / "models" / "model.pickle")
# def task_fit_model_python(depends_on, produces):


# for group in GROUPS:

# @pytask.mark.depends_on(
# },
# @pytask.mark.task(id=group, kwargs=kwargs)
# def task_predict_python(depends_on, group, produces):
@pytask.mark.depends_on(
    {
        "scripts": ["model.py"],
        "data": BLD / "python" / "data" / "mechanisms.dta",
        "data_info": SRC / "data_management" / "data_info.yaml",
    },
)
@pytask.mark.wip13
@pytask.mark.produces(BLD / "python" / "models" / "mechanism.pkl")
def task_mechanism_vw(depends_on, produces):
    """Fit a two way fixed effect regression model (Python version)."""
    data = pd.read_stata(depends_on["data"])
    outcome_vars = ["volunteer_work", "mental_health", "school_representative"]
    covariates = [
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
    ]
    reg_results = mechanism_regression(data, outcome_vars, covariates)
    with open(produces, "wb") as f:
        pickle.dump(reg_results, f)
    # results_mechanism = reg_results.summary().tables[1]
    with open(produces, "rb") as f:
        data = pickle.load(f)
    print(data)


import csv
import pickle


@pytask.mark.depends_on(
    {
        "scripts": ["model.py"],
        "data": BLD / "python" / "data" / "mechanisms.dta",
        "data_info": SRC / "data_management" / "data_info.yaml",
    },
)
@pytask.mark.wip15
@pytask.mark.produces(BLD / "python" / "models" / "mechanism_voluteer_work.pkl")
def task_mechanism_regression(depends_on, produces):
    """Fit a two way fixed effect regression model (Python version)."""
    data = pd.read_stata(depends_on["data"])
    mech_vw = mechanism_regression(
        data,
        outcome_vars="volunteer_work",
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
    with open(produces, "wb") as f:
        pickle.dump(mech_vw, f)


@pytask.mark.depends_on(
    {
        "scripts": ["model.py"],
        "data": BLD / "python" / "data" / "mechanisms.dta",
        "data_info": SRC / "data_management" / "data_info.yaml",
    },
)
@pytask.mark.wip19
@pytask.mark.produces(BLD / "python" / "models" / "mechanism_school_rep.pkl")
def task_mechanism_regression(depends_on, produces):
    """Fit a two way fixed effect regression model (Python version)."""
    data = pd.read_stata(depends_on["data"])
    mech_sch_rep = mechanism_regression(
        data,
        outcome_vars="school_representative",
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
    with open(produces, "wb") as f:
        pickle.dump(mech_sch_rep, f)


@pytask.mark.depends_on(
    {
        "scripts": ["model.py"],
        "data": BLD / "python" / "data" / "mechanisms.dta",
        "data_info": SRC / "data_management" / "data_info.yaml",
    },
)
@pytask.mark.wip21
@pytask.mark.produces(BLD / "python" / "models" / "mechanism_mental_health.pkl")
def task_mechanism_regression(depends_on, produces):
    """Fit a two way fixed effect regression model (Python version)."""
    data = pd.read_stata(depends_on["data"])
    mech_mental_hlth = mechanism_regression(
        data,
        outcome_vars="mental_health",
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
    with open(produces, "wb") as f:
        pickle.dump(mech_mental_hlth, f)


@pytask.mark.depends_on(
    {
        "scripts": ["model.py"],
        "data": BLD / "python" / "data" / "mechanisms.dta",
        "data_info": SRC / "data_management" / "data_info.yaml",
    },
)
@pytask.mark.wip23
@pytask.mark.produces(BLD / "python" / "models" / "mechanism_sport_active.pkl")
def task_mechanism_regression(depends_on, produces):
    """Fit a two way fixed effect regression model (Python version)."""
    data = pd.read_stata(depends_on["data"])
    mech_sport_act = mechanism_regression(
        data,
        outcome_vars="sport_active",
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
    with open(produces, "wb") as f:
        pickle.dump(mech_sport_act, f)


# @pytask.mark.produces(BLD / "python" / "models" / "mechanism_voluteer_work.tex")
# def task_mechanism_regression(depends_on, produces):
#     """Fit a two way fixed effect regression model (Python version)."""
#     data = pd.read_stata(depends_on["data"])

#     outcome_vars = ['volunteer_work']
#     covariates = [
#         "Treat",
#         "Age",
#         "female",
#         "rural",
#         "East",
#         "low_performing",
#         "highest_educ_hh",
#         "migration_backgrnd",
#         "work_father",
#         "work_mother",
#         "reli_hh",
#     ]
#     reg_results = mechanism_regression(data, outcome_vars, covariates)

#     coefficients = []
#     pvalues = []
#     tvalues = []
#     for i in range(len(reg_results)):
#         result = reg_results[i]
#         coefficients.append(result.params)
#         pvalues.append(result.pvalues)
#         tvalues.append(result.tvalues)

#     df = pd.DataFrame(
#         {
#             "coefficients": coefficients,
#             "p-values": pvalues,
#             "t-values": tvalues,
#         },
#         index=outcome_vars,
#     )

#     # Save the DataFrame to a LaTeX file
#     with open(produces, "w") as f:
#         f.write(df.to_latex())
