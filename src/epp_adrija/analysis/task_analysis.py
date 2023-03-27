"""Tasks running the core analyses."""


import pandas as pd
import pytask
import pickle
from epp_adrija.analysis.model import run_dd_regression, mechanism_regression, placebo_regression
from epp_adrija.config import BLD, SRC

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
@pytask.mark.produces(BLD / "python" / "models" / "main_regression.pkl")
def task_run_dd_regression(depends_on, produces):
    """Fit a two way fixed effect regression model (Python version)."""
    data = pd.read_stata(depends_on["data"])
    reg_model_gym = run_dd_regression(
        data,
        outcome_var="std_trust_var",
        covariates=[
            "Treat",
            "female",
            "East",
            "low_performing",
            "highest_educ_hh",
            "work_father",
            "work_mother",
            "reli_hh",
            "single_parent",
            "migration_classmate"
        ],
    )
    with open(produces, "wb") as f:
        pickle.dump(reg_model_gym, f)


@pytask.mark.depends_on(
    {
        "scripts": ["model.py"],
        "data": BLD / "python" / "data" / "mechanisms.dta",
        "data_info": SRC / "data_management" / "data_info.yaml",
    },
)
@pytask.mark.wip15
@pytask.mark.produces(BLD / "python" / "models" / "mechanism_voluteer_work.pkl")
def task_mech_vw_regression(depends_on, produces):
    """Fit a two way fixed effect regression model (Python version)."""
    data = pd.read_stata(depends_on["data"])
    mech_vw = mechanism_regression(
        data,
        alpha = 2.0,
        outcome_vars="volunteer_work",
        covariates=[
            "Treat",
            "female",
            "East",
            "low_performing",
            "highest_educ_hh",
            "work_father",
            "work_mother",
            "reli_hh",
            "single_parent",
            "migration_classmate",
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
@pytask.mark.wip21
@pytask.mark.produces(BLD / "python" / "models" / "mechanism_mental_health.pkl")
def task_mech_mental_hlth_regression(depends_on, produces):
    """Fit a two way fixed effect regression model (Python version)."""
    data = pd.read_stata(depends_on["data"])
    mech_mental_hlth = mechanism_regression(
        data,
        alpha = 2.0,
        outcome_vars="mental_health",
        covariates=[
            "Treat",
            "female",
            "East",
            "low_performing",
            "highest_educ_hh",
            "work_father",
            "work_mother",
            "reli_hh", 
            "single_parent",
            "migration_classmate",
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
@pytask.mark.wip34
@pytask.mark.produces(BLD / "python" / "models" / "mechanism_sport_active.pkl")
def task_mech_sch_sport_regression(depends_on, produces):
    """Fit a two way fixed effect regression model (Python version)."""
    data = pd.read_stata(depends_on["data"])
    mech_sport_act = mechanism_regression(
        data,
        alpha = 2.0, 
        outcome_vars="sport_active",
        covariates=[
            "Treat",
            "female",
            "East",
            "low_performing",
            "highest_educ_hh",
            "work_father",
            "work_mother",
            "reli_hh",
            "single_parent", 
            "migration_classmate"
        ],
    )
    with open(produces, "wb") as f:
        pickle.dump(mech_sport_act, f)


@pytask.mark.depends_on(
    {
        "scripts": ["model.py"],
        "data": BLD / "python" / "data" / "mechanisms.dta",
        "data_info": SRC / "data_management" / "data_info.yaml",
    },
)
@pytask.mark.wip25
@pytask.mark.produces(BLD / "python" / "models" / "mechanism_sch_group.pkl")
def task_mech_sch_group_regression(depends_on, produces):
    """Fit a two way fixed effect regression model (Python version)."""
    data = pd.read_stata(depends_on["data"])
    mech_sch_group = mechanism_regression(
        data,
        alpha = 2.0, 
        outcome_vars="some_school_group",
        covariates=[
            "Treat",
            "female",
            "East",
            "low_performing",
            "highest_educ_hh",
            "work_father",
            "work_mother",
            "reli_hh",
            "single_parent", 
            "migration_classmate"
        ],
    )
    with open(produces, "wb") as f:
        pickle.dump(mech_sch_group, f)


@pytask.mark.depends_on(
    {
        "scripts": ["model.py"],
        "data": BLD / "python" / "data" / "final_df.dta",
        "data_info": SRC / "data_management" / "data_info.yaml",
    },
)
@pytask.mark.wip27
@pytask.mark.produces(BLD / "python" / "models" / "placebo_regression.pkl")
def task_placebo_regression(depends_on, produces):
    """Fit a two way fixed effect regression model (Python version)."""
    data = pd.read_stata(depends_on["data"])
    placebo_reg = placebo_regression(
        data,
        outcome_var="trust_var",
        covariates=[
            "Treat",
            "female",
            "East",
            "low_performing",
            "highest_educ_hh",
            "work_father",
            "work_mother",
            "reli_hh",
            "single_parent",
            "migration_classmate"
        ],
    )
    with open(produces, "wb") as f:
        pickle.dump(placebo_reg, f)


