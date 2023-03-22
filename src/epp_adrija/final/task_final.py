# """Tasks running the results formatting (tables, figures)."""

import pandas as pd
import pytask
from epp_adrija.config import BLD, SRC
from epp_adrija.analysis.model import load_model

# from epp_adrija.config import BLD, GROUPS, SRC
from epp_adrija.final.plot import plot_event_study
from epp_adrija.analysis.model import mechanism_regression
import plotnine as p
from plotnine import *
import matplotlib
from pathlib import Path

# from epp_adrija.utilities import read_yaml

# for group in GROUPS:

#     kwargs = {
#         "group": group,
#         "depends_on": {"predictions": BLD / "python" / "predictions" / f"{group}.csv"},
#         "produces": BLD / "python" / "figures" / f"smoking_by_{group}.png",
#     }

#     @pytask.mark.depends_on(
#         {
#             "data_info": SRC / "data_management" / "data_info.yaml",
#             "data": BLD / "python" / "data" / "data_clean.csv",
#         },
#     )
#     @pytask.mark.task(id=group, kwargs=kwargs)
#     def task_plot_results_by_age_python(depends_on, group, produces):
#         """Plot the regression results by age (Python version)."""
#         data_info = read_yaml(depends_on["data_info"])
#         data = pd.read_csv(depends_on["data"])
#         predictions = pd.read_csv(depends_on["predictions"])
#         fig = plot_regression_by_age(data, data_info, predictions, group)
#         fig.write_image(produces)


# @pytask.mark.depends_on(BLD / "python" / "models" / "model.pickle")
# @pytask.mark.produces(BLD / "python" / "tables" / "estimation_results.tex")
# def task_create_results_table_python(depends_on, produces):
#     """Store a table in LaTeX format with the estimation results (Python version)."""
#     #model = load_model(depends_on)
#     table = model.summary().as_latex()
#     with open(produces, "w") as f:
#         f.writelines(table)


@pytask.mark.depends_on(
    {
        "eventstudy_df": BLD / "python" / "data" / "eventstudy.dta",
    },
)
@pytask.mark.wip9
@pytask.mark.produces(BLD / "python" / "figures" / "event_study_gym.png")
def task_event_study_plot(depends_on, produces):
    input = pd.read_stata(depends_on["eventstudy_df"])
    event_study_gym = plot_event_study(
        input,
        outcome_var="std_trust_var",
        covariates=[
            "lead1",
            "lead2",
            "lead3",
            "lead4",
            "lead5",
            "lead6",
            "lead7",
            "lag1",
            "lag2",
            "lag3",
            "lag4",
        ],
    )
    # event_study_gym.save(produces, event_study_gym, device="png")
    filename = str(produces)
    ggsave(filename=filename, plot=event_study_gym, dpi=300)

    # ggplot.save(filename , event_study_gym)
    # ggsave(event_study_gym, filename)
    # event_study_gym.ggsave(produces)
    #


# @pytask.mark.depends_on(BLD / "python" / "models" / "mechanism.pkl")
# @pytask.mark.wip14

# @pytask.mark.produces(BLD / "python" / "tables" / "mechanism_results.tex")
# def task_mechanism_result_table_python(depends_on, produces):
# #     """Store a table in LaTeX format with the estimation results (Python version)."""
#     model_mech = load_model(depends_on)
#     table_mech = model_mech.summary().as_latex()
#     with open(produces, "w") as f:
#          f.writelines(table_mech)

# import csv
# import pickle
# @pytask.mark.depends_on(
#     {
#         "scripts": ["model.py"],
#         "data": BLD / "python" / "data" / "mechanisms.dta",
#         "data_info": SRC / "data_management" / "data_info.yaml",
#     },
# )
# @pytask.mark.wip15
# @pytask.mark.produces(BLD / "python" / "models" / "mechanism.tex")
# def task_mechanism_regression(depends_on, produces):
#     """Fit a two way fixed effect regression model (Python version)."""
#     data = pd.read_stata(depends_on["data"])
#     outcome_vars = ['volunteer_work', 'mental_health', 'school_representative']
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

#     # Write the regression results to a pickle file
#     # with open(produces[0], 'wb') as f:
#     #     pickle.dump(reg_results, f)

#     # Write the regression results to a .tex file
#     with open(produces, 'w') as f:
#         f.write(reg_results.to_latex())


@pytask.mark.depends_on(BLD / "python" / "models" / "reg_model_gym2.pkl")
@pytask.mark.wip17
@pytask.mark.produces(BLD / "python" / "tables" / "reg_model_gym2.tex")
def task_reg_result_table_python(depends_on, produces):
    #     """Store a table in LaTeX format with the estimation results (Python version)."""
    model_mech = load_model(depends_on)
    table_mech = model_mech.summary().as_latex()
    with open(produces, "w") as f:
        f.writelines(table_mech)


@pytask.mark.depends_on(BLD / "python" / "models" / "mechanism_voluteer_work.pkl")
@pytask.mark.wip18
@pytask.mark.produces(BLD / "python" / "tables" / "mechanism_volunteer_work.tex")
def task_mech_vw_python(depends_on, produces):
    #     """Store a table in LaTeX format with the estimation results (Python version)."""
    model_mech_vw = load_model(depends_on)
    table_mech_vw = model_mech_vw.summary().as_latex()
    with open(produces, "w") as f:
        f.writelines(table_mech_vw)


@pytask.mark.depends_on(BLD / "python" / "models" / "mechanism_school_rep.pkl")
@pytask.mark.wip20
@pytask.mark.produces(BLD / "python" / "tables" / "mechanism_school_rep.tex")
def task_mech_schrep_python(depends_on, produces):
    #     """Store a table in LaTeX format with the estimation results (Python version)."""
    model_sch_rep = load_model(depends_on)
    table_sch_rep = model_sch_rep.summary().as_latex()
    with open(produces, "w") as f:
        f.writelines(table_sch_rep)


@pytask.mark.depends_on(BLD / "python" / "models" / "mechanism_mental_health.pkl")
@pytask.mark.wip22
@pytask.mark.produces(BLD / "python" / "tables" / "mechanism_mental_health.tex")
def task_mech_mentalhlth_python(depends_on, produces):
    #     """Store a table in LaTeX format with the estimation results (Python version)."""
    model_mental_hlth = load_model(depends_on)
    table_mental_hlth = model_mental_hlth.summary().as_latex()
    with open(produces, "w") as f:
        f.writelines(table_mental_hlth)


@pytask.mark.depends_on(BLD / "python" / "models" / "mechanism_sport_active.pkl")
@pytask.mark.wip24
@pytask.mark.produces(BLD / "python" / "tables" / "mechanism_sport_active.tex")
def task_mech_sch_act_python(depends_on, produces):
    #     """Store a table in LaTeX format with the estimation results (Python version)."""
    model_sport_act = load_model(depends_on)
    table_sport_act = model_sport_act.summary().as_latex()
    with open(produces, "w") as f:
        f.writelines(table_sport_act)
