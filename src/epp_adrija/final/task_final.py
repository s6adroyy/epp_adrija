# """Tasks running the results formatting (tables, figures)."""

import pandas as pd
import pytask
from epp_adrija.config import BLD
from epp_adrija.analysis.model import load_model
from epp_adrija.final.plot import plot_event_study,descriptive_stats_plot
#from epp_adrija.analysis.model import mechanism_regression
#import plotnine as p
from plotnine import *
#import matplotlib
#from pathlib import Path

# from epp_adrija.utilities import read_yaml


@pytask.mark.depends_on(
    {
        "eventstudy_df": BLD / "python" / "data" / "eventstudy.dta",
    },
)
@pytask.mark.wip31
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
    filename = str(produces)
    ggsave(filename=filename, plot=event_study_gym, dpi=300)


@pytask.mark.depends_on(
    {
        "path": BLD / "python" / "data" / "eventstudy.dta",
    },
)
@pytask.mark.wip51
@pytask.mark.produces(BLD / "python" / "figures" / "descriptive_stats.png")
def task_descriptive_plot(depends_on, produces):
    data = pd.read_stata(depends_on["path"])
    fig = descriptive_stats_plot(data)
    fig.savefig(produces)

@pytask.mark.depends_on(BLD / "python" / "models" / "main_regression.pkl")
@pytask.mark.wip17
@pytask.mark.produces(BLD / "python" / "tables" / "main_regression.tex")
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
@pytask.mark.wip37
@pytask.mark.produces(BLD / "python" / "tables" / "mechanism_sport_active.tex")
def task_mech_schsport_python(depends_on, produces):
    #     """Store a table in LaTeX format with the estimation results (Python version)."""
    model_sport_act = load_model(depends_on)
    table_sport_act = model_sport_act.summary().as_latex()
    with open(produces, "w") as f:
        f.writelines(table_sport_act)


@pytask.mark.depends_on(BLD / "python" / "models" / "mechanism_sch_group.pkl")
@pytask.mark.wip26
@pytask.mark.produces(BLD / "python" / "tables" / "mechanism_sch_group.tex")
def task_mech_sch_act_python(depends_on, produces):
    #     """Store a table in LaTeX format with the estimation results (Python version)."""
    model_sch_group = load_model(depends_on)
    table_sch_group = model_sch_group.summary().as_latex()
    with open(produces, "w") as f:
        f.writelines(table_sch_group)


@pytask.mark.depends_on(BLD / "python" / "models" / "placebo_regression.pkl")
@pytask.mark.wip28
@pytask.mark.produces(BLD / "python" / "tables" / "placebo_regression.tex")
def task_mech_sch_act_python(depends_on, produces):
    #     """Store a table in LaTeX format with the estimation results (Python version)."""
    model_placebo = load_model(depends_on)
    table_placebo = model_placebo.summary().as_latex()
    with open(produces, "w") as f:
        f.writelines(table_placebo)