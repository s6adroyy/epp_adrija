"""Tasks for managing the data."""


import pandas as pd
import pytask

from epp_adrija.config import BLD, SRC
from epp_adrija.data_management.clean_data import (
    create_treatment,
    filter_based_on_school,
    process_data,
    rename_independent_var,
    rename_variables,
    mechanism,
    event_study,
)


# @pytask.mark.depends_on(
# },
# @pytask.mark.produces(BLD / "python" / "data" / "data_clean.csv")
# def task_copy_data(depends_on, produces):


# @pytask.mark.produces(BLD / "data" / "raw_dataframe.dta")
# def task_copy_data(depends_on, produces):


@pytask.mark.depends_on(
    {
        "df": SRC / "data" / "merge_original_youth_data.dta",
    },
)
@pytask.mark.wip1
@pytask.mark.produces(BLD / "python" / "data" / "raw_dataframe.dta")
def task_copy_original_data(depends_on, produces):
    original_dataframe = pd.read_stata(depends_on["df"], convert_categoricals=False)
    raw_dataframe = original_dataframe.copy()
    raw_dataframe.to_stata(produces)


# @pytask.mark.depends_on(
# @pytask.mark.wip
# @pytask.mark.produces(BLD /"python"/"data"/"abc.dta")
# def task_process_data(depends_on, produces):


@pytask.mark.depends_on(
    {
        "dataframe": BLD / "python" / "data" / "raw_dataframe.dta",
    },
)
@pytask.mark.wip4
@pytask.mark.produces(BLD / "python" / "data" / "final_df.dta")
def task_process_data(depends_on, produces):
    input_file = pd.read_stata(depends_on["dataframe"])
    processed_dataframe1 = process_data(input_file)
    processed_data2 = filter_based_on_school(processed_dataframe1)
    processed_data3 = create_treatment(processed_data2)
    processed_data4 = rename_variables(processed_data3)
    processed_data5 = rename_independent_var(processed_data4)
    final_df = processed_data5.copy()
    final_df.to_stata(produces)


@pytask.mark.depends_on(
    {
        "dataframe2": BLD / "python" / "data" / "final_df.dta",
    },
)
@pytask.mark.wip6
@pytask.mark.produces(BLD / "python" / "data" / "output_df.csv")
def task_process_data_csv(depends_on, produces):
    output_df = pd.read_stata(depends_on["dataframe2"])
    output_df.to_csv(produces)


@pytask.mark.depends_on(
    {
        "dataframe3": BLD / "python" / "data" / "final_df.dta",
    },
)
@pytask.mark.wip8
@pytask.mark.produces(BLD / "python" / "data" / "eventstudy.dta")
def task_event_study(depends_on, produces):
    input_df = pd.read_stata(depends_on["dataframe3"])
    eventstudy = event_study(input_df)
    eventstudy.to_stata(produces)


@pytask.mark.depends_on(
    {
        "dataframe4": BLD / "python" / "data" / "final_df.dta",
    },
)
@pytask.mark.wip12
@pytask.mark.produces(BLD / "python" / "data" / "mechanisms.dta")
def task_meachanisms(depends_on, produces):
    input_df4 = pd.read_stata(depends_on["dataframe4"])
    mechanisms = mechanism(input_df4)
    mechanisms.to_stata(produces)


# @pytask.mark.depends_on(
#     },
# @pytask.mark.wip2
# @pytask.mark.produces(BLD / "python" / "data" / "final_df.csv")
# def task_final_data(depends_on, produces):


# @pytask.mark.depends_on(
#     },
# @pytask.mark.wip3
# @pytask.mark.produces(BLD / "python" / "data" / "event_study_df.csv")
# def task_doing_event_study_data(depends_on, produces):


# @pytask.mark.depends_on(
# @pytask.mark.produces(BLD / "data" / "chs" / "my_df.dta")
# def combined_processing(depends_on, produces):


# @pytask.mark.depends_on(
# },
# @pytask.mark.produces(BLD / "python" / "data" / "data_clean.csv")
# def task_clean_data_python(depends_on, produces):
