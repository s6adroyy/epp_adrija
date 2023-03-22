"""Functions plotting results."""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotnine as p
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from plotnine import *

# def plot_regression_by_age(data, data_info, predictions, group):
#     """Plot regression results by age.

#     Args:
#         data (pandas.DataFrame): The data set.
#         data_info (dict): Information on data set stored in data_info.yaml. The
#             following keys can be accessed:
#             - 'outcome': Name of dependent variable column in data
#             - 'outcome_numerical': Name to be given to the numerical version of outcome
#             - 'columns_to_drop': Names of columns that are dropped in data cleaning step
#             - 'categorical_columns': Names of columns that are converted to categorical
#             - 'column_rename_mapping': Old and new names of columns to be renamend,
#                 stored in a dictionary with design: {'old_name': 'new_name'}
#             - 'url': URL to data set
#         predictions (pandas.DataFrame): Model predictions for different age values.
#         group (str): Categorical column in data set. We create predictions for each
#             unique value in column data[group]. Cannot be 'age' or 'smoke'.

#     Returns:
#         plotly.graph_objects.Figure: The figure.

#     """
#     plot_data = predictions.melt(
#         id_vars="age",
#         value_vars=predictions.columns,
#         value_name="prediction",
#         var_name=group,
#     )

#     outcomes = data[data_info["outcome_numerical"]]

#     fig = px.line(
#         plot_data,
#         x="age",
#         y="prediction",
#         color=group,
#         labels={"age": "Age", "prediction": "Probability of Smoking"},
#     )

#     fig.add_traces(
#         go.Scatter(
#             x=data["age"],
#             y=outcomes,
#             mode="markers",
#             marker_color="black",
#             marker_opacity=0.1,
#             name="Data",
#         ),
#     )
#     return fig


def plot_event_study(data, outcome_var, covariates):
    formula = f"{outcome_var} ~  {'+'.join(covariates)} + C(year_hgsch_entry)"
    event_study_formula = smf.ols(formula, data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["State"]},
    )
    lags = [
        "lag4",
        "lag3",
        "lag2",
        "lag1",
    ]
    leads = [
        "lead1",
        "lead2",
        "lead3",
        "lead4",
        "lead5",
        "lead6",
        "lead7",
    ]
    print(leads)
    print(lags)
    leadslags_plot = pd.DataFrame(
        {
            "sd": np.concatenate(
                [
                    np.sqrt(
                        np.diag(event_study_formula.cov_params().loc[lags][lags]),
                    ),
                    np.array([0]),
                    np.sqrt(
                        np.diag(event_study_formula.cov_params().loc[leads][leads])
                    ),
                ],
            ),
            "mean": np.concatenate(
                [
                    event_study_formula.params[lags],
                    np.array([0]),
                    event_study_formula.params[leads],
                ],
            ),
            "label": np.arange(-4, 8),
        },
    )
    print(event_study_formula)
    leadslags_plot["lb"] = leadslags_plot["mean"] - leadslags_plot["sd"] * 1.96
    leadslags_plot["ub"] = leadslags_plot["mean"] + leadslags_plot["sd"] * 1.96

    plot = (
        p.ggplot(leadslags_plot, p.aes(x="label", y="mean", ymin="lb", ymax="ub"))
        + p.geom_hline(yintercept=0.0769, color="red")
        + p.geom_pointrange()
        + p.theme_minimal()
        + p.theme_bw()
        + p.xlab("Years before and after policy")
        + p.ylab("Event-Study")
        + p.geom_hline(yintercept=0, linetype="dashed")
        + p.geom_vline(xintercept=0, linetype="dashed")
    )
    # plot.save("leadslags_plot4.png")
    # plot.save(filename="leadslags_plot1.jpg", dpi= 1000)

    # ggplot.save("leadslags_plot4.png")
    return plot
    # ggsave("leadslags_plot4.png", plot)
