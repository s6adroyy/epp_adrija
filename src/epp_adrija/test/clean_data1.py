"""Function(s) for cleaning the data set(s)."""


import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=DeprecationWarning)


def process_data(input_dataframe):
    input_dataframe["syear"] = pd.to_numeric(input_dataframe["syear"], errors="coerce")
    input_dataframe["jl0233"] = pd.to_numeric(
        input_dataframe["jl0233"],
        errors="coerce",
    )
    # Filter and get rows where syear in 2006 to 2018
    input_dataframe = input_dataframe.loc[
        ((input_dataframe["syear"] >= 2006) & (input_dataframe["syear"] <= 2018))
    ].copy()

    migration_info = [15, 16, 17, 18, 19]

    input_dataframe = input_dataframe.loc[
        ~input_dataframe["psample"].isin(migration_info)
    ]

    input_dataframe["Age"] = input_dataframe["syear"] - input_dataframe["jl0233"]
    input_dataframe = input_dataframe.loc[(input_dataframe["Age"] == 17)].copy()
    input_dataframe = input_dataframe.loc[(input_dataframe["jl0164"] == 2)].copy()

    output_dataframe = input_dataframe.copy()

    return output_dataframe


def filter_based_on_school(df1):
    # Create empty column for Gymnasium

    # Set gymnasium to 1 or 0
    df1["Gymnasium"] = 0
    df1.loc[
        ((df1["jl0125_v3"] == 3) | ((df1["jl0125_v3"] == 6) & (df1["jl0127_h"] == 4))),
        "Gymnasium",
    ] = 1

    df1 = df1.loc[(df1["Gymnasium"] == 1)].copy()
    # Create empty col of high school entry for replacing the NaN values instead of dropping
    df1["year_hgsch_entry"] = df1["bet3year"].fillna(0).values
    df1 = df1.rename(columns={"bula_h": "State"})

    # list of states where the year_hgsch_entry is birth year(gebjahr) + 12 years
    states_list1 = [11.0, 12.0, 13.0]
    df1.loc[
        (~df1["State"].isin(states_list1))
        & ((df1["year_hgsch_entry"] == 0) | (df1["year_hgsch_entry"] == -2)),
        "year_hgsch_entry",
    ] = (
        df1["gebjahr"] + 10
    )

    df1.loc[
        (df1["State"].isin(states_list1))
        & ((df1["year_hgsch_entry"] == 0) | (df1["year_hgsch_entry"] == -2)),
        "year_hgsch_entry",
    ] = (
        df1["gebjahr"] + 12
    )
    # Remove states where reform has not been implemented state-wide
    states_list2 = [7.0]

    # Exclude students from Hesse who entered high school in 2004 or 2005 when schools operated under both schemes (G8 and G9)
    df1 = df1.loc[
        ~(
            (df1["State"] == 6.0)
            & ((df1["year_hgsch_entry"] == 2004) | (df1["year_hgsch_entry"] == 2005))
        )
    ]

    df1 = df1.loc[~df1["State"].isin(states_list2)]

    return df1


def create_treatment(df1):
    df1 = df1.copy()
    df1["Treat"] = df1.apply(lambda _: 0, axis=1)

    State_list = [14.0, 16.0]
    df1.loc[(df1["State"].isin(State_list)), "Treat"] = 1  # Always treated
    df1.loc[((df1["State"] == 1) & (df1["year_hgsch_entry"] >= 2008)), "Treat"] = 1
    df1.loc[((df1["State"] == 2) & (df1["year_hgsch_entry"] >= 2002)), "Treat"] = 1
    df1.loc[((df1["State"] == 3) & (df1["year_hgsch_entry"] >= 2003)), "Treat"] = 1
    df1.loc[((df1["State"] == 4) & (df1["year_hgsch_entry"] >= 2004)), "Treat"] = 1
    df1.loc[((df1["State"] == 5) & (df1["year_hgsch_entry"] >= 2005)), "Treat"] = 1
    df1.loc[((df1["State"] == 6) & (df1["year_hgsch_entry"] >= 2006)), "Treat"] = 1
    df1.loc[((df1["State"] == 8) & (df1["year_hgsch_entry"] >= 2004)), "Treat"] = 1
    df1.loc[((df1["State"] == 9) & (df1["year_hgsch_entry"] >= 2003)), "Treat"] = 1
    df1.loc[((df1["State"] == 10) & (df1["year_hgsch_entry"] >= 2001)), "Treat"] = 1
    df1.loc[((df1["State"] == 11) & (df1["year_hgsch_entry"] >= 2006)), "Treat"] = 1
    df1.loc[((df1["State"] == 12) & (df1["year_hgsch_entry"] >= 2006)), "Treat"] = 1
    df1.loc[((df1["State"] == 13) & (df1["year_hgsch_entry"] >= 2002)), "Treat"] = 1
    df1.loc[((df1["State"] == 15) & (df1["year_hgsch_entry"] >= 1999)), "Treat"] = 1
    # df1['Gymnasium'] = pd.Categorical(df1['Gymnasium']
    return df1


def rename_variables(df1):
    trust_variables = ["jl0361", "jl0362", "jl0363"]
    for var in trust_variables:
        df1.loc[~(df1[var] == -1)]
    df1.rename(
        columns={
            "jl0361": "trust",
            "jl0362": "rely_none",
            "jl0363": "distrust_stranger",
        },
        inplace=True,
    )

    df1["rely_someone"] = df1.apply(lambda _: 0, axis=1)
    df1.loc[(~df1["rely_none"] <= 0), "rely_someone"] = 8 - df1["rely_none"]
    df1["trust_stranger"] = df1.apply(lambda _: 0, axis=1)
    df1.loc[(~df1["distrust_stranger"] <= 0), "trust_stranger"] = (
        8 - df1["distrust_stranger"]
    )
    df1["trust_var"] = df1.apply(lambda _: 0, axis=1)
    df1.loc[(~df1["trust"] <= 0), "trust_var"] = (
        df1["trust"] + df1["rely_someone"] + df1["trust_stranger"]
    )
    df1["std_trust_var"] = (df1["trust_var"] - df1["trust_var"].mean()) / df1[
        "trust_var"
    ].std()
    df1["std_trust"] = (df1["trust"] - df1["trust"].mean()) / df1["trust"].std()

    return df1


def rename_independent_var(df2):
    df2["female"] = df2.apply(lambda _: 0, axis=1)
    df2.loc[(df2["sex"] == 2), "female"] = 1

    df2["rural"] = df2.apply(lambda _: " ", axis=1)
    df2 = df2.loc[~(df2["jl0272"] == 4)]
    df2.loc[(df2["jl0272"] == 4), "rural"] = 1
    df2.loc[~(df2["jl0272"] == 4), "rural"] = 0

    df2["East"] = df2.apply(lambda _: 0, axis=1)
    East_states = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    df2.loc[(df2["State"].isin(East_states)), "East"] = 1

    df2["migration_backgrnd"] = df2.apply(lambda _: 0, axis=1)
    df2.loc[~(df2["migback"] == -1), "migration_backgrnd"] = 1

    df2["low_performing"] = df2.apply(lambda _: " ", axis=1)
    df2.loc[(~df2["jl0151"] == -1)]
    df2.loc[(df2["jl0151"] == 3), "low_performing"] = 0
    Non_gym_school = [1, 2, 4]
    df2.loc[(df2["jl0151"].isin(Non_gym_school)), "low_performing"] = 1

    Nan_fsedu = [-5.0, -1.0]
    df2 = df2.loc[~df2["fsedu"].isin(Nan_fsedu)]
    df2 = df2.rename(columns={"fsedu": "father_educ1"})
    df2.loc[~(df2["father_educ1"] == 4.0), "father_educ1"] = 0
    df2.loc[(df2["father_educ1"] == 4.0), "father_educ1"] = 1

    Nan_msedu = [-5.0, -1.0]
    df2 = df2.loc[~df2["msedu"].isin(Nan_msedu)]
    df2 = df2.rename(columns={"msedu": "mother_educ1"})
    df2.loc[~(df2["mother_educ1"] == 4.0), "mother_educ1"] = 0
    df2.loc[(df2["mother_educ1"] == 4.0), "mother_educ1"] = 1

    Nan_fprofedu = [-5.0, -1.0]
    df2.loc[~df2["fprofedu"].isin([Nan_fprofedu])]
    df2 = df2.rename(columns={"fprofedu": "father_educ2"})
    df2.loc[
        ((df2["father_educ2"] < 28.0) | (df2["father_educ2"] > 32.0)),
        "father_educ2",
    ] = 0
    df2.loc[
        ~((df2["father_educ2"] < 28.0) | (df2["father_educ2"] > 32.0)),
        "father_educ2",
    ] = 1

    Nan_mprofedu = [-5.0, -1.0]
    df2.loc[~df2["mprofedu"].isin([Nan_mprofedu])]
    df2 = df2.rename(columns={"mprofedu": "mother_educ2"})
    df2.loc[
        ((df2["mother_educ2"] < 28.0) | (df2["mother_educ2"] > 32.0)),
        "mother_educ2",
    ] = 0
    df2.loc[
        ~((df2["mother_educ2"] < 28.0) | (df2["mother_educ2"] > 32.0)),
        "mother_educ2",
    ] = 1

    df2["highest_educ_hh"] = df2.apply(lambda _: 0, axis=1)
    df2.loc[
        (
            (df2["father_educ1"] == 1)
            | (df2["mother_educ1"] == 1)
            | (df2["father_educ2"] == 1)
            | (df2["mother_educ2"] == 1)
        ),
        "highest_educ_hh",
    ] = 1

    df2 = df2.loc[~(df2["freli"] == -1.0)]
    df2 = df2.rename(columns={"freli": "father_reli"})
    df2.loc[
        ((df2["father_reli"] == 1.0) | (df2["father_reli"] == 2.0)),
        "father_reli",
    ] = 1
    df2.loc[
        ~((df2["father_reli"] == 1.0) | (df2["father_reli"] == 2.0)),
        "father_reli",
    ] = 0

    df2 = df2.loc[~(df2["mreli"] == -1.0)]
    df2 = df2.rename(columns={"mreli": "mother_reli"})
    df2.loc[
        ((df2["mother_reli"] == 1.0) | (df2["mother_reli"] == 2.0)),
        "mother_reli",
    ] = 1
    df2.loc[
        ~((df2["mother_reli"] == 1.0) | (df2["mother_reli"] == 2.0)),
        "mother_reli",
    ] = 0

    df2["reli_hh"] = df2.apply(lambda _: 1, axis=1)
    df2.loc[((df2["mother_reli"] == 0) & (df2["mother_reli"] == 0)), "reli_hh"] = 0

    df2 = df2.loc[~(df2["mprofstat"] == -1.0)]
    df2 = df2.rename(columns={"mprofstat": "work_mother"})
    df2.loc[
        ~(df2["work_mother"] < 10.0) & (df2["work_mother"] > 15.0),
        "work_mother",
    ] = 1
    df2.loc[
        (df2["work_mother"] >= 10.0) & (df2["work_mother"] <= 15.0),
        "work_mother",
    ] = 0

    Nan_fegp88 = [-1.0, -2.0]
    df2 = df2.loc[~df2["fegp88"].isin(Nan_fegp88)]
    df2 = df2.rename(columns={"fegp88": "work_father"})
    df2.loc[(df2["work_father"] < 8.0), "work_father"] = 0
    df2.loc[(df2["work_father"] >= 8.0), "work_father"] = 1

    Nan_living1 = [-2.0, -3.0]
    df2 = df2.loc[~df2["living1"].isin(Nan_living1)]
    df2 = df2.rename(columns={"living1": "single_parent"})
    df2.loc[(df2["single_parent"] < 15.0), "single_parent"] = 1
    df2.loc[(df2["single_parent"] == 15.0), "single_parent"] = 0

    df2 = df2.loc[~(df2["jl0176_h"] == -1)]
    df2 = df2.rename(columns={"jl0176_h": "migration_classmate"})
    df2.loc[(df2["migration_classmate"] < 6), "migration_classmate"] = 1
    df2.loc[(df2["migration_classmate"] == 6), "migration_classmate"] = 0

    return df2


def run_dd_regression(data, treatment_var, outcome_var, covariates, clustering_var):
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
    formula = f"{outcome_var} ~ {treatment_var} + {'+'.join(covariates)} + C(year_hgsch_entry)"

    # Run the regression using statsmodels
    smf.ols(formula=formula, data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data[clustering_var]},
    )

    # Return the regression results


def event_study(df3):
    df3["treat_year"] = df3.apply(lambda _: 0, axis=1)
    treat_year_dict = {
        "[2] Hamburg": 2002,
        "[3] Niedersachsen": 2003,
        "[4] Bremen": 2004,
        "[5] Nordrhein-Westfalen": 1,
        "[8] Baden-Wuerttemberg": 2004,
        "[9] Bayern": 2003,
        "[10] Saarland": 2001,
        "[11] Berlin": 2006,
        "[12] Brandenburg": 2006,
        "[13] Mecklenburg-Vorpommern": 2002,
        "[15] Sachsen-Anhalt": 1999,
    }
    for states in treat_year_dict:
        df3.loc[(df3["State"].isin([states])), "treat_year"] = treat_year_dict[states]
    return df3


def lead_lag(df4):
    df4["t"] = df4["year_hgsch_entry"] - df4["treat_year"]
    df4["lag0"] = df4["t"] == 0
    for i in range(1, 8):
        df4["lag" + str(i)] = df4["t"] == -i
        df4["lead" + str(i)] = df4["t"] == i
    return df4


def plot_event_study(formula, data):
    event_study_formula = smf.ols(formula, data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["State"]},
    )

    lags = [
        "lead7[T.True]",
        "lead6[T.True]",
        "lead7[T.True]",
        "lead4[T.True]",
        "lead3[T.True]",
        "lead2[T.True]",
        "lead1[T.True]",
    ]
    leads = [
        "lag0[T.True]",
        "lag1[T.True]",
        "lag2[T.True]",
        "lag3[T.True]",
        "lag4[T.True]",
        "lag5[T.True]",
        "lag6[T.True]",
        "lag7[T.True]",
    ]

    leadslags_plot = pd.DataFrame(
        {
            "sd": np.concatenate(
                [
                    np.sqrt(
                        np.diag(event_study_formula.cov_params().loc[leads][leads]),
                    ),
                    np.array([0]),
                    np.sqrt(np.diag(event_study_formula.cov_params().loc[lags][lags])),
                ],
            ),
            "mean": np.concatenate(
                [
                    event_study_formula.params[leads],
                    np.array([0]),
                    event_study_formula.params[lags],
                ],
            ),
            "label": np.arange(-8, 8),
        },
    )

    leadslags_plot["lb"] = leadslags_plot["mean"] - leadslags_plot["sd"] * 1.96
    leadslags_plot["ub"] = leadslags_plot["mean"] + leadslags_plot["sd"] * 1.96

    plot = (
        p.ggplot(leadslags_plot, p.aes(x="label", y="mean", ymin="lb", ymax="ub"))
        + p.geom_hline(yintercept=0.0769, color="red")
        + p.geom_pointrange()
        + p.theme_minimal()
        + p.xlab("Years before and after policy")
        + p.ylab("Event-Study")
        + p.geom_hline(yintercept=0, linetype="dashed")
        + p.geom_vline(xintercept=0, linetype="dashed")
    )
    plt.savefig("leadslags_plot.png")
    return plot.draw()


# "+".join(xvar))
#     "+".join(xvar_non_gym))

# dd_reg = smf.ols(dd_formula,
# data = Ind_Rename).fit(cov_type="cluster",cov_kwds={'groups':Ind_Rename['State']})
# dd_reg_non_gym = smf.ols(dd_formula_non_gym,
#             data = Ind_Rename_non_gym).fit(cov_type="cluster",cov_kwds={'groups':Ind_Rename_non_gym['State']})


# event_study_formula = smf.ols(formula,
# data = Ind_Rename).fit(cov_type='cluster', cov_kwds={'groups':Ind_Rename['State']})
# event_study_formula_non_gym = smf.ols(formula,
#             data = Ind_Rename_non_gym).fit(cov_type='cluster', cov_kwds={'groups':Ind_Rename_non_gym['State']})


#'label': np.arange(-8,8)})

#     'label': np.arange(-8,8)})


# p.ggplot(leadslags_plot_non_gym, p.aes(x = 'label', y = 'mean',
#     p.geom_hline(yintercept = 0,
#     p.geom_vline(xintercept = 0,
#              linetype = "dashed")

# p.ggplot(leadslags_plot, p.aes(x = 'label', y = 'mean',
# p.geom_hline(yintercept = 0,
# p.geom_vline(xintercept = 0,
# linetype = "dashed")

# p.ggplot(leadslags_plot_non_gym, p.aes(x = 'label', y = 'mean',
#     p.geom_hline(yintercept = 0,
#     p.geom_vline(xintercept = 0,
#              linetype = "dashed")


# processed_dataframe.shape

# for school in ['gymnasium','not_gymnasium']:


"""
results = run_dd_regression(
    data=Ind_Rename,
    treatment_var="Treat",
    outcome_var="std_trust_var",
    covariates=[
        "Age",
        "female",
        "rural",
        "East",
        "low_performing",
        "highest_educ_hh",
        "migration_backgrnd",
        "father_blue_collar",
        "m_work",
        "reli_hh",
        "single_parent",
    ],
    clustering_var="State",
)
"""
