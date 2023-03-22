import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# pip install pydrive
import plotnine as p
import statsmodels.formula.api as smf


def process_data(input_dataframe):
    input_dataframe["syear"] = input_dataframe["syear"].astype(int)
    input_dataframe["syear"] = input_dataframe["syear"].fillna(0)

    # Filter and get rows where syear in 2006 to 2018
    input_dataframe = input_dataframe.loc[
        ((input_dataframe["syear"] >= 2006) & (input_dataframe["syear"] <= 2018))
    ]

    input_dataframe["jl0233"] = input_dataframe["jl0233"].astype(int)
    migration_info = [
        "[15] M1 2013 Migration (1995-2010)",
        "[16] M2 2015 Migration (2009-2013)",
        "[17] M3 2016 Flucht (2013-2015)",
        "[18] M4 2016 Flucht/Familie (2013-2015)",
        "[19] M5 2017 Flucht (2013-2016)",
    ]
    input_dataframe = input_dataframe.loc[
        ~input_dataframe["psample"].isin(migration_info)
    ]
    input_dataframe["Age"] = input_dataframe["syear"] - input_dataframe["jl0233"]
    input_dataframe = input_dataframe.loc[(input_dataframe["Age"] == 17)]
    input_dataframe = input_dataframe.loc[
        (input_dataframe["jl0164"].isin(["[2] Nein"]))
    ]

    output_dataframe = input_dataframe.copy()
    return output_dataframe


def filter_based_on_school(df1, flag):
    # Create empty column for Gymnasium
    df1["Gymnasium"] = df1.apply(lambda _: " ", axis=1)

    # Set gymnasium to 1 or 0

    df1.loc[
        (
            (df1["jl0125_v3"] == "[3] Ja Gymnasium")
            | (
                (df1["jl0125_v3"] == "[6] Nein")
                & (df1["jl0127_h"] == "[4] Fachhochshulreife/Abitur")
            )
        ),
        "Gymnasium",
    ] = 1

    non_gym_list = [
        "[1] Ja Hauptschule",
        "[2] Ja Realschule",
        "[4] Ja Gesamtschule/andere",
        "[5] Ja berufliche Schule",
    ]
    df1.loc[
        (
            (df1["jl0125_v3"].isin(non_gym_list))
            | (
                (df1["jl0125_v3"] == "[6] Nein")
                & (df1["jl0127_h"] != "[4] Fachhochshulreife/Abitur")
            )
        ),
        "Gymnasium",
    ] = 0

    if flag == "gymnasium":
        # Select only those rows which have Gymnasium = 1
        df1 = df1.loc[(df1["Gymnasium"] == 1)]
    elif flag == "not_gymnasium":
        df1 = df1.loc[(df1["Gymnasium"] == 0)]
    else:
        raise Exception("Flag can either be 'gymnasium' or 'not_gymnasium'")

    # Create empty col
    df1["year_hgsch_entry"] = df1.apply(lambda _: " ", axis=1)

    df1["year_hgsch_entry"] = df1["bet3year"].fillna(0).values

    df1 = df1.rename(columns={"bula_h": "State"})
    states_list1 = ["[11] Berlin", "[12] Brandenburg", "[13] Mecklenburg-Vorpommern"]
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

    # Remove problematic states from data
    states_list2 = ["[7] Rheinland-Pfalz,Saarland"]

    df1 = df1.loc[
        ~(
            (df1["State"] == "[6] Hessen")
            & ((df1["year_hgsch_entry"] == 2004) | (df1["year_hgsch_entry"] == 2005))
        )
    ]

    df1 = df1.loc[~df1["State"].isin(states_list2)]

    return df1


def create_treatment(df1):
    df1["Treat"] = df1.apply(lambda _: 0, axis=1)
    df1.loc[
        (df1["State"].isin(["[14] Sachsen", "[16] Thueringen"])),
        "Treat",
    ] = 1  # Always treated
    df1.loc[
        (
            (df1["State"].isin(["[1] Schleswig-Holstein"]))
            & (df1["year_hgsch_entry"] >= 2008)
        ),
        "Treat",
    ] = 1
    df1.loc[
        ((df1["State"].isin(["[2] Hamburg"])) & (df1["year_hgsch_entry"] >= 2002)),
        "Treat",
    ] = 1
    df1.loc[
        (
            (df1["State"].isin(["[3] Niedersachsen"]))
            & (df1["year_hgsch_entry"] >= 2003)
        ),
        "Treat",
    ] = 1
    df1.loc[
        ((df1["State"].isin(["[4] Bremen"])) & (df1["year_hgsch_entry"] >= 2004)),
        "Treat",
    ] = 1
    df1.loc[
        (
            (df1["State"].isin(["[5] Nordrhein-Westfalen"]))
            & (df1["year_hgsch_entry"] >= 2005)
        ),
        "Treat",
    ] = 1
    df1.loc[
        ((df1["State"].isin(["[6] Hessen"])) & (df1["year_hgsch_entry"] >= 2006)),
        "Treat",
    ] = 1
    df1.loc[
        (
            (df1["State"].isin(["[8] Baden-Wuerttemberg"]))
            & (df1["year_hgsch_entry"] >= 2004)
        ),
        "Treat",
    ] = 1
    df1.loc[
        ((df1["State"].isin(["[9] Bayern"])) & (df1["year_hgsch_entry"] >= 2003)),
        "Treat",
    ] = 1
    df1.loc[
        ((df1["State"].isin(["[10] Saarland"])) & (df1["year_hgsch_entry"] >= 2001)),
        "Treat",
    ] = 1
    df1.loc[
        ((df1["State"].isin(["[11] Berlin"])) & (df1["year_hgsch_entry"] >= 2006)),
        "Treat",
    ] = 1
    df1.loc[
        ((df1["State"].isin(["[12] Brandenburg"])) & (df1["year_hgsch_entry"] >= 2006)),
        "Treat",
    ] = 1
    df1.loc[
        (
            (df1["State"].isin(["[13] Mecklenburg-Vorpommern"]))
            & (df1["year_hgsch_entry"] >= 2002)
        ),
        "Treat",
    ] = 1
    df1.loc[
        (
            (df1["State"].isin(["[15] Sachsen-Anhalt"]))
            & (df1["year_hgsch_entry"] >= 1999)
        ),
        "Treat",
    ] = 1
    # df1['Gymnasium'] = pd.Categorical(df1['Gymnasium']

    return df1


def rename_variables(df1):
    trust_variables = ["jl0361", "jl0362", "jl0363"]
    for var in trust_variables:
        df1.loc[~df1[var].isin(["[-1] keine Angabe"])]
    df1.rename(
        columns={
            "jl0361": "trust",
            "jl0362": "rely_none",
            "jl0363": "distrust_stranger",
        },
        inplace=True,
    )
    df1["trust"] = (
        (
            ((df1["trust"].astype("str")).str.split(" ", n=1, expand=True)[0]).astype(
                "str",
            )
        )
        .str.strip("[]")
        .astype(int)
    )
    df1["rely_none"] = (
        (
            (
                (df1["rely_none"].astype("str")).str.split(" ", n=1, expand=True)[0]
            ).astype("str")
        )
        .str.strip("[]")
        .astype(int)
    )
    df1["distrust_stranger"] = (
        (
            (
                (df1["distrust_stranger"].astype("str")).str.split(
                    " ",
                    n=1,
                    expand=True,
                )[0]
            ).astype("str")
        )
        .str.strip("[]")
        .astype(int)
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
    df2.loc[(df2["sex"].isin(["[2] weiblich"])), "female"] = 1

    df2["rural"] = df2.apply(lambda _: " ", axis=1)
    df2.loc[~df2["jl0272"].isin(["[-1] keine Angabe"])]
    df2.loc[(df2["jl0272"].isin(["[4] Auf dem Land"])), "rural"] = 1
    df2.loc[~(df2["jl0272"].isin(["[4] Auf dem Land"])), "rural"] = 0

    df2["East"] = df2.apply(lambda _: 0, axis=1)
    df2.loc[
        (
            df2["State"].isin(
                [
                    "[11] Berlin",
                    "[12] Brandenburg",
                    "[13] Mecklenburg-Vorpommern",
                    "[14] Sachsen",
                    "[15] Sachsen-Anhalt",
                    "[16] Thueringen",
                ],
            )
        ),
        "East",
    ] = 1

    df2["migration_backgrnd"] = df2.apply(lambda _: 0, axis=1)
    df2.loc[
        ~(df2["migback"].isin(["[1] kein Migrationshintergrund"])),
        "migration_backgrnd",
    ] = 1

    df2["low_performing"] = df2.apply(lambda _: " ", axis=1)
    df2.loc[~df2["jl0151"].isin(["[-1] keine Angabe"])]
    df2.loc[(df2["jl0151"].isin(["[3] Gymnasialempfehlung"])), "low_performing"] = 0
    df2.loc[
        (
            df2["jl0151"].isin(
                [
                    "[1] Hauptschulempfehlung",
                    "[2] Realschulempfehlung",
                    "[4] keine Empfehlung",
                ],
            )
        ),
        "low_performing",
    ] = 1
    df2["low_performing"] = pd.Categorical(df2["low_performing"])

    df2.loc[
        ~df2["fsedu"].isin(
            ["[-5] In Fragebogenversion nicht enthalt", "[-1] keine Angabe"],
        )
    ]
    df2 = df2.rename(columns={"fsedu": "father_educ1"})
    df2["father_educ1"] = df2["father_educ1"].astype("str")
    new_5 = df2["father_educ1"].str.split(" ", n=1, expand=True)
    df2["feduc_1"] = new_5[0]
    df2["feduc_no"] = new_5[1]
    df2["feduc_1"] = df2["feduc_1"].astype("str")
    df2["feduc_1"] = df2["feduc_1"].str.strip("[]").astype(int)
    df2.loc[~(df2["feduc_1"] == 4), "feduc_1"] = 0
    df2.loc[(df2["feduc_1"] == 4), "feduc_1"] = 1

    df2.loc[
        ~df2["msedu"].isin(
            ["[-5] In Fragebogenversion nicht enthalt", "[-1] keine Angabe"],
        )
    ]
    df2 = df2.rename(columns={"msedu": "mother_educ1"})
    df2["mother_educ1"] = df2["mother_educ1"].astype("str")
    new_6 = df2["mother_educ1"].str.split(" ", n=1, expand=True)
    df2["meduc_1"] = new_6[0]
    df2["meduc_no"] = new_6[1]
    df2["meduc_1"] = df2["meduc_1"].astype("str")
    df2["meduc_1"] = df2["meduc_1"].str.strip("[]").astype(int)
    df2.loc[~(df2["meduc_1"] == 4), "meduc_1"] = 0
    df2.loc[(df2["meduc_1"] == 4), "meduc_1"] = 1

    df2.loc[
        ~df2["fprofedu"].isin(
            ["[-5] In Fragebogenversion nicht enthalt", "[-1] keine Angabe"],
        )
    ]
    df2 = df2.rename(columns={"fprofedu": "father_educ2"})
    df2["father_educ2"] = df2["father_educ2"].astype("str")
    new_7 = df2["father_educ2"].str.split(" ", n=1, expand=True)
    df2["feduc_2"] = new_7[0]
    df2["feduc_no2"] = new_7[1]
    df2["feduc_2"] = df2["feduc_2"].astype("str")
    df2["feduc_2"] = df2["feduc_2"].str.strip("[]").astype(int)
    df2.loc[((df2["feduc_2"] < 28) | (df2["feduc_2"] > 32)), "feduc_2"] = 0
    df2.loc[~((df2["feduc_2"] < 28) | (df2["feduc_2"] > 32)), "feduc_2"] = 1

    df2.loc[
        ~df2["mprofedu"].isin(
            ["[-5] In Fragebogenversion nicht enthalt", "[-1] keine Angabe"],
        )
    ]
    df2 = df2.rename(columns={"mprofedu": "mother_educ2"})
    df2["mother_educ2"] = df2["mother_educ2"].astype("str")
    new_8 = df2["mother_educ2"].str.split(" ", n=1, expand=True)
    df2["meduc_2"] = new_8[0]
    df2["meduc_no2"] = new_8[1]
    df2["meduc_2"] = df2["meduc_2"].astype("str")
    df2["meduc_2"] = df2["meduc_2"].str.strip("[]").astype(int)
    df2.loc[((df2["meduc_2"] < 28) | (df2["meduc_2"] > 32)), "meduc_2"] = 0
    df2.loc[~((df2["meduc_2"] < 28) | (df2["meduc_2"] > 32)), "meduc_2"] = 1

    df2["highest_educ_hh"] = df2.apply(lambda _: 0, axis=1)
    df2.loc[
        (
            (df2["feduc_1"] == 1)
            | (df2["meduc_1"] == 1)
            | (df2["feduc_2"] == 1)
            | (df2["meduc_2"] == 1)
        ),
        "highest_educ_hh",
    ] = 1

    df2.loc[~df2["freli"].isin(["[-1] keine Angabe"])]
    df2 = df2.rename(columns={"freli": "father_reli"})
    df2["father_reli"] = df2["father_reli"].astype("str")
    new_9 = df2["father_reli"].str.split(" ", n=1, expand=True)
    df2["freli"] = new_9[0]
    df2["freli"] = df2["freli"].astype("str")
    df2["freli"] = df2["freli"].str.strip("[]").astype(int)
    df2.loc[((df2["freli"] == 1) | (df2["freli"] == 2)), "freli"] = 1
    df2.loc[~((df2["freli"] == 1) | (df2["freli"] == 2)), "freli"] = 0

    df2.loc[~df2["mreli"].isin(["[-1] keine Angabe"])]
    df2 = df2.rename(columns={"mreli": "mother_reli"})
    df2["mother_reli"] = df2["mother_reli"].astype("str")
    new_10 = df2["mother_reli"].str.split(" ", n=1, expand=True)
    df2["mreli"] = new_10[0]
    df2["mreli"] = df2["mreli"].str.strip("[]").astype(int)
    df2.loc[((df2["mreli"] == 1) | (df2["mreli"] == 2)), "mreli"] = 1
    df2.loc[~((df2["mreli"] == 1) | (df2["mreli"] == 2)), "mreli"] = 0

    df2["reli_hh"] = df2.apply(lambda _: 1, axis=1)
    df2.loc[((df2["freli"] == 0) & (df2["mreli"] == 0)), "reli_hh"] = 0

    df2 = df2.loc[~df2["mprofstat"].isin(["[-1] keine Angabe"])]
    df2 = df2.rename(columns={"mprofstat": "work_mother"})
    df2["work_mother"] = df2["work_mother"].astype("str")
    new_11 = df2["work_mother"].str.split(" ", n=1, expand=True)
    df2["m_work"] = new_11[0]
    df2["m_work"] = df2["m_work"].str.strip("[]").astype(int)
    df2.loc[~(df2["m_work"] < 10) & (df2["m_work"] > 15), "m_work"] = 1
    df2.loc[(df2["m_work"] >= 10) & (df2["m_work"] <= 15), "m_work"] = 0

    df2 = df2.loc[~df2["fegp88"].isin(["[-1] keine Angabe", "[-2] trifft nicht zu"])]
    df2 = df2.rename(columns={"fegp88": "work_father"})
    df2["work_father"] = df2["work_father"].astype("str")
    new_12 = df2["work_father"].str.split(" ", n=1, expand=True)
    df2["father_blue_collar"] = new_12[0]
    df2["father_blue_collar"] = df2["father_blue_collar"].str.strip("[]").astype(int)
    df2.loc[(df2["father_blue_collar"] < 8), "father_blue_collar"] = 0
    df2.loc[(df2["father_blue_collar"] >= 8), "father_blue_collar"] = 1

    df2 = df2.loc[~df2["living1"].isin(["[-2] trifft nicht zu", "[-3] nicht valide"])]
    df2 = df2.rename(columns={"living1": "single_parent"})
    df2.loc[(df2["single_parent"] < 15), "single_parent"] = 1
    df2.loc[(df2["single_parent"] == 15), "single_parent"] = 0

    df2 = df2.loc[~df2["jl0176_h"].isin(["[-1] keine Angabe"])]
    df2 = df2.rename(columns={"jl0176_h": "migration_classmate"})
    df2["migration_classmate"] = df2["migration_classmate"].astype("str")
    new_13 = df2["migration_classmate"].str.split(" ", n=1, expand=True)
    df2["migback_classmate"] = new_13[0]
    df2["migback_classmate"] = df2["migback_classmate"].str.strip("[]").astype(int)
    df2.loc[(df2["migback_classmate"] < 6), "migback_classmate"] = 1
    df2.loc[(df2["migback_classmate"] == 6), "migback_classmate"] = 0

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
    print(leadslags_plot)
    leadslags_plot["lb"] = leadslags_plot["mean"] - leadslags_plot["sd"] * 1.96
    leadslags_plot["ub"] = leadslags_plot["mean"] + leadslags_plot["sd"] * 1.96
    print(leadslags_plot)
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
    # plot.savefig("leadslags_plot1.png")
    # plot.show()
    plot.save(filename="leadslags_plot3.jpg", dpi=1000)
    return plot.draw()


data = pd.read_stata(r"C:\input_data\merge_original_youth_data.dta")
data.head()

raw_dataframe = data.copy()
raw_dataframe.set_index("pid", inplace=True)


processed_dataframe = process_data(raw_dataframe)
# processed_dataframe.shape


for school in ["gymnasium", "not_gymnasium"]:

    print("THE SCHOOL VALUE IS" + school)
    half_done = filter_based_on_school(processed_dataframe, school)
    Treatment = create_treatment(half_done)

    Variable_renamed = rename_variables(Treatment)

    Ind_Rename = rename_independent_var(Variable_renamed)
    Doing_event = event_study(Ind_Rename)
    doing_lead_lag = lead_lag(Ind_Rename)

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
# for school_type in ['gymnasium', 'not_gymnasium']:


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
formula = "std_trust_var ~ lead1 + lead2 + lead3 + lead4 + lead5 + lead6 + lead7 + lag0 + lag1 + lag2 + lag3 + lag4 + lag5 + lag6 + lag7+ C(year_hgsch_entry)"

result_plot = plot_event_study(formula, Ind_Rename)
