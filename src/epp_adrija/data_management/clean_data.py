"""Function(s) for cleaning the data set(s)."""


import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)


def process_data(input_dataframe):
    input_dataframe["syear"] = pd.to_numeric(input_dataframe["syear"], errors="coerce")
    input_dataframe["jl0233"] = pd.to_numeric(
        input_dataframe["jl0233"],
        errors="coerce",
    )
    # Filter and get rows where syear in 2006 to 2018 and restrict the sample 
     # with no migration background
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
    # Locked the students in gymnasium(jl0125_v3) and those not in gymnasium presntly but in university 
    df1.loc[
        ((df1["jl0125_v3"] == 3) | ((df1["jl0125_v3"] == 6) & (df1["jl0127_h"] == 4))),
        "Gymnasium",
    ] = 1

    df1 = df1.loc[(df1["Gymnasium"] == 1)].copy()
    # Create empty col of high school entry for replacing the NaN values instead of dropping
    df1["year_hgsch_entry"] = df1["bet3year"].fillna(0).values
    df1 = df1.rename(columns={"bula_h": "State"})

    # list of states (more info in data_info.yaml) where the year_hgsch_entry is birth year(gebjahr) + 12 years
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
    # Remove states where reform has not been implemented state-wide (Rhineland-Palatinate)
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
    
    return df1

# All values below 0 has either not been replied or not is not in the survey year 
# Any observations having value below 0 has been removed 
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
    # Reverse the scale for "nagetive" items*
    df1["rely_someone"] = df1.apply(lambda _: 0, axis=1)
    df1.loc[(~df1["rely_none"] <= 0), "rely_someone"] = 8 - df1["rely_none"]
    df1["trust_stranger"] = df1.apply(lambda _: 0, axis=1)
    df1.loc[(~df1["distrust_stranger"] <= 0), "trust_stranger"] = (
        8 - df1["distrust_stranger"]
    )
    # Adding the scores of trust 
    df1["trust_var"] = df1.apply(lambda _: 0, axis=1)
    df1.loc[(~df1["trust"] <= 0), "trust_var"] = (
        df1["trust"] + df1["rely_someone"] + df1["trust_stranger"]
    )
    # Standardizing the value
    df1["std_trust_var"] = (df1["trust_var"] - df1["trust_var"].mean()) / df1[
        "trust_var"
    ].std()
    df1["std_trust"] = (df1["trust"] - df1["trust"].mean()) / df1["trust"].std()

    return df1

# Any observations having value below 0 has been removed 
# All minor details in data_info.yaml

def rename_independent_var(df2):
    df2["female"] = df2.apply(lambda _: 0, axis=1)
    df2.loc[(df2["sex"] == 2), "female"] = 1

    df2["rural"] = 0
    df2 = df2.loc[~(df2["jl0272"] == 4)]
    df2.loc[(df2["jl0272"] == 4), "rural"] = 1

    df2["East"] = df2.apply(lambda _: 0, axis=1)
    East_states = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    df2.loc[(df2["State"].isin(East_states)), "East"] = 1

    df2["migration_backgrnd"] = df2.apply(lambda _: 0, axis=1)
    df2.loc[~(df2["migback"] == -1), "migration_backgrnd"] = 1

    df2["low_performing"] = 0
    df2.loc[(~df2["jl0151"] == -1)]
    # jl0151 = 3 (Gymnasium)
    df2.loc[(df2["jl0151"] == 3), "low_performing"] = 0
    # Non_gym_school [1= Hauptschule , 2= Realschule, 4= Not recommended]
    Non_gym_school = [1, 2, 4]
    df2.loc[(df2["jl0151"].isin(Non_gym_school)), "low_performing"] = 1

    Nan_fsedu = [-5.0, -1.0]
    df2 = df2.loc[~df2["fsedu"].isin(Nan_fsedu)]
    df2 = df2.rename(columns={"fsedu": "father_educ1"})
    # fsedu = 4 [Abitur]
    df2.loc[~(df2["father_educ1"] == 4.0), "father_educ1"] = 0
    df2.loc[(df2["father_educ1"] == 4.0), "father_educ1"] = 1

    Nan_msedu = [-5.0, -1.0]
    df2 = df2.loc[~df2["msedu"].isin(Nan_msedu)]
    df2 = df2.rename(columns={"msedu": "mother_educ1"})
    # msedu = 4[Abitur]
    df2.loc[~(df2["mother_educ1"] == 4.0), "mother_educ1"] = 0
    df2.loc[(df2["mother_educ1"] == 4.0), "mother_educ1"] = 1

    Nan_fprofedu = [-5.0, -1.0]
    df2.loc[~df2["fprofedu"].isin([Nan_fprofedu])]
    df2 = df2.rename(columns={"fprofedu": "father_educ2"})
    # 32 <fprofedu <28 has all information of ongoing schooling 
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
    # 32 <mprofedu <28 has all information of ongoing schooling 
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
    # mprofstat < 10 (no work)  , 15 < mprofstat (still under training/studies)
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
    # living1 = 15 is all years till age 15 
    df2.loc[(df2["single_parent"] < 15.0), "single_parent"] = 1
    df2.loc[(df2["single_parent"] == 15.0), "single_parent"] = 0

    df2 = df2.loc[~(df2["jl0176_h"] == -1)]
    df2 = df2.rename(columns={"jl0176_h": "migration_classmate"})
    # jl0176_h < 6 students having migrataed classmates 
    df2.loc[(df2["migration_classmate"] < 6), "migration_classmate"] = 1
    df2.loc[(df2["migration_classmate"] == 6), "migration_classmate"] = 0

    return df2


def mechanism(df3):
    Nan_jl0072 = [-5.0, -8.0, -1.0]
    df3 = df3.loc[~df3["jl0072"].isin(Nan_jl0072)]
    df3 = df3.rename(columns={"jl0072": "volunteer_work"})
    df3.loc[(df3["volunteer_work"] < 4), "volunteer_work"] = 1
    df3.loc[
        (df3["volunteer_work"] >= 4) & (df3["volunteer_work"] <= 5), "volunteer_work"
    ] = 0

    Nan_jl0218 = [-8.0, -1.0]
    df3 = df3.loc[~df3["jl0218"].isin(Nan_jl0218)]
    df3 = df3.rename(columns={"jl0218": "mental_health"})
    df3.loc[(df3["mental_health"] < 4), "mental_health"] = 1
    df3.loc[
        (df3["mental_health"] >= 4) & (df3["mental_health"] <= 5), "mental_health"
    ] = 0


     # Recode values of jl0139 to jl0146
     # -2 here means not applicable 
    df3['jl0139'] = df3['jl0139'].replace([-2, 1], [0, 1])
    df3['jl0140'] = df3['jl0140'].replace([-2, 1], [0, 1])
    df3['jl0141'] = df3['jl0141'].replace([-2,1], [0, 1])
    df3['jl0142'] = df3['jl0142'].replace([-2,1], [0, 1])
    df3['jl0143'] = df3['jl0143'].replace([-2,1], [0, 1])
    df3['jl0144'] = df3['jl0144'].replace([-2,1], [0, 1])
    df3['jl0145'] = df3['jl0145'].replace([-2, 1], [0, 1])
    df3 = df3.loc[~(df3["jl0146"] == -1.0)]
    df3['jl0146'] = df3['jl0146'].replace([-2, 1], [0, 1])

    # Rename variables
    df3 = df3.rename(columns={'jl0139': 'class_rprsttv',
                            'jl0140': 'student_rprsttv',
                            'jl0141': 'school_magazine',
                            'jl0142': 'drama_dance_group',
                            'jl0143': 'choir_orchestra',
                            'jl0144': 'sport_group',
                            'jl0145': 'other_school_group',
                            'jl0146': 'none_school_activity'})
    
    df3["some_school_group"] = df3.apply(lambda _: 0, axis=1)
    school_group = ["class_rprsttv", "student_rprsttv", "school_magazine",
                    "drama_dance_group", "choir_orchestra", "sport_group",
                    "other_school_group"]
    # Replacing the vales of the some_school_group == 1 iff they are enagaged in any school group
    for var in school_group:
        df3.loc[~(df3[var] == 1), "some_school_group"] = 1

    # Nan_jl0105_h = [-1.0]
    df3 = df3.loc[~(df3["jl0105_h"] == -1.0)]
    # df3 = df3.loc[~df3["jl0105_h"].isin(Nan_jl0105_h)]
    df3 = df3.rename(columns={"jl0105_h": "sport_active"})
    df3.loc[(df3["sport_active"] == 1), "sport_active"] = 1
    df3.loc[(df3["sport_active"] == 2), "sport_active"] = 0

    if "level_0" in df3.columns:
        # # if it exists, remove it
        df3 = df3.drop("level_0", axis=1)

    return df3


def event_study(df3):
    df3["treat_year"] = df3.apply(lambda _: 0, axis=1)
    treat_year_dict = {
        2: 2002,
        3: 2003,
        4: 2004,
        5: 2002,
        8: 2004,
        9: 2003,
        10: 2001,
        11: 2006,
        12: 2006,
        13: 2002,
        15: 1999,
    }
    for states in treat_year_dict:
        df3.loc[(df3["State"].isin([states])), "treat_year"] = treat_year_dict[states]
    df3["t"] = df3["year_hgsch_entry"] - df3["treat_year"]
    df3["lag0"] = df3["t"] == 0
    # Creating leads and lags depending on t ( negative values lags and postive values leads)
    for i in range(1, 8):
        df3["lag" + str(i)] = df3["t"] == -i
        df3["lead" + str(i)] = df3["t"] == i
    if "level_0" in df3.columns:
        # # if it exists, remove it
        df3 = df3.drop("level_0", axis=1)
    return df3


