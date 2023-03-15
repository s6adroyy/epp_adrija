import epp_adrija.data_management.clean_data as clean_df
import pandas as pd
import pytest

# @pytest.fixture()
# def data():


# @pytest.fixture()
# def data_info():


@pytest.fixture()
def input_dataframe():
    # Create a test input dataframe
    return pd.DataFrame(
        {
            "syear": [2005, 2010, 2015, 2020],
            "jl0233": [1988, 1993, 1998, 2003],
            "psample": [10, 11, 12, 13],
            "jl0164": [1, 2, 1, 2],
        },
    )


def test_process_data(input_dataframe):
    # Call the process_data function
    output_dataframe = clean_df.process_data(input_dataframe)

    # Check that the output dataframe has the expected shape
    assert output_dataframe.shape == (1, 5)

    # Check that the output dataframe has the expected values
    expected_output = pd.DataFrame(
        {
            "syear": [2010],
            "jl0233": [1993],
            "psample": [11],
            "jl0164": [2],
            "Age": [17],
        },
    )
    output_dataframe.reset_index(drop=True, inplace=True)
    expected_output.reset_index(drop=True, inplace=True)
    pd.testing.assert_frame_equal(output_dataframe, expected_output)


@pytest.fixture()
def example_df():
    # create a small example dataframe for testing
    import pandas as pd

    example_data = {
        "jl0125_v3": [3, 6, 2, 4, 5, 6],
        "jl0127_h": [1, 4, 2, 3, 4, 5],
        "bet3year": [2000, 2001, 2002, None, 2004, 2005],
        "gebjahr": [1990, 1995, 2000, 2005, 2010, 2015],
        "bula_h": [1, 2, 3, 4, 5, 6],
    }
    return pd.DataFrame(example_data)


def test_filter_based_on_school(example_df):
    # check that the function returns a dataframe
    output_df = clean_df.filter_based_on_school(example_df)
    assert isinstance(output_df, pd.DataFrame)

    # check that the output dataframe has the expected columns
    expected_columns = [
        "jl0125_v3",
        "jl0127_h",
        "bet3year",
        "gebjahr",
        "Gymnasium",
        "year_hgsch_entry",
        "State",
    ]
    assert all(col in output_df.columns for col in expected_columns)

    # check that the output dataframe only contains rows where Gymnasium == 1
    assert all(output_df["Gymnasium"] == 1)

    # check that the output dataframe only contains rows with valid year_hgsch_entry values
    assert all(output_df["year_hgsch_entry"].notnull())

    # check that the output dataframe does not contain any rows with State == 7 or 6 (with year_hgsch_entry == 2004 or 2005)
    assert not any(
        (output_df["State"] == 7.0)
        | (
            (output_df["State"] == 6.0)
            & (
                (output_df["year_hgsch_entry"] == 2004)
                | (output_df["year_hgsch_entry"] == 2005)
            )
        ),
    )

    # check that the output dataframe does not contain any rows with State not in [11, 12, 13]
    assert not any(
        ~output_df["State"].isin([11.0, 12.0, 13.0])
        & (
            (output_df["year_hgsch_entry"] == 0) | (output_df["year_hgsch_entry"] == -2)
        ),
    )

    # check that the output dataframe contains only rows where Gymnasium is set correctly based on the jl0125_v3 and jl0127_h columns
    assert all(
        (output_df["jl0125_v3"] == 3)
        | ((output_df["jl0125_v3"] == 6) & (output_df["jl0127_h"] == 4))
        == output_df["Gymnasium"],
    )


@pytest.fixture()
def example_dataframe():
    # Create example dataframe
    df = pd.DataFrame(
        {"State": [1, 2, 3, 4, 5], "year_hgsch_entry": [2008, 2002, 2003, 2004, 2005]},
    )
    return df


def test_create_treatment(example_dataframe):
    # Call the function on the example dataframe
    output_df = clean_df.create_treatment(example_dataframe)

    # Check that the output dataframe has the expected number of rows and columns
    assert output_df.shape == (5, 3)

    # Check that the "Treat" column was added to the output dataframe
    assert "Treat" in output_df.columns

    # Check that the "Treat" column has the expected values for each row
    expected_treat_values = [1, 1, 1, 1, 1]
    assert list(output_df["Treat"]) == expected_treat_values


"""
@pytest.fixture()
def input_data():
    return pd.DataFrame(
        {
            "syear": [2005, 2006, 2007, 2018, 2019],
            "jl0233": [1995, 1998, 2001, 2000, 2003],
            "psample": [15 , 16, 19],
            "jl0164": [2, 1, 2, 1, 2],
        },
    )


def test_process_data(input_data):
    expected_output = pd.DataFrame(
        {
            "syear": [2006, 2007, 2018],
            "jl0233": [1998.0, 2001.0, 2000.0],
            "psample": ["Other", "[16] M2 2015 Migration (2009-2013)", "Other"],
            "jl0164": ["[1] Ja", "[2] Nein", "[1] Ja"],
            "Age": [8.0, 6.0, 18.0],
        },
    )
    output = clean_df.process_data(input_data)
    pd.testing.assert_frame_equal(output, expected_output)


# def test_process_data():
# create input dataframe

# create expected output dataframe

# call the process_data function

# assert that the output dataframe is the same as the expected output dataframe


@pytest.fixture()
def df():
    data = {
        "jl0125_v3": [
            "[3] Ja Gymnasium",
            "[6] Nein",
            "[1] Ja Hauptschule",
            "[2] Ja Realschule",
        ],
        "jl0127_h": [
            "[4] Fachhochshulreife/Abitur",
            "[5] Nichts",
            "[5] Nichts",
            "[5] Nichts",
        ],
        "gebjahr": [2000, 2001, 2002, 2003],
        "bet3year": [2012, 0, 2010, 2009],
        "bula_h": [
            "[6] Hessen",
            "[11] Berlin",
            "[5] Bremen",
            "[7] Rheinland-Pfalz,Saarland",
        ],
    }
    df = pd.DataFrame(data)
    return df


def test_filter_based_on_school(df):
    # Expected output for flag='gymnasium'
    expected_output_gym = pd.DataFrame(
        {
            "jl0125_v3": ["[3] Ja Gymnasium"],
            "jl0127_h": ["[4] Fachhochshulreife/Abitur"],
            "gebjahr": [2000],
            "bet3year": [2012],
            "bula_h": ["[6] Hessen"],
            "Gymnasium": [1],
            "year_hgsch_entry": [2010],
            "State": ["[6] Hessen"],
        },
    )
    # Expected output for flag='not_gymnasium'
    expected_output_not_gym = pd.DataFrame(
        {
            "jl0125_v3": ["[6] Nein", "[1] Ja Hauptschule", "[2] Ja Realschule"],
            "jl0127_h": ["[5] Nichts", "[5] Nichts", "[5] Nichts"],
            "gebjahr": [2001, 2002, 2003],
            "bet3year": [0, 2010, 2009],
            "bula_h": ["[11] Berlin", "[5] Bremen", "[7] Rheinland-Pfalz,Saarland"],
            "Gymnasium": [0, 0, 0],
            "year_hgsch_entry": [2012, 0, 0],
            "State": ["[11] Berlin", "[5] Bremen", "[7] Rheinland-Pfalz,Saarland"],
        },
    )

    # Test the function with flag='gymnasium'
    output_gym = clean_df.filter_based_on_school(df, "gymnasium")
    assert_frame_equal(output_gym, expected_output_gym)

    # Test the function with flag='not_gymnasium'
    output_not_gym = clean_df.filter_based_on_school(df, "not_gymnasium")
    assert_frame_equal(output_not_gym, expected_output_not_gym)



# @pytest.fixture
def test_create_treatment():
    # Create test dataframe
    df = pd.DataFrame(
        {
            "State": ["[14] Sachsen", "[16] Thueringen", "[1] Schleswig-Holstein"],
            "year_hgsch_entry": [2004, 2005, 2008],
        },
    )

    # Create expected output
    expected_output = pd.DataFrame(
        {
            "State": ["[14] Sachsen", "[16] Thueringen", "[1] Schleswig-Holstein"],
            "year_hgsch_entry": [2004, 2005, 2008],
            "Treat": [1, 1, 1],
        },
    )

    # Call function and compare output with expected output
    output = clean_df.create_treatment(df)
    assert output.equals(
        expected_output,
    ), "Test failed: output does not match expected output"


# @pytest.fixture
def test_rename_variables():
    # Create test dataframe
    df = pd.DataFrame(
        {
            "jl0361": [-1, 2, 3],
            "jl0362": [
                "[-1] keine Angabe",
                " [1] trifft voll zu",
                " [3] trifft eher zu",
            ],
            "jl0363": [
                "[1] trifft voll zu",
                "[-1] keine Angabe",
                "[2] trifft eher nicht zu",
            ],
        },
    )

    # Call function to rename variables
    df_renamed = clean_df.rename_variables(df)

    # Assert that columns are renamed correctly
    assert "trust" in df_renamed.columns
    assert "rely_none" in df_renamed.columns
    assert "distrust_stranger" in df_renamed.columns

    # Assert that missing values are removed from trust_variables
    assert all(df_renamed["jl0361"] != -1)

    # Assert that variable values are correctly extracted and converted
    assert df_renamed["trust"].equals(pd.Series([0, 2, 3]))
    assert df_renamed["rely_none"].equals(pd.Series([0, 1, 3]))
    assert df_renamed["distrust_stranger"].equals(pd.Series([1, 0, 2]))

    # Assert that rely_someone, trust_stranger, and trust_var are correctly calculated
    assert df_renamed["rely_someone"].equals(pd.Series([8, 7, 5]))
    assert df_renamed["trust_stranger"].equals(pd.Series([7, 8, 6]))
    assert df_renamed["trust_var"].equals(pd.Series([15, 17, 14]))

    # Assert that std_trust_var and std_trust are correctly calculated
    assert df_renamed["std_trust_var"].equals(
        pd.Series([-0.21821789, 1.09108945, -0.87287156]),
    )
    assert df_renamed["std_trust"].equals(pd.Series([-1.22474487, 0.0, 1.22474487]))


from numpy.testing import assert_array_equal


def test_rename_independent_var():
    data = {
        "sex": {0: "[1] männlich", 1: "[2] weiblich", 2: "[2] weiblich"},
        "jl0272": {
            0: "[-1] keine Angabe",
            1: "[4] Auf dem Land",
            2: "[-1] keine Angabe",
        },
        "State": {
            0: "[11] Berlin",
            1: "[12] Brandenburg",
            2: "[13] Mecklenburg-Vorpommern",
        },
        "migback": {
            0: "[1] kein Migrationshintergrund",
            1: "[1] kein Migrationshintergrund",
            2: "[3] Aussiedler",
        },
        "jl0151": {
            0: "[-1] keine Angabe",
            1: "[2] Realschulempfehlung",
            2: "[1] Hauptschulempfehlung",
        },
        "fsedu": {
            0: "[3] Fachabitur",
            1: "[2] Realschulabschluss",
            2: "[-1] keine Angabe",
        },
        "msedu": {
            0: "[1] Hauptschulabschluss",
            1: "[-1] keine Angabe",
            2: "[2] Realschulabschluss",
        },
        "fprofedu": {
            0: "[-1] keine Angabe",
            1: "[4] Techniker oder Meister",
            2: "[2] Fachschulabschluss",
        },
        "age": {0: 16, 1: 17, 2: 16},
    }
    df = pd.DataFrame(data)
    df2 = clean_df.rename_independent_var(df)

    assert_array_equal(df2["female"].to_numpy(), np.array([0, 1, 1]))
    assert_array_equal(df2["rural"].to_numpy(), np.array([0, 1, 0]))
    assert_array_equal(df2["East"].to_numpy(), np.array([1, 1, 1]))
    assert_array_equal(df2["migration_backgrnd"].to_numpy(), np.array([0, 0, 1]))
    assert_array_equal(df2["low_performing"].to_numpy(), np.array([1, 0, 1]))
    assert_array_equal(df2["feduc_1"].to_numpy(), np.array([0, 0, 1]))
    assert_array_equal(df2["meduc_1"].to_numpy(), np.array([1, 0, 0]))
    assert_array_equal(df2["feduc_2"].to_numpy(), np.array(["nan", "4", "2"]))


# def test_clean_data_drop_columns(data, data_info):


# def test_clean_data_dropna(data, data_info):


# def test_clean_data_categorical_columns(data, data_info):
#   for cat_col in data_info["categorical_columns"]:


# def test_clean_data_column_rename(data, data_info):


# def test_convert_outcome_to_numerical(data, data_info):
# assert outcome_numerical_name in data_clean.columns
"""
