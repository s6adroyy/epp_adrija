import epp_adrija.data_management.clean_data as clean_df
import pandas as pd
import pytest



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


