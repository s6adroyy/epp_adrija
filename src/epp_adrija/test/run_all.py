import clean_data1
import pandas as pd

raw_dataset_path = "../data/merge_original_youth_data.dta"


def main():
    # Read Data
    raw_data = pd.read_stata(raw_dataset_path, convert_categoricals=False)

    # Clean Data
    processed_data1 = clean_data1.process_data(raw_data)
    processed_data2 = clean_data1.filter_based_on_school(processed_data1)
    processed_data3 = clean_data1.create_treatment(processed_data2)
    processed_data4 = clean_data1.rename_variables(processed_data3)
    clean_data1.rename_independent_var(processed_data4)

    # Analysis

    # Plot results

    # End


main()
