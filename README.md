# Effective Programming Practices for Economists Final Project Winter Semester 2020-2021

## Usage

To get started $ conda activate epp_adrija

This repo has all the folders including the tests and the task files for plotting, analysis and data cleaning.

# Important details about the repository

#### The src folder represents the source files. It contains the following folders:

- epp_adrija which has data - original data from the SOEP data set which I already merged previously in stata and then stored here . In general the megred_original_youth_data contains the merge of ppathl, hbrutto, bioedu, bioparen, jugendl data sets.
- data management folder contains the data cleaning process as clean_data and data_info files .
- analysis folder provides model.py file which has the functions of loading the model , running the main diffrence in difference regression and also function of the mechanism regressions. Second it also has a task file as task_analysis.py which
  produces the results of the model.py in pickel format in the models folder in bld/python.
- final folder contains plot.py which has the function of plot and the task_final.py which has the task for producing the plot in figures folder in bld and task for producing the regression result in tex format in tables folder in bld/python.
- tests folder caontains one tets for analysis as test_model.py and somes tets for data cleaning as test_clean_data.py.

#### The bld /python folder represents the output files. It contains the following folders:

- data has the final cleaned data sets of which the raw_dataframe is just the copy of the original data and three cleaned data sets of which final_df is used for the main diff-in-diff regression. The eventstudy.dta used for plotting has some more
  changes like including the leads and lags column to the final_df and lastly the mechanisms.dta which used the function mechanism from clean_data.py on final_df has 34 observations.
- models contains the pickle results of main_regression and four other mechanisms.
- figures contains the event study plot.
- tables contain the tex format of all the regression models.
- paper contains the final pdf in tex format produced with task_paper.py.

### Final paper -

For the final paper pdf to generate, please make sure that pytask-latex and all the latex related packages are properly installed.

### Testing

Testing functions are located in tests folder. These functions are written to test and assert the original functions on a small representative data set.

Run the test file using

```
$ pytest
```

Run task files using $ pytask Run specific task files using $ pytask -m wip

#### Other changes include:

- the environment.yml contains two new packages statsmodel and plotnine.

## Acknowledgment

We would like to express our deep gratitude to Professor Hans-Martin von Gaudecker, Janos Gabler and Gregor Boehl for their support and guidance.
