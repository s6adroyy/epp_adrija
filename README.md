# Effective Programming Practices for Economists Final Project 

This project aims to look at the trust attitude of 17 year old children who has been 
treated under German G8 educational reform. Merged five data sets. Click to get the 
information of the datasets [ppathl](https://www.diw.de/documents/publikationen/73/diw_01.c.745961.de/diw_ssp0835.pdf), [hbrutto](https://www.diw.de/documents/publikationen/73/diw_01.c.850363.de/diw_ssp1180.pdf), [bioedu](https://paneldata.org/soep-core/datasets/bioedu/),[bioparen](https://www.diw.de/documents/publikationen/73/diw_01.c.850342.de/diw_ssp1177.pdf),
and [jugendl](https://paneldata.org/soep-core/datasets/jugendl/). 

## Usage

To get started 

  $ conda env create -f environment.yml
  $ conda activate epp_adrija

This repo has all the folders including the tests and the task files for plotting, analysis and data cleaning.

# Important details about the repository

#### The src folder represents the source files. It contains the following folders:

- epp_adrija which has data - original data from the SOEP data set.
- data management folder contains the data cleaning process as clean_data and data_info files .
- analysis folder provides model.py file which has the functions of loading the model , running the main diffrence in difference regression and also function of the mechanism regressions. 
- final folder contains plot.py which has the function of plot and the task_final.py which has the task for producing the plot in figures folder in bld (not in version control).
- tests folder caontains one tests for analysis as test_model.py and somes tets for data cleaning as test_clean_data.py.


### Final paper -

For the final paper pdf to generate, please make sure that pytask-latex and all the latex related packages are properly installed.

### Testing

Testing functions are located in tests folder. These functions are written to test and assert the original functions on a small representative data set.

Run the test file using

```
$ pytest
```

Run task files using 

  $ pytask 

#### Other changes include:

- the environment.yml contains two new packages statsmodel and plotnine.

## Acknowledgment

We would like to express our deep gratitude to Professor Hans-Martin von Gaudecker, Janos Gabler and Gregor Boehl for their support and guidance.
