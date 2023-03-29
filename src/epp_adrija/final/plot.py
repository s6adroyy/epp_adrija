"""Functions plotting results."""

import pandas as pd
import numpy as np
import plotnine as p
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from plotnine import *



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
        + p.ylab("Coeffs")
        + p.geom_hline(yintercept=0, linetype="dashed")
        + p.geom_vline(xintercept=0, linetype="dashed")
    )
    # plot.save("leadslags_plot4.png")
    # plot.save(filename="leadslags_plot1.jpg", dpi= 1000)

    # ggplot.save("leadslags_plot4.png")
    return plot
    # ggsave("leadslags_plot4.png", plot)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def descriptive_stats_plot(data):
    # Generate binary variables for female and migration_back
    data['female_'] = data['female'].replace({1: 'Female', 0: ''})
    data['migration_classmate_'] = data['migration_classmate'].replace({1: 'Migration class.', 0: ''})

    # Generate categorical variables for highest_edu_hh, father_blue_collar, single_parent, religion_hh, and low_perform
    #data['highest_edu_hh_'] = pd.cut(data['highest_edu_hh'], bins=[-1, 0, 6, np.inf], labels=["", "Low education", "High education"])
    data['highest_edu_hh_'] = data['highest_educ_hh'].replace({1: 'High_educ_hh', 0: ''})
    data['father_blue_collar_'] = data['work_father'].replace({1: 'Working-class father', 0: ''})
    data['east_'] = data['East'].replace({1: 'East-de', 0: ''})
    data['work_mother_'] = data['work_mother'].replace({1: 'Working-class mother', 0: ''})
    data['single_parent_'] = data['single_parent'].replace({1: 'Single parent', 0: ''})
    data['religion_hh_'] = data['reli_hh'].replace({1: 'Christian parents', 0: ''})
    data['low_perform_'] = data['low_performing'].replace({1: 'Low-perform. student', 0: ''})

    # Compute means for each variable by treatment status using pivot_table
    pivot_data = pd.pivot_table(data, values=['female', 'migration_classmate', 'highest_educ_hh', 'work_father','East', 
                                              'work_mother', 'single_parent', 'reli_hh', 'low_performing'], index=['Treat'], aggfunc=np.mean)

    # Create a matrix of means
    m1 = pivot_data.transpose()
    m1.columns = ['Control', 'Treatment']
    m1.index.name = 'Variables'

    # Rename the rows of the matrix
    m1 = m1.rename(index={'female_': 'Female', 'migration_classmate_': 'Migration class', 'highest_edu_hh_': 'High parental educ.', 
                           'father_blue_collar_': 'Working-class father', 'east_': 'East_de', 'work_mother_': 'Working-class mother',
                           'single_parent_': 'Single parent', 'religion_hh_': 'Christian parents',
                           'low_perform_': 'Low-perform. student'})

    # Round the values in the matrix to 2 decimal places
    m1 = np.round(m1, 2)
    
    # Create a horizontal bar graph
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(m1.index))
    ax.barh(y_pos - 0.2, m1['Control'], height=0.4, color='blue', label='Control')
    ax.barh(y_pos + 0.2, m1['Treatment'], height=0.4, color='cyan', label='Treatment')
    ax.legend(loc='lower right')
    ax.set_xlabel('Mean')
    ax.set_title('Descriptive statistics by treatment status')
    # Add annotations for each bar
    for i, (control, treatment) in enumerate(zip(m1['Control'], m1['Treatment'])):
        ax.text(max(control, treatment) + 0.01, i - 0.2, f'{control:.2f}', va='center', fontsize=12)
        ax.text(min(control, treatment) - 0.12, i + 0.2, f'{treatment:.2f}', va='center', fontsize=12)

    # Set the y-tick labels to the variable names
    ax.set_yticks(y_pos)
    ax.set_yticklabels(m1.index)
    ax.set_ylabel('')

    return fig





    