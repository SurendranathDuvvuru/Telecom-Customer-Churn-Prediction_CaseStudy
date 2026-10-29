# Telecom Customer Churn Prediction
## Overview
Analyze telecom customer behavior and use machine learning to identify customers at risk of churn.
Notebook: `Surendranath_Duvvuru.ipynb`.
## Objectives
- Explore customer usage and clean the data.
- Select high-value customers using June–July recharge behavior.
- Define churn as zero call and internet usage in August.
- Engineer behavioral features and compare classification models.
## Tools
Python, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.
Methods: Logistic Regression, Random Forest, PCA, and cross-validation.
## Workflow
1. Inspect data types, missing values, and descriptive statistics.
2. Impute missing values and remove unsuitable/highly incomplete columns.
3. Filter customers at or above the 70th percentile of average recharge.
4. Create the churn target and month-over-month features; perform EDA and cap outliers.
5. Split data, encode categorical features, and train models.
6. Apply PCA, tune models, and rank features with Random Forest.
7. Compare sensitivity, specificity, ROC-AUC, and cross-validation scores.
## Results
Saved outputs show high ROC-AUC; PCA + Logistic Regression reports higher churn sensitivity than Random Forest experiments.
Rerun the notebook and check for data leakage before relying on results.
## Supplied CSV
`Surendranath_Duvvuru.csv` has 30,000 rows and columns `id` and `churn_probability`.
All churn values are `0`; there are no missing values.
This CSV differs from the raw `test.csv` structure referenced in the notebook; verify its purpose before using it as input.
## Run
Install: `pip install pandas numpy matplotlib seaborn scikit-learn jupyter`
Add the correct source dataset, then run `jupyter notebook Surendranath_Duvvuru.ipynb`.
## Business Use
Use churn insights to guide retention offers, review roaming/calling packages, and investigate customer complaints.
