This project is based off the problem statement of a completed Kaggle competition, the original prompt can be found here: https://www.kaggle.com/competitions/ventilator-pressure-prediction/overview

***Used only the train dataset, as the test dataset does not have pressure values

Intial Steps:
-Read the problem prompt and identified goal, which was to predict ventilator pressure based off previous data
-Wrote simple notes about each variable to quickly identify their function (documented in "notes")

Using linear regression for intial observations:
-Began programming, graphed pressures based off u_in for singular breaths to understand patterns
-Used scikit-learn linear regression functions for basic models/predictions

Using xgboost for improved model accuracy:
-Chose to use xgboost for increased accuracy, linear regression consistently returned errors>100
-Intially tested with half of the dataset for train and other half for test
-Started off with a basic xgboost model (Model 1), which used all variables except 'id', 'pressure', 'breath_id', and 'time_step' as model inputs
-Added multiple features for improved model accuracy (Model 2)
-Removed top 5% of MSEs to account for outliers and further decrease error (Model 3)
-Final model (Model 4), chose to include 'breath_id' and 'time_step' in inputs, used random train/test splits with 70% for train, 30% for test

Results (MSEs):
-Linear Regression: >100
-Model 1: 32.066039083385554
-Model 2: 4.20245476933641
-Model 3: 1.3253897842206432
-Model 4 (Final): 0.7577091890221975

Visualizations:
-Check "screenshots" folder in repository

-"overall prediction based off primary input": visualization of pressure based off u_in, the primary control value
-"error distribution": visualization of error distribution for each breath
-"high low mse examples": left graph is an example of an accurate breath (error=0.0104), right graph is example of an inaccurate breath (error=193.4240)

