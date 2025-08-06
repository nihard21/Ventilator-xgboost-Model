# Ventilator xgboost Model 

This project is based off the problem statement of a 2021 Kaggle competition run by Google Brain, the original prompt can be found here: <https://www.kaggle.com/competitions/ventilator-pressure-prediction/overview>

## Background

Predicting ventilator pressure from sensor data is an important step toward improving how mechanical ventilators are controlled. Ventilators help patients breathe by pumping oxygen into their lungs through a tube in the windpipe, but managing them requires constant attention from clinicians. Developing models that can predict and adjust pressure automatically could reduce difficulty and make treatments more accessible. The datasets used in this challenge come from an accurate simulation, where a modified open-source ventilator was connected to an artificial bellows test lung. My personal project uses only the training dataset from the Kaggle challenge, ignoring the testing data due to the absence of pressure values. 

## Variables 

These are my following notes about each of the variables found in the datasets, simplified for quick understanding:

* id=row index
* breath_id=breath index
* R=change in pressure based on air flow (airway resistance)
* C=change in volume based on pressure (lung compliance)
* time_step=time from start of breath
* u_in=percentage of opening of inspiratory solenoid valve for incoming breath
* u_out=boolean indicating whether expiratory valve is open or closed
* pressure=airway pressure


I began by conducting an exploratory analysis, understanding the function of each variable and visualizing the relationship between one of the controls, u_in, and pressure across individual breaths to identify patterns.
​Initially, I used linear regression from scikit-learn, which established basic understanding, but produced mean squared errors (MSE) consistently exceeding 100, requiring a more powerful model for increased accuracy.

To improve predictive performance, we adopted XGBoost, progressively refining the model through four stages. The first model excluded non-predictive identifiers such as id, breath_id, pressure, and time_step, achieving an MSE of 32.07. Subsequent iterations incorporated feature engineering (Model 2) to capture dynamic relationships, reducing the MSE to 4.20. Outlier removal targeting the top 5% of errors (Model 3) further decreased the MSE to 1.33. Finally, the inclusion of all available features, including breath_id and time_step, combined with a randomized 70/30 train-test split, yielded the final model (Model 4) with an MSE of 0.76. These results demonstrate a clear progression from simple linear models to a highly accurate gradient-boosted approach, illustrating the effectiveness of iterative model refinement in solving complex time-dependent regression tasks.

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

