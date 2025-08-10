# Ventilator XGBoost Model 

This project is based off the problem statement of a 2021 Kaggle competition run by Google Brain, the original prompt can be found here: <https://www.kaggle.com/competitions/ventilator-pressure-prediction/overview>

### Background

Predicting ventilator pressure from sensor data is an important step toward improving how mechanical ventilators are controlled. Ventilators help patients breathe by pumping oxygen into their lungs through a tube in the windpipe, but managing them requires constant attention from clinicians. Developing models that can predict and adjust pressure automatically could reduce difficulty and make treatments more accessible. The datasets used in this challenge come from a simulation where a modified open-source ventilator was connected to an artificial bellows test lung, and was used to simulate over 125,000 unique breaths, or over 6,000,000 individual values. Participants were expected to use the data recorded from the sensors to create a model, which accurately predicts ventilator pressure. My personal project uses only the training dataset from the Kaggle challenge, ignoring the testing data due to the absence of pressure values. 

### Variables 

The dataset consists of the following variables:

* id=row index
* breath_id=breath index
* time_step=time elapsed from start of breath (even though the dataset uses the name time_step, this is not the change in time, but the actual time)
* R=change in pressure based on air flow (airway resistance, which does not depend on time_step)
* C=change in volume based on pressure (lung compliance, which does not depend on time_step)
* u_in=percentage of opening of inspiratory solenoid valve for incoming breath for each time_step
* u_out=boolean indicating whether expiratory valve is open or closed for each time_step
* pressure=airway pressure for each time_step

### Experiment Procedure

I began by conducting an exploratory analysis, understanding the function of each variable and visualizing the relationship between one of the controls, u_in, and pressure across individual breaths to identify patterns.
​Initially, I used linear regression from scikit-learn, which established basic understanding, but produced mean squared errors (MSE) consistently exceeding 100. Using multiple linear regression did not significantly reduce MSE, requiring a more powerful model for increased accuracy. To improve model performance, I switched to XGBoost. The first model excluded non-predictive identifiers such as id, breath_id, pressure, and time_step, already significantly reducing MSE. The next iteration incorporated feature engineering to account for dynamic relationships, reducing the MSE even further. The exact features I ended up using included cumulative sum, previous 2 u_in values, mean, max, min, standard deviation, and combining R and C values into a single categorical feature. Next, I plotted the histogram of the MSE values, and found that there were a few cases with very high error. This indicated that while most breaths were modeled accurately, there were some breaths that were disproportionately contributing to the MSE. The visualization below displays an accurate and an inaccurate prediction of a breath. Therefore, I removed the top 5% of error cases, which further reduced the MSE. The final histogram of the MSE can be found below. Finally, the inclusion of all available features, combined with a randomized 70/30 train-test split, resulted in my final model, with the lowest MSE yet. The results demonstrate a progression from simple linear models to a highly accurate gradient-boosted approach, showing the effectiveness of model refinement in solving complex regression tasks.

### Results

| Model        | Result (MSE)  |
| ------------- |:-------------:|
| Scikit Learn (linear regression)    |  >100 |
| XGBoost (excluding non-predictive identifiers)      | 32.066039083385554     |
| XGBoost (incorporated feature engineering) | 4.20245476933641      |
| XGBoost (outlier removal) | 1.3253897842206432     |
| XGBoost (inclusion of all available features) | 0.7577091890221975      |

### Visualizations

##### Example of accurate and inaccurate breath (MSE is 0.0104 and 193.4240 respectively)
![alt text](https://github.com/nihard21/Ventilator-xgboost-Model/blob/main/visualizations/mse_examples.png?raw=true)

##### Visualization of pressure based off u_in, the primary control value
![alt text](https://github.com/nihard21/Ventilator-xgboost-Model/blob/main/visualizations/overall_prediction.png?raw=true)

##### Visualization of MSE distrubution for individual breaths
![alt text](https://github.com/nihard21/Ventilator-xgboost-Model/blob/main/visualizations/mse_distributions.png?raw=true)
