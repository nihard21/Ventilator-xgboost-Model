#Importing necessary libraries/functions
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

#Reading train.csv dataset into dataframe
train=pd.read_csv('/Users/nihar/Desktop/Ventilator Model Tools/ventilator-pressure-prediction/train.csv')

#Feature engineering for improved model accuracy
train['u_in_cumsum']=train.groupby('breath_id')['u_in'].cumsum()
train['u_in_lag1']=train.groupby('breath_id')['u_in'].shift(1).fillna(0)
train['u_in_lag2']=train.groupby('breath_id')['u_in'].shift(2).fillna(0)
train['u_in_mean']=train.groupby('breath_id')['u_in'].transform('mean')
train['u_in_max']=train.groupby('breath_id')['u_in'].transform('max')
train['u_in_min']=train.groupby('breath_id')['u_in'].transform('min')
train['u_in_std']=train.groupby('breath_id')['u_in'].transform('std').fillna(0)
train['u_in_u_out']=train['u_in']*train['u_out']

#Combining R and C into single categorical feature
train['R_C']=train['R'].astype(str)+"_"+train['C'].astype(str)
train['R_C']=pd.factorize(train['R_C'])[0]

#Identifying inputs and outputs for model
target=['pressure']
exclude=['id', 'pressure']
feature=[col for col in train.columns if col not in exclude]
x=train[feature]
y=train[target]

#Random train and test splits, 30% test/70% train
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3, random_state=42)  

#xgboost model
model=xgb.XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42)
model.fit(x_train, y_train)

#Predicting pressure values for test dataset using model
test_pressure_pred=model.predict(x_test)

#Compute MSE for each point
mse_per_point=(y_test.values.flatten()-test_pressure_pred.flatten())**2

#Remove highest 5% of MSEs
threshold=np.percentile(mse_per_point, 95)
mask=mse_per_point<threshold
x_test=x_test[mask]
y_test=y_test[mask]
test_pressure_pred=test_pressure_pred[mask]

#Determining overall MSE
mse=mean_squared_error(test_pressure_pred, y_test)
print("Mean Squared Error:", mse)

#Graphing results
plt.subplot(1, 2, 1)
plt.scatter(x_test['u_in'], test_pressure_pred, color='blue')
plt.xlabel("u_in")
plt.ylabel("Predicted pressure")
plt.title("Predicted pressure vs u_in")

plt.subplot(1, 2, 2)
plt.scatter(x_test['u_in'], y_test, color='red')
plt.xlabel("u_in")
plt.ylabel("Actual pressure")
plt.title("Actual pressure vs u_in")

plt.suptitle("Predicted vs. Actual pressure Based on u_in")
plt.tight_layout()
plt.show()

plt.hist((y_test.values.flatten()-test_pressure_pred.flatten())**2, bins=100, color='red')
plt.xlabel("MSE")
plt.ylabel("Frequency")
plt.title("Histogram of MSEs (Excluding Top 5%)")
plt.show()