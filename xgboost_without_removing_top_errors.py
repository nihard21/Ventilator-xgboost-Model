import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Reading train.csv dataset into dataframe

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

train['R_C'] = train['R'].astype(str) + "_" + train['C'].astype(str)
train['R_C'] = pd.factorize(train['R_C'])[0]

# Cleaning up dataset, removing all values of 0 for u_out

train = train[train['u_out'] == 0]
target = ['pressure']
exclude = ['id', 'pressure', 'breath_id', 'time_step']
feature = [col for col in train.columns if col not in exclude]
x = train[feature]
y = train[target]

# Identifying train and test sets

mid = len(train) // 2
train_data = train.iloc[:mid]
test_data = train.iloc[mid:]

# Identifying x and y for model training and testing

x_train = train_data[feature]
y_train = train_data[target]
x_test = test_data[feature]
y_test = test_data[target]

# xgboost model

model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42)
model.fit(x_train, y_train)

# Predicting pressure values for test dataset using model

test_pressure_pred = model.predict(x_test)

# Graphing results

plt.subplot(1, 2, 1)
plt.scatter(x_test['u_in'], test_pressure_pred, color='blue')
plt.xlabel("u_in")
plt.ylabel("Predicted Pressure")
plt.title("Predicted Pressure vs u_in")

plt.subplot(1, 2, 2)
plt.scatter(x_test['u_in'], y_test, color='red')
plt.xlabel("u_in")
plt.ylabel("Actual Pressure")
plt.title("Actual Pressure vs u_in")

plt.suptitle("XGBoost Model: Predicted vs Actual Pressure Based on u_in")
plt.tight_layout()
plt.show()

# Determining error

mse = mean_squared_error(test_pressure_pred, y_test)
print("Mean Squared Error:", mse)

#Next steps
#pressure vs time actual vs predicted for different breaths, show breaths where prediction is good and bad
#mse histogram
#put code on github
#report should be in readme, add images of screenshots
