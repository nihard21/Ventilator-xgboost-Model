import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

#Reading train.csv dataset into dataframe

train = pd.read_csv('/Users/nihar/Desktop/Ventilator Model Tools/ventilator-pressure-prediction/train.csv')

#Cleaning up dataset, removing all values of 0 for u_out

train = train[train['u_out']==0]
target = ['pressure']
exclude = ['id', 'pressure', 'breath_id', 'time_step']
feature = [col for col in train.columns if col not in exclude]
x = train[feature]
y = train[target]

#identifying train and test sets

mid = len(train) // 2
train_data = train.iloc[:mid]
test_data = train.iloc[mid:]

#Identifying x and y for model training and testing

x_train = train_data[feature]
y_train = train_data[target]
x_test = test_data[feature]
y_test = test_data[target]

#xgboost model

model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42)
model.fit(x_train, y_train)

#Predicting pressure values for test dataset using model

test_pressure_pred = model.predict(x_test)

#Graphing results
plt.subplot(1,2,1)
plt.scatter(x_test['u_in'], test_pressure_pred, alpha=0.8, color='blue', label='Predicted')
plt.subplot(1,2,2)
plt.scatter(x_test['u_in'], y_test, alpha=0.2, color='red', label='Actual')
plt.xlabel("u_in")
plt.ylabel("pressure")
plt.title("Predicted vs Actual Pressure based on u_in")
plt.tight_layout()
plt.show()

#Determining error

mse = mean_squared_error(test_pressure_pred, y_test)
print(mse)

