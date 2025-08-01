import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

# Reading train.csv dataset into dataframe
train = pd.read_csv('/Users/nihar/Desktop/one pager/ventilator-pressure-prediction/train.csv')

# Feature engineering for improved model accuracy
train['u_in_cumsum'] = train.groupby('breath_id')['u_in'].cumsum()
train['u_in_lag1'] = train.groupby('breath_id')['u_in'].shift(1).fillna(0)
train['u_in_lag2'] = train.groupby('breath_id')['u_in'].shift(2).fillna(0)
train['u_in_mean'] = train.groupby('breath_id')['u_in'].transform('mean')
train['u_in_max'] = train.groupby('breath_id')['u_in'].transform('max')
train['u_in_min'] = train.groupby('breath_id')['u_in'].transform('min')
train['u_in_std'] = train.groupby('breath_id')['u_in'].transform('std').fillna(0)
train['u_in_u_out'] = train['u_in'] * train['u_out']

# Combining R and C into single categorical feature
train['R_C'] = train['R'].astype(str) + "_" + train['C'].astype(str)
train['R_C'] = pd.factorize(train['R_C'])[0]

# Cleaning up dataset, removing all values of 0 for u_out
train = train[train['u_out'] == 0]
target = ['pressure']
exclude = ['id', 'pressure', 'breath_id', 'time_step']
feature = [col for col in train.columns if col not in exclude]
x = train[feature]
y = train[target]

# Random train and test splits
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42  # 30% test, 70% train
)

# xgboost model
model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(x_train, y_train)

# Predicting pressure values for test dataset using model
test_pressure_pred = model.predict(x_test)

# Determining overall error without filtering
mse = mean_squared_error(y_test, test_pressure_pred)
print("Mean Squared Error:", mse)

# Add breath_id and time_step columns back to x_test from original train dataframe
x_test_with_id = x.loc[x_test.index].copy()
x_test_with_id['breath_id'] = train.loc[x_test.index, 'breath_id'].values
x_test_with_id['time_step'] = train.loc[x_test.index, 'time_step'].values

# Add actual and predicted pressures
x_test_with_id['actual_pressure'] = y_test.values.flatten()
x_test_with_id['predicted_pressure'] = test_pressure_pred.flatten()

# Calculate squared errors per row
x_test_with_id['squared_error'] = (x_test_with_id['actual_pressure'] - x_test_with_id['predicted_pressure']) ** 2

# Group by breath_id and calculate mean squared error per breath
mse_per_breath = x_test_with_id.groupby('breath_id')['squared_error'].mean()

# Find breath_id with lowest and highest MSE
lowest_mse_breath = mse_per_breath.idxmin()
lowest_mse_value = mse_per_breath.min()
highest_mse_breath = mse_per_breath.idxmax()
highest_mse_value = mse_per_breath.max()

print(f"Breath ID with lowest MSE: {lowest_mse_breath} (MSE = {lowest_mse_value:.4f})")
print(f"Breath ID with highest MSE: {highest_mse_breath} (MSE = {highest_mse_value:.4f})")

# Breath IDs to plot
breath_ids_to_plot = [lowest_mse_breath, highest_mse_breath]

plt.figure(figsize=(12, 6))

for i, bid in enumerate(breath_ids_to_plot):
    breath_df = x_test_with_id[x_test_with_id['breath_id'] == bid].sort_values('time_step')

    plt.subplot(1, 2, i + 1)
    plt.plot(breath_df['time_step'], breath_df['actual_pressure'], label='Actual Pressure', color='red')
    plt.plot(breath_df['time_step'], breath_df['predicted_pressure'], label='Predicted Pressure', color='blue')
    plt.title(f"breath_id: {bid}")
    plt.xlabel("time_step")
    plt.ylabel("pressure")
    plt.legend()

plt.suptitle("Actual vs. Predicted Pressure for Low and High MSE Breaths")
plt.tight_layout()
plt.show()
