import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.linear_model import LinearRegression

df = pd.read_csv('/Users/nihar/Desktop/one pager/ventilator-pressure-prediction/train.csv')

# #Making dataframes from dataset of u_in, u_out, and pressure
# x_in = df[['u_in']].values 
# x_out = df[['u_out']].values
# y_pressure = df['pressure'].values

# #Making linear regression models for u_in and u_out, where each one will be x values, and corresponding pressure will be y value
# model_in = LinearRegression()
# model_in.fit(x_in, y_pressure)
# y_pred_in = model_in.predict(x_in)

# model_out = LinearRegression()
# model_out.fit(x_out, y_pressure)
# y_pred_out = model_out.predict(x_out)

# #Plotting lines of best fit along with original datapoints
# plt.subplot(1,2,1)
# plt.scatter(x_in, y_pressure, color='blue')
# plt.plot(x_in, y_pred_in, color='red', label='Regression')
# plt.xlabel('u_in')
# plt.ylabel('Pressure')
# plt.title('u_in vs Pressure')

# plt.subplot(1,2,2)
# plt.scatter(x_out, y_pressure, color='blue')
# plt.plot(x_out, y_pred_out, color='red', label='Regression')
# plt.xlabel('u_out')
# plt.ylabel('Pressure')
# plt.title('u_out vs Pressure')

# plt.tight_layout()
# plt.show()

#END OF INITIAL OBSERVATIONS

#MODELING AND TRAINING FOR BREATH_ID=21 ONLY

#Creating dataframes for x and y values
x=df[['u_in', 'u_out']].values
y=df['pressure'].values

#Fitting model to those x and y values
multiple_model=LinearRegression()
multiple_model.fit(x,y)

#Taking values for unique breath_id=21 from the dataframes
breath_id_unique=21
breath_21=df[df['breath_id']==breath_id_unique]
x_unique=breath_21[['u_in', 'u_out']].values
y_unique=breath_21['pressure'].values
time_step=breath_21['time_step'].values

#Predicting values for the unique breath_id using the model
y_predicted_new=multiple_model.predict(x_unique)

#Visualizations
# plt.plot(time_step, y_unique, color='blue')
# plt.plot(time_step, y_predicted_new, color='red')
# plt.xlabel("time_step")
# plt.ylabel("pressure")
# plt.title("Actual vs. Predicted Pressure breath_id=21")
# plt.show()

#Preparing matrices for training
x_matrix=breath_21[['u_in','u_out']].reset_index(drop=True).values
y_matrix=y_unique.reshape(-1,1)
y_predicted_matrix=y_predicted_new.reshape(-1,1)
w_matrix=np.zeros((2,1))
b_matrix=0.0

#Helper function for summation of differences between y_predicted and y_actual
def variance(y_actual, y_predicted):
    return y_predicted - y_actual

def dmse_dw(x_matrix, y_actual, y_predicted):
    m=len(y_actual)
    error=variance(y_actual, y_predicted)
    return (2/m)*x_matrix.T @ error

def dmse_db(y_actual, y_predicted):
    m=len(y_actual)
    error=variance(y_actual, y_predicted)
    return (2/m)*np.sum(error)

#iterate and write train funciton to minimze mse
def train(x_matrix, y_matrix, w_matrix, b_matrix, learning_rate=0.001, epochs=1000):
    for i in range(epochs):
        y_pred = x_matrix @ w_matrix + b_matrix
        dw = dmse_dw(x_matrix, y_matrix, y_pred)
        db = dmse_db(y_matrix, y_pred)
        w_matrix = w_matrix - learning_rate * dw
        b_matrix = b_matrix - learning_rate * db
    return w_matrix, b_matrix

#Training the model
w_trained, b_trained = train(x_matrix, y_matrix, w_matrix, b_matrix)

#Predictions using trained weights
y_manual_pred = x_matrix @ w_trained + b_trained

#Visualizing prediction from manual gradient descent
plt.plot(time_step, y_matrix, color='blue', label='Actual')
plt.plot(time_step, y_manual_pred, color='green', label='Gradient Descent Prediction')
plt.xlabel("time_step")
plt.ylabel("pressure")
plt.title("Manual Gradient Descent Prediction breath_id=21")
plt.legend()
plt.show()
