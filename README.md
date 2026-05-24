# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset and initialize weights, bias, learning rate, and number of iterations.
2.Calculate the predicted values using the sigmoid function: Y^=11+e−(wX+b)\hat{Y} = \frac{1}{1 + e^{-(wX + b)}}Y^=1+e−(wX+b)1
​3.Compute gradients for weights and bias, then update them using gradient descent formulas. 
4.Repeat the process until iterations complete, then display final weights, bias, and model accuracy.
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SUJITH MANO M
RegisterNumber:  212225220109
*/
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("Placement_Data.csv")
data['status'] = data['status'].map({'Placed': 1, 'Not Placed': 0})
X = data[['ssc_p', 'mba_p']].values
y = data['status'].values
scaler = StandardScaler()
X = scaler.fit_transform(X)
m = len(y)
X = np.c_[np.ones(m), X]
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def cost_function(X, y, theta):
    h = sigmoid(X @ theta)
    return (-1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h))
theta = np.zeros(X.shape[1])
alpha = 0.1
cost_history = []
for i in range(500):
    z = X @ theta
    h = sigmoid(z)
    gradient = (1/m) * X.T @ (h - y)
    theta = theta - alpha * gradient
    
    cost = cost_function(X, y, theta)
    cost_history.append(cost)
y_pred = (sigmoid(X @ theta) >= 0.5).astype(int)
accuracy = np.mean(y_pred == y) * 100
print("Weights:", theta)
print("Accuracy:", accuracy, "%")
plt.figure()
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Logistic Regression using Gradient Descent")
plt.show()
```

## Output:
<img width="842" height="627" alt="image" src="https://github.com/user-attachments/assets/b15c339c-519c-47f6-86ef-fdc635419d54" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

