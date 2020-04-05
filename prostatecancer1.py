import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
data=pd.read_csv("C:\\Users\\srisa\\Desktop\\prostate_cancer.csv")
print(data)
data.head()
data.shape
data.info()
data.dropna(inplace=True)
data.drop(['id'], axis = 1, inplace = True)
data.diagnosis_result = [1 if each == "M" else 0 for each in data.diagnosis_result]
data.info()
data.shape
Y=data.diagnosis_result.values
Y.shape
X_data = data.drop(['diagnosis_result'], axis = 1)
X_data.shape
X = (X_data -np.min(X_data))/(np.max(X_data)-np.min(X_data)).values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42)
X_train = X_train.T
X_test = X_test.T
Y_train = Y_train.T
Y_test = Y_test.T

print("x train: ",X_train.shape)
print("x test: ",X_test.shape)
print("y train: ",Y_train.shape)
print("y test: ",Y_test.shape)
def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w, b
def sigmoid(z):
    Y_head = 1/(1+np.exp(-z))
    return Y_head
def forward_backward_propagation(w,b,X_train,Y_train):
    z = np.dot(w.T,x_train) + b
    Y_head = sigmoid(z)
    loss = -Y_train*np.log(Y_head)-(1-Y_train)*np.log(1-Y_head)
    cost = (np.sum(loss))/X_train.shape[1]
    derivative_weight = (np.dot(X_train,((Y_head-y_train).T)))/X_train.shape[1]
    derivative_bias = np.sum(Y_head-Y_train)/X_train.shape[1]
    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}
    return cost,gradients
def update(w, b, X_train, Y_train, learning_rate,number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    for i in range(number_of_iterarion):
        cost,gradients = forward_backward_propagation(w,b,X_train,Y_train)
        cost_list.append(cost)
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        parameters = {"weight": w,"bias": b}
    return parameters, gradients, cost_list
def predict(w,b,X_test):
    z = sigmoid(np.dot(w.T,X_test)+b)
    y_prediction = np.zeros((1,X_test.shape[1]))
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            y_prediction[0,i] = 0
        else:
            y_prediction[0,i] = 1

    return y_prediction
def logistic_regression(X_train, Y_train, X_test, Y_test, learning_rate ,  num_iterations):
    dimension =  X_train.shape[0]
    w,b = initialize_weights_and_bias(dimension)
    parameters, gradients, cost_list = update(w, b, X_train, Y_train, learning_rate,num_iterations)
    
    Y_prediction_test = predict(parameters["weight"],parameters["bias"],X_test)
    Y_prediction_train = predict(parameters["weight"],parameters["bias"],X_train)
    train_accuracy = (100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100)
    test_accuracy = (100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100)
    return train_accuracy, test_accuracy
a = 0
for i in range(1,11):
    for j in range(1,11):
        train_accuracy = logistic_regression(X_train, Y_train, X_test, Y_test, learning_rate = i*0.1, num_iterations = j*50)[0]
        test_accuracy = logistic_regression(X_train, Y_train, X_test, Y_test, learning_rate = i*0.1, num_iterations = j*50)[1]
        num_iterations = j*50
        learning_rate = i*0.1
        if test_accuracy > a:
            a = test_accuracy
            b = train_accuracy
            c = learning_rate
            d = num_iterations
print('learning_rate: ', c,'num_iterations: ',d )
print('train_accuracy:',b, 'test_accuracy:', a)

from sklearn import linear_model
logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 500)
print("train accuracy: {} ".format(logreg.fit(X_train.T, Y_train.T).score(X_train.T, Y_train.T)))
print("test accuracy: {} ".format(logreg.fit(X_train.T, Y_train.T).score(X_test.T, Y_test.T)))