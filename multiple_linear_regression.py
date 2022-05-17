# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
#print(X)

# Encoding categorical data column 3 (states) one hot encoder turns this column into three
# and Inserts those columns in the front of the data set
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
#print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
# Making a single prediction
# (for example the profit of a startup with R&D Spend = 160000, Administration Spend = 130000,
# Marketing Spend = 300000 and State = 'California')
print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))
# Important note 1: Notice that the values of the features were all input in a double pair of square brackets.
# That's because the "predict" method always expects a 2D array as the format of its inputs.
# And putting our values into a double pair of square brackets makes the input exactly a 2D array.
# Simply put:

# 1,0,0,160000,130000,300000→scalars
#
# [1,0,0,160000,130000,300000]→1D array
#
# [[1,0,0,160000,130000,300000]]→2D array

# Important note 2:
# Notice also that the "California" state was not input as a string in the last column but
# as "1, 0, 0" in the first three columns. That's because of course the predict method expects
# the one-hot-encoded values of the state,
# and as we see in the second row of the matrix of features X,
# "California" was encoded as "1, 0, 0". And be careful to include these values in the first three columns,
# not the last three ones, because the dummy variables are always created in the first columns.