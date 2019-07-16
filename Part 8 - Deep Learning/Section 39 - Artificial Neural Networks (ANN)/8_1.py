# Artificial Neural Network

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding the Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1]) # more than 2
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] # Taking care of summy variable trap

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

###############################################################################
# Starting with the neural network

# Importing the libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
# 6 from 11+1/2, uniform fn to initialise the weigths with numbers close to 0, rule for rectifier activation function for hidden layers
#, 11 because 11 inputs and as this is the first layer it requies number of inputs

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the Output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid')) # activation will be softmax if more than two categories of dependent variable, here ony one

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # optimiser is the algorithm to find the weights, loos function to be optimised,
# binary_crossentrpy for binay outcome of dependent variable, categorical_crossentropy for more than two outcomes for dependent variable, metrics for evaluation of model as list

# Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100) # epochs are kind of rounds

###############################################################################
# Predicting the Test set results
y_pred = classifier.predict(X_test) # probabilities
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Accuracy
accuracy = (1528 + 207)/2000 # from cm