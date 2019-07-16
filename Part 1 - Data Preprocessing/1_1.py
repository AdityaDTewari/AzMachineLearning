# Data Preprocessing

# Importing the libraries
import numpy as np # is the shortcut
import matplotlib.pyplot as plt # pyplot is the sublibrary
import pandas as pd # pandas to import and manage data

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values # independent values matrix, rows,columns
y = dataset.iloc[:, 3].values # dependent vector

# Taking care of missing data
from sklearn.preprocessing import Imputer # imputer class for missing data
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0) # object of class imputer
imputer = imputer.fit(X[:, 1:3]) # applying to the X only at places where required
X[:, 1:3] = imputer.transform(X[:, 1:3]) # applying changes (replacing by mean of column)

# Encoding categorical data
# Encoding independent variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# Encoding dependent variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state=0) # random_state for getting same result as another person

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)