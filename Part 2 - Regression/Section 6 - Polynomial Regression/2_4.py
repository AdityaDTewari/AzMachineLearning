# Polinomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values # 1:2 to get X as a vector
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
# We don't have enough observations/information to train/test

# No feature scaling required because same library (linear)

# Fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting polinomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X) # creating a matrix with x^2 x^3... also a column of 1s for the constant term
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Visualizing the linear regression results
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Linear regression visual')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

# Visulaizing the polynomial regression results
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color='blue')
plt.title('Polynomial regression visual')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

# Visulaizing the polynomial regression results (smoother curve and better resolution)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1) # conversion to matrix
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Polynomial regression visual')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

# Predicting a new result with linear regression
lin_reg.predict(6.5)

# Predicting a new result with Polynomial regression
lin_reg2.predict(poly_reg.fit_transform(6.5))