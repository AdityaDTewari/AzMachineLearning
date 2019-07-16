# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = [] # empty list for input in the algorithm
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

# Training the Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2) # for a product purchased 3 times a day, in a week 3*7/7500 (/transaction number)

# Visualizing the result
results = list(rules) # rules are already sorted by their relevance

for i in results:
    print(i)
    print('**********') # printing the results as not visible in variable explorer
