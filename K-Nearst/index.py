# Car Evaluation Model
# This code trains a KNearestNeighbors classifier on the car evaluation dataset to predict car quality labels.

# Imports and Data Loading
# Import core data science libraries like Pandas, NumPy, and scikit-learn. Load the CSV dataset into a Pandas DataFrame.

# Data Preprocessing
# Encode categorical string features to numeric using sklearn's LabelEncoder. Convert feature columns and labels to lists to structure as X (features) and y (labels) arrays.

# Train Test Split
# Split the data into 90% train and 10% test sets using sklearn's train_test_split.

# Model Training
# Train a KNeighborsClassifier model on the training data with 9 neighbors.

# Model Evaluation
# Evaluate on the test data and print accuracy score. Make predictions and print actual vs predicted values.

# KNeighbors
# Get the nearest neighbors for a sample using model.kneighbors() and print the results.

# Notes
# Sample workflow for training and evaluating a KNN classifier model in scikit-learn
# Uses standard preprocessing, model training, evaluation, and prediction steps
# Can be extended by tuning model hyperparameters, adding new features, etc    


# n = model.kneighbors([x_test[x]], 9, True)
# Uses the trained KNeighbors classifier model to get the 9 nearest neighbors for a specific test data point.

# Breaking it down:

# model is the trained KNN classifier
# .kneighbors() is a method to query the nearest neighbors
# [x_test[x]] passes a single test data point
# 9 specifies we want 9 nearest neighbors
# True returns the distances as well
# The results are stored in n.

# So what it is doing is:

# Taking one x_test data point
# Finding its 9 nearest neighbors in the model
# Returning the indexes and distances of neighbors
# This allows us to inspect the local neighborhood of a data point to understand the model better.

# We can print n to see the actual neighbor indexes and distances for the specified point.

# In summary, it retrieves the k neighbors for a data point from the trained kNN model for analysis.

import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car-data.csv")
print(data.head())

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

print(acc)

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 9, True)
    print("N: ", n)


