# Breast Cancer Classification
# This code trains an SVM model on the breast cancer dataset to classify tumors as malignant or benign.

# Imports
# Import scikit-learn's SVM, metrics, and dataset modules for the model, evaluation, and data. Also import KNeighborsClassifier for comparison.

# Load Data
# Load the breast cancer dataset using sklearn's load_breast_cancer() function. It returns the features matrix (x) and response vector (y).

# Train/Test Split
# Split the data into 80% training and 20% test sets using sklearn's train_test_split() function.

# Define Classes
# The target classes are malignant (0) and benign (1). Store them in a list for reference.

# Model Training
# Train a Support Vector Classifier (SVC) model on the training data using a linear kernel and C=2.

# Can also try polynomial kernel or KNN by uncommenting those lines.

# Prediction and Evaluation
# Use model to predict labels for test set. Compare to true y_test labels to calculate accuracy score.

# Notes
# Example of standard machine learning workflow: load data, preprocess, train, evaluate.
# SVM with linear kernel performs well on this dataset.
# Kernel and hyperparameters can be tuned further to optimize accuracy.
# KNN is an alternative non-linear model that could be tested.

import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)
# print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

classes = ['malignant', 'benign']

clf = svm.SVC(kernel="linear", C=2)
# clf = svm.SVC(kernel="poly", degree=2)
# clf = KNeighborsClassifier(n_neighbors=9)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)

print(acc)