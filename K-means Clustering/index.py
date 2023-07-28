import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
# Import the necessary libraries: NumPy, scikit-learn, and the specific modules needed for this implementation of K-Means.

digits = load_digits()
data = scale(digits.data)
y = digits.target
# Load the digits dataset and scale the data so that all features have a mean of 0 and a variance of 1.
# y is assigned to digits.target

k = 10
samples, features = data.shape
# Set the number of clusters to 10 and get the number of samples and features in the dataset.

def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             sklearn.metrics.homogeneity_score(y, estimator.labels_),
             sklearn.metrics.completeness_score(y, estimator.labels_),
             sklearn.metrics.v_measure_score(y, estimator.labels_),
             sklearn.metrics.adjusted_rand_score(y, estimator.labels_),
             sklearn.metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             sklearn.metrics.silhouette_score(data, estimator.labels_,
                                              metric='euclidean')))
# Define a function bench_k_means that takes an estimator, a name, and data as input.
# Fit the estimator to the data.
# Print the name of the estimator, its inertia, and various scores (homogeneity, completeness, v-measure, adjusted rand index, adjusted mutual information, and silhouette score) using scikit-learn's metrics module.
# The rest of the code sets up the K-Means algorithm with the desired number of clusters and calls the bench_k_means function to evaluate the performance of the algorithm.

clf = KMeans(n_clusters=k, init="random", n_init=10)
bench_k_means(clf, "1", data)

# The code clf = KMeans(n_clusters=k, init="random", n_init=10) initializes the KMeans clustering algorithm with the number of clusters k, the initialization method init, and the number of times the algorithm will be run with different centroid seeds n_init.
# The init parameter specifies the method used to initialize the centroids. In this case, it is set to "random", which means that the initial centroids will be chosen randomly from the data points. Other initialization methods include "k-means++", which chooses the initial centroids to be far apart from each other, and "PCA", which uses principal component analysis to initialize the centroids.
# The n_init parameter specifies the number of times the algorithm will be run with different centroid seeds. This is important because the algorithm can converge to a suboptimal solution if the initial centroids are chosen poorly. Running the algorithm multiple times with different initial centroids helps to mitigate this problem.

# The bench_k_means(clf, "1", data) line calls the bench_k_means function with the initialized KMeans object clf, the name "1", and the data data. The bench_k_means function fits the KMeans object to the data and prints various scores that evaluate the performance of the algorithm.
# Overall, this code initializes and runs the KMeans clustering algorithm with the specified parameters and evaluates its performance using the bench_k_means function