import numpy as np
import scipy as sp
from sklearn import datasets
from matplotlib import pyplot as plt

iris = datasets.load_iris()
X = iris.data
y = iris.target

# Dividng sepal length / sepal width and petal length / petal width
for i in range(0, len(X)):
    X[i][0] /= X[i][1]
    X[i][2] /= X[i][3]

# Using only two features
data = X[:, [0, 2]]

# Plot shoing the data points in 3 clusters
plt.scatter(data[:, 0], data[:, 1], c=y, edgecolor='k')
plt.show()


def k_init(X, k):
    """ k-means++: initialization algorithm

    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    k: int
        The number of clusters

    Returns
    -------
    init_centers: array (k, d)
        The initialize centers for kmeans++
    """
    minimum_distance = [[0 for x in range(1)] for y in range(len(X))]
    centroids = np.empty((k, 2))
    # Take the first center randomly
    centroids[0] = data[np.random.randint(0, len(X)-1)]

    if (k == 1):
        return centroids
    for i in range(1, k):
        distance = [[0 for x in range(i)] for y in range(len(X))]
        for j in range(0, len(X)):
            empty = []
            for m in range(0, i):
                # Find distance between each point and each centroid
                distance[j][m] = dist(X[j, :], centroids[m])
                if (m == 0):
                    minimum_distance[j][0] = distance[j][m]
                else:
                    for alpha in range(0, m+1):
                        empty.append(distance[j][alpha])
                    # Take minimum distance for multiple centroids
                    minimum_distance[j][0] = min(empty)
        for n in range(0, len(X)):
            # Square of distance
            minimum_distance[n][0] = minimum_distance[n][0] ** 2
        for n in range(1, len(X)):
            # Cumulative distance (cdf)
            minimum_distance[n][0] += minimum_distance[n-1][0]

        # Pick a number at random
        new = np.random.randint(0, int(minimum_distance[len(X)-1][0]))
        count = 0
        while (1):
            if (new <= minimum_distance[counter][0]):
                break
            else:
                count += 1
        centroids[i] = X[count, :]
    return centroids


def k_means_pp(X, k, max_iter):
    """ k-means++ clustering algorithm

    step 1: call k_init() to initialize the centers
    step 2: iteratively refine the assignments

    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    k: int
        The number of clusters

    max_iter: int
        Maximum number of iteration

    Returns
    -------
    final_centers: array, shape (k, d)
        The final cluster centers
    """
    pass


def assign_data2clusters(X, C):
    """ Assignments of data to the clusters
    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    C: array, shape(k ,d)
        The final cluster centers

    Returns
    -------
    data_map: array, shape(n, k)
        The binary matrix A which shows the assignments of data points (X) to
        the input centers (C).
    """
    pass


def compute_objective(X, C):
    """ Compute the clustering objective for X and C
    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    C: array, shape(k ,d)
        The final cluster centers

    Returns
    -------
    accuracy: float
        The objective for the given assigments
    """
    pass
