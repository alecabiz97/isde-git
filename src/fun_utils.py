from pandas import read_csv
import numpy as np


def load_data(filename):
    """
    Load data from a csv file

    Parameters
    ----------
    filename : string
        Filename to be loaded.

    Returns
    -------
    X : ndarray
        the data matrix.

    y : ndarray
        the labels of each sample.
    """
    data = read_csv(filename)
    z = np.array(data)
    y = z[:, 0]
    X = z[:, 1:]
    return X, y


def split_data(x, y, tr_fraction=0.5):
    """
    Split the data x, y into two random subsets

    """
    pass

def fit(x_tr, y_tr):
    """Estimate the centroid for each class from the training data"""
    labels = np.unique(y_tr)
    centroids = np.zeros(shape=(labels.size, x_tr.shape[1]))
    for i, label in enumerate(labels):
        centroids[i, :] = x_tr[y_tr == label, :].mean(axis=0)  # centr. for class i
        return centroids, labels
        centroids, labels = fit(xtr, ytr)
        plot_ten_images(centroids, labels, shape=(28, 28))