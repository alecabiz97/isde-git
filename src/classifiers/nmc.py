import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class NMC(object):
    """
    Class implementing the Nearest Mean Centroid (NMC) classifier.

    This classifier estimates one centroid per class from the training data,
    and predicts the label of a never-before-seen (test) point based on its
    closest centroid.

    Attributes
    -----------------
    - centroids: read-only attribute containing the centroid values estimated
        after training

    Methods
    -----------------
    - fit(x,y) estimates centroids from the training data
    - predict(x) predicts the class labels on testing points

    """

    def __init__(self):
        self._centroids = None
        self._class_labels = None  # class labels may not be contiguous indices

    @property
    def centroids(self):
        return self._centroids

    @property
    def class_labels(self):
        return self._class_labels


    def fit(self, x_tr, y_tr):
        """Estimate the centroid for each class from the training data"""
        labels = np.unique(y_tr)
        centroids = np.zeros(shape=(labels.size, x_tr.shape[1]))
        for i, label in enumerate(labels):
            centroids[i, :] = x_tr[y_tr == label, :].mean(axis=0)
        self._centroids = centroids

    def predict(self, x_ts):

        if self._centroids is None:
            raise ValueError("The classifier is not trained. Call fit first!")

        dist_euclidean = euclidean_distances(x_ts, self._centroids)
        yc = np.argmin(dist_euclidean, axis=1)
        return yc

