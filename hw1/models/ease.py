import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .base import Recommender


class EASE(Recommender):
    def __init__(self, regularization=100.0):
        self.regularization = regularization

    def fit(self, counts):
        self.history = counts.astype(np.float64).copy().tocsr()
        self.history.data[:] = 1.0
        gram = (self.history.T @ self.history).toarray()
        gram.flat[:: gram.shape[0] + 1] += self.regularization
        factor = cho_factor(gram, lower=True, check_finite=False)
        precision = cho_solve(factor, np.eye(gram.shape[0]), check_finite=False)
        self.coefficients = -precision / np.diag(precision)[None, :]
        np.fill_diagonal(self.coefficients, 0.0)
        return self

    def score(self, users):
        return np.asarray(self.history[users] @ self.coefficients)
