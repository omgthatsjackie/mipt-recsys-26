import numpy as np
from scipy import sparse

from .base import Recommender


class ImplicitALS(Recommender):
    def __init__(self, factors=32, regularization=10.0, alpha=20.0, epochs=12, seed=42):
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.epochs = epochs
        self.seed = seed

    def _update(self, other, strength, preference, batch_size=32):
        dimension = other.shape[1]
        base = other.T @ other + self.regularization * np.eye(dimension)
        updated = np.empty((strength.shape[0], dimension), dtype=np.float64)
        for begin in range(0, strength.shape[0], batch_size):
            end = min(begin + batch_size, strength.shape[0])
            lhs = np.broadcast_to(base, (end - begin, dimension, dimension)).copy()
            rhs = np.zeros((end - begin, dimension))
            for offset, row in enumerate(range(begin, end)):
                start, stop = strength.indptr[row:row + 2]
                idx = strength.indices[start:stop]
                local = other[idx]
                extra = self.alpha * strength.data[start:stop]
                lhs[offset] += local.T @ (extra[:, None] * local)
                rhs[offset] = local.T @ ((1.0 + extra) * preference.data[start:stop])
            updated[begin:end] = np.linalg.solve(lhs, rhs[..., None])[..., 0]
        return updated

    def fit(self, counts, preferences=None):
        counts = counts.astype(np.float64).tocsr(copy=True)
        counts.sort_indices()
        strength = counts.copy()
        strength.data = np.log1p(strength.data)
        if preferences is None:
            preferences = counts.copy()
            preferences.data[:] = 1.0
        else:
            preferences = sparse.csr_matrix(preferences, dtype=np.float64)
            preferences.sort_indices()
        rng = np.random.default_rng(self.seed)
        self.item_factors = rng.normal(0, 0.05, (counts.shape[1], self.factors))
        self.item_factors[np.asarray(counts.sum(axis=0)).ravel() == 0] = 0
        strength_t, preference_t = strength.T.tocsr(), preferences.T.tocsr()
        for _ in range(self.epochs):
            self.user_factors = self._update(self.item_factors, strength, preferences)
            self.item_factors = self._update(self.user_factors, strength_t, preference_t)
        self.preferences_ = preferences
        return self

    def score(self, users):
        return self.user_factors[users] @ self.item_factors.T
