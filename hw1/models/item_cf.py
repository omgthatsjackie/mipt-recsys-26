import numpy as np
from scipy import sparse

from .base import Recommender


def cosine_with_overlap(matrix, observation_matrix, axis, min_common=1, shrinkage=10):
    if axis == "item":
        gram = (matrix.T @ matrix).toarray()
        overlap = (observation_matrix.T @ observation_matrix).toarray()
    else:
        gram = (matrix @ matrix.T).toarray()
        overlap = (observation_matrix @ observation_matrix.T).toarray()
    norms = np.sqrt(np.maximum(np.diag(gram), 0))
    denominator = np.outer(norms, norms)
    cosine = np.divide(gram, denominator, out=np.zeros_like(gram), where=denominator > 0)
    similarity = np.divide(cosine * overlap, overlap + shrinkage,
                           out=np.zeros_like(gram), where=overlap + shrinkage > 0)
    similarity[overlap < min_common] = 0
    np.fill_diagonal(similarity, 0)
    return similarity, overlap


class ItemCF(Recommender):
    def __init__(self, neighbors=80, shrinkage=20.0):
        self.neighbors = neighbors
        self.shrinkage = shrinkage

    def fit(self, counts):
        self.history = counts.astype(np.float64).copy().tocsr()
        self.history.data[:] = 1.0
        sim, _ = cosine_with_overlap(self.history, self.history, "item",
                                      min_common=1, shrinkage=self.shrinkage)
        rows, cols, values = [], [], []
        for item in range(sim.shape[0]):
            idx = np.argsort(-sim[item], kind="stable")[: self.neighbors]
            idx = idx[sim[item, idx] > 0]
            rows.extend([item] * len(idx))
            cols.extend(idx)
            values.extend(sim[item, idx])
        # Строка i хранит соседей целевого товара i; score = R @ S.T
        self.similarity = sparse.csr_matrix((values, (rows, cols)), shape=sim.shape)
        return self

    def score(self, users):
        return (self.history[users] @ self.similarity.T).toarray()
