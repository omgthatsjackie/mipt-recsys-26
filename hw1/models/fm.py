import numpy as np
from scipy import sparse
from scipy.special import expit

from .base import Recommender


def fm_forward(features, bias, linear, factors):
    squared = features.multiply(features)
    projected = features @ factors
    scores = bias + np.asarray(features @ linear).ravel() + 0.5 * np.sum(
        projected**2 - squared @ (factors**2), axis=1)
    return scores, projected, squared


def fm_predict(features, bias, linear, factors):
    return fm_forward(features, bias, linear, factors)[0]


def fm_loss_gradient(features, labels, bias, linear, factors, regularization):
    scores, projected, squared = fm_forward(features, bias, linear, factors)
    error = (expit(scores) - labels) / len(labels)
    grad_factors = (features.T @ (error[:, None] * projected)
                    - (squared.T @ error)[:, None] * factors
                    + regularization * factors)
    grad_linear = features.T @ error + regularization * linear
    loss = np.mean(np.logaddexp(0, scores) - labels * scores)
    loss += 0.5 * regularization * (np.sum(linear**2) + np.sum(factors**2))
    return float(loss), (np.asarray(error.sum()), grad_linear, grad_factors)


def sample_training_pairs(counts, negative_ratio=2, seed=42):
    """Отрицательные пары берём только из неизвестных в текущей истории товаров."""
    rng = np.random.default_rng(seed)
    users, items = counts.nonzero()
    candidates = np.flatnonzero(np.asarray(counts.sum(axis=0)).ravel() > 0)
    negative_users = np.repeat(users, negative_ratio)
    negative_items = rng.choice(candidates, size=len(negative_users))
    invalid = np.asarray(counts[negative_users, negative_items]).ravel() > 0
    while invalid.any():
        negative_items[invalid] = rng.choice(candidates, size=int(invalid.sum()))
        invalid = np.asarray(counts[negative_users, negative_items]).ravel() > 0
    return (np.concatenate([users, negative_users]),
            np.concatenate([items, negative_items]),
            np.concatenate([np.ones(len(users)), np.zeros(len(negative_users))]))


class FactorizationMachine(Recommender):
    def __init__(self, factors=16, regularization=0.0001, learning_rate=0.01,
                 epochs=8, batch_size=4096, negative_ratio=2, seed=42):
        self.factors = factors
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.negative_ratio = negative_ratio
        self.seed = seed

    def fit(self, counts, user_features, item_features):
        self.user_features = sparse.csr_matrix(user_features, dtype=np.float64)
        self.item_features = sparse.csr_matrix(item_features, dtype=np.float64)
        users, items, labels = sample_training_pairs(counts, self.negative_ratio, self.seed)
        features = (self.user_features[users] + self.item_features[items]).tocsr()
        rng = np.random.default_rng(self.seed)
        n_features = features.shape[1]
        self.bias = np.array(-np.log(self.negative_ratio), dtype=np.float64)
        self.linear = np.zeros(n_features)
        self.embedding = rng.normal(0, 0.03, (n_features, self.factors))
        parameters = [self.bias, self.linear, self.embedding]
        first, second = [np.zeros_like(p) for p in parameters], [np.zeros_like(p) for p in parameters]
        step = 0
        self.losses_ = []
        for _ in range(self.epochs):
            order = rng.permutation(len(labels))
            total_loss = 0.0
            for start in range(0, len(labels), self.batch_size):
                idx = order[start : start + self.batch_size]
                loss, gradients = fm_loss_gradient(features[idx], labels[idx],
                                                   self.bias, self.linear, self.embedding,
                                                   self.regularization)
                total_loss += loss * len(idx)
                step += 1
                for p, m, v, gradient in zip(parameters, first, second, gradients):
                    m *= 0.9
                    m += 0.1 * gradient
                    v *= 0.999
                    v += 0.001 * gradient**2
                    p -= self.learning_rate * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
            self.losses_.append(total_loss / len(labels))
        self._prepare_scoring()
        return self

    def _prepare_scoring(self):
        def decompose(features):
            summed = features @ self.embedding
            offset = features @ self.linear
            offset += 0.5 * (summed**2 - features.multiply(features) @ (self.embedding**2)).sum(axis=1)
            return summed, offset
        self.user_vectors, self.user_offset = decompose(self.user_features)
        self.item_vectors, self.item_offset = decompose(self.item_features)

    def score(self, users):
        return (self.bias + self.user_offset[users, None] + self.item_offset[None, :]
                + self.user_vectors[users] @ self.item_vectors.T)
