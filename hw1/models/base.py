import numpy as np


def top_items(scores, seen, candidate_mask, k=10):
    eligible = candidate_mask.copy()
    eligible[seen] = False
    candidates = np.flatnonzero(eligible)
    order = np.lexsort((candidates, -scores[candidates]))
    return candidates[order[:k]]


class Recommender:
    def recommend(self, users, seen, candidates, k=10, batch_size=128):
        users = np.asarray(users, dtype=np.int64)
        candidates = np.asarray(candidates, dtype=bool)
        result = np.empty((len(users), k), dtype=np.int32)
        for start in range(0, len(users), batch_size):
            batch = users[start : start + batch_size]
            scores = np.asarray(self.score(batch), dtype=np.float64)
            for offset, user in enumerate(batch):
                begin, end = seen.indptr[user:user + 2]
                top = top_items(scores[offset], seen.indices[begin:end], candidates, k)
                result[start + offset] = top
        return result
