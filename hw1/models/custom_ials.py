import numpy as np

from .ials import ImplicitALS


class RecencyRatingALS(ImplicitALS):
    def __init__(self, half_life=180.0, floor=0.25, **kwargs):
        super().__init__(**kwargs)
        self.half_life = half_life
        self.floor = floor

    def build_preferences(self, counts, last_review, ratings, cutoff):
        counts = counts.tocsr(copy=True)
        dates, ratings = last_review.tocsr(copy=True), ratings.tocsr(copy=True)
        for matrix in [counts, dates, ratings]:
            matrix.sort_indices()
        age_days = (float(cutoff) - dates.data) / 86400.0
        recency = np.exp2(-age_days / self.half_life)
        enthusiasm = (ratings.data - 3.0) / 2.0
        preferences = counts.astype(np.float64)
        preferences.data = self.floor + (1.0 - self.floor) * (0.7 * recency + 0.3 * enthusiasm)
        return preferences

    def fit(self, counts, last_review, ratings, cutoff):
        preferences = self.build_preferences(counts, last_review, ratings, cutoff)
        return super().fit(counts, preferences=preferences)
