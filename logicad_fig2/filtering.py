from __future__ import annotations

import random
import warnings

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from .embeddings import normalize


def select_description(embeddings, *, seed=42, n_neighbors=2):
    """Small-K LOF on normalized embeddings, followed by seeded random inlier choice.

    Neighbor count and Euclidean metric are reconstruction assumptions. With K=3
    and two neighbors LOF often cannot distinguish an outlier; we never force one.
    """
    matrix = normalize(embeddings)
    if matrix.ndim != 2:
        raise ValueError("LOF requires a matrix of description embeddings")
    count = len(matrix)
    neighbors = min(max(1, n_neighbors), count - 1)
    inliers = list(range(count))
    scores, fallback = [1.0] * count, None
    if count < 3:
        fallback = "fewer_than_three_descriptions"
    elif np.allclose(matrix, matrix[0], atol=1e-10, rtol=0):
        fallback = "identical_embeddings"
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                model = LocalOutlierFactor(n_neighbors=neighbors, contamination="auto", metric="euclidean")
                labels = model.fit_predict(matrix)
            factors = -model.negative_outlier_factor_
            if not np.isfinite(factors).all():
                fallback = "nonfinite_lof"
            else:
                scores = factors.tolist()
                if np.allclose(factors, factors[0]):
                    fallback = "tied_lof"
                else:
                    inliers = np.flatnonzero(labels == 1).tolist()
                    if not inliers:
                        inliers = list(range(count))
                        fallback = "no_inliers"
        except ValueError:
            fallback = "lof_unavailable"
    selected = random.Random(seed).choice(inliers)
    return {"selected_index": selected, "inlier_indices": inliers, "lof_scores": scores,
            "n_neighbors": neighbors, "seed": seed, "fallback": fallback}
