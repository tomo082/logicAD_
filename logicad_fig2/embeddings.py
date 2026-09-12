from __future__ import annotations

import numpy as np

from .cache import fingerprint


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float64)
    if vector.ndim not in (1, 2) or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("Embeddings must be a finite nonempty vector or matrix")
    lengths = np.linalg.norm(vector, axis=-1, keepdims=True)
    if (lengths <= 1e-12).any() or not np.isfinite(lengths).all():
        raise ValueError("Cannot normalize zero/invalid embeddings")
    return vector / lengths


def cosine_anomaly_score(normal, query) -> float:
    normal, query = normalize(normal), normalize(query)
    if normal.ndim != 1 or query.shape != normal.shape:
        raise ValueError("Expected two embedding vectors of the same dimension")
    # Clamp only floating point drift. The equation has range [0,2], not [0,1].
    return float(1.0 - np.clip(np.dot(normal, query), -1.0, 1.0))


def cached_embeddings(client, cache, texts, model):
    key = f"embeddings/{fingerprint({'model': model, 'texts': texts})}.json"
    value = cache.get(key)
    if value is None:
        value = client.embed(texts, model=model)
        # Retain invalid provider answers as attempts, not successful cache hits,
        # so --retry-errors can recover without discarding every successful stage.
        attempts_key = key.removesuffix(".json") + ".attempts.json"
        cache.put(attempts_key, [*(cache.get(attempts_key) or []), value])
    vectors = np.asarray(value["vectors"], dtype=np.float64)
    if vectors.ndim != 2 or len(vectors) != len(texts):
        raise ValueError("Embedding batch shape does not match descriptions")
    normalize(vectors)
    cache.put(key, value)
    return vectors, key
