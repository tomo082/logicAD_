from __future__ import annotations

from .cache import canonical_json, fingerprint
from .embeddings import cached_embeddings, normalize
from .prompts import get_prompts
from .structured import cached_structured


def canonical_features(value):
    """Normalize dictionary/array serialization so enumeration order is not a feature."""
    if isinstance(value, dict):
        return {key: canonical_features(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return sorted((canonical_features(item) for item in value), key=canonical_json)
    return value


class FormatEmbedder:
    def __init__(self, client, cache, config):
        self.client, self.cache, self.config = client, cache, config

    def format_and_embed(self, text, category):
        prompts, c = get_prompts(category), self.config
        signature = {"text": text, "prompt": prompts.format, "schema": prompts.schema, "model": c.format_model,
                     "max_tokens": c.max_tokens, "version": 1}
        key = f"formatted/{category}/{fingerprint(signature)}.json"
        value = cached_structured(self.client, self.cache, key, prompt=prompts.format + "\nObservations:\n" + text,
                                  schema=prompts.schema, model=c.format_model, retries=c.json_retries)
        canonical = canonical_json(canonical_features(value))
        vectors, embedding_key = cached_embeddings(self.client, self.cache, [canonical], c.embedding_model)
        return {"formatted": value, "canonical": canonical, "embedding": normalize(vectors[0]).tolist(),
                "formatted_cache": key, "embedding_cache": embedding_key}


class FeaturePipeline:
    def __init__(self, extractor, formatter, cache, config):
        self.extractor, self.formatter, self.cache, self.config = extractor, formatter, cache, config

    def image(self, path, category, *, reference=False):
        extracted = self.extractor.extract(path, category)
        formatted = self.formatter.format_and_embed(extracted["selected_text"], category)
        result = {**extracted, **formatted}
        if reference:
            key = f"references/{category}/{fingerprint(result)}.json"
            if self.cache.get(key) is None:
                self.cache.put(key, result)
            result["reference_cache"] = key
        return result
