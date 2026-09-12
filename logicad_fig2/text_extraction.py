from __future__ import annotations

from pathlib import Path

from .cache import Cache, file_digest, fingerprint
from .config import Config
from .embeddings import cached_embeddings
from .filtering import select_description
from .prompts import get_prompts


class TextExtractor:
    def __init__(self, client, roi, cache: Cache, config: Config):
        self.client, self.roi, self.cache, self.config = client, roi, cache, config

    def describe(self, path: Path, category: str):
        """Commit every raw response immediately, resuming incomplete K batches."""
        path = Path(path)
        c, prompts = self.config, get_prompts(category)
        feature = c.feature_prompt or prompts.feature
        settings = {"model": c.vlm_model, "prompt": prompts.extraction, "feature": feature,
                    "temperature": c.temperature, "top_p": c.top_p, "max_tokens": c.max_tokens,
                    "roi": self.roi.signature(), "version": 1}
        image_id = fingerprint({"path": path.resolve().as_posix(), "sha256": file_digest(path)})
        prefix = f"{category}/{path.stem}-{image_id}/{fingerprint(settings)}"
        key = prefix + "/descriptions.json"
        record = self.cache.get(key)
        if record is None:
            record = {"image": str(path.resolve()), "settings": settings, "descriptions": [], "failed_responses": []}
        if len(record["descriptions"]) < c.k:
            images, roi_metadata = self.roi.extract(path, feature, category)
            record["roi"] = roi_metadata
            for _ in range(len(record["descriptions"]), c.k):
                response = self.client.chat(prompts.extraction, model=c.vlm_model, images=images,
                                            temperature=c.temperature, top_p=c.top_p)
                if not response.get("text", "").strip() or response.get("refusal") or response.get("finish_reason") == "length":
                    record["failed_responses"].append(response)
                    self.cache.put(key, record)
                    raise ValueError("Empty, refused or truncated image description; raw response cached")
                record["descriptions"].append(response)
                self.cache.put(key, record)
        return {"prefix": prefix, "image_id": image_id, "texts": [r["text"] for r in record["descriptions"][:c.k]],
                "roi": record.get("roi"), "descriptions_cache": key}

    def extract(self, path: Path, category: str):
        result = self.describe(path, category)
        c = self.config
        vectors, embedding_key = cached_embeddings(self.client, self.cache, result["texts"], c.embedding_model)
        seed = c.seed + int(result["image_id"][:8], 16)
        selection_settings = {"embedding": embedding_key, "seed": seed, "neighbors": c.lof_neighbors,
                              "vectors": vectors.tolist(), "version": 1}
        key = result["prefix"] + f"/selection-{fingerprint(selection_settings)}.json"
        selected = self.cache.get(key)
        if selected is None:
            selected = select_description(vectors, seed=seed, n_neighbors=c.lof_neighbors)
            selected["text"] = result["texts"][selected["selected_index"]]
            selected["embedding_cache"] = embedding_key
            self.cache.put(key, selected)
        return {**result, "selection": selected, "selected_text": selected["text"], "selection_cache": key}
