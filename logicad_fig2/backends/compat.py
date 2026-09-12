"""Compatibility for callers injecting the original chat/embed client contract."""
from __future__ import annotations

from .base import BackendBundle
from .openai import OpenAIEmbeddingBackend, OpenAITextBackend, OpenAIVisionBackend


class LegacyIdentity:
    def signature(self):
        cls = type(self.client)
        return {"backend": "legacy-client", "client": cls.__module__ + "." + cls.__qualname__, "model": self.model}

    def _generation_kwargs(self, generation):
        # Old injected clients need not accept max_tokens or seed.
        return {"temperature": generation.temperature, "top_p": generation.top_p}


class LegacyVision(LegacyIdentity, OpenAIVisionBackend):
    pass


class LegacyText(LegacyIdentity, OpenAITextBackend):
    pass


class LegacyEmbedding(LegacyIdentity, OpenAIEmbeddingBackend):
    pass


def as_backends(value, config):
    if isinstance(value, BackendBundle):
        return value
    return BackendBundle(LegacyVision(value, config.vlm_model), LegacyText(value, config.format_model),
                         LegacyEmbedding(value, config.embedding_model), LegacyText(value, config.logic_model))
