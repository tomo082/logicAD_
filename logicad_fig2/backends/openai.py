"""The four roles share the existing lazy OpenAI transport and usage/retry log."""
from __future__ import annotations

from .base import GenerationConfig
from ..prompts import VISUAL_CONTEXT


class OpenAIAdapter:
    def __init__(self, client, model):
        self.client, self.model = client, model

    def signature(self):
        return {"backend": "openai", "model": self.model, "adapter_version": 1}

    def unload(self):
        pass  # No local model allocation; the shared transport is left reusable.

    def _generation_kwargs(self, generation: GenerationConfig):
        return {"temperature": generation.temperature if generation.do_sample else 0.0,
                "top_p": generation.top_p if generation.do_sample else 1.0,
                "max_tokens": generation.max_tokens, "seed": generation.seed}


class OpenAIVisionBackend(OpenAIAdapter):
    def signature(self):
        return {**super().signature(), "visual_context": VISUAL_CONTEXT, "multi_image_strategy": "native"}

    def describe(self, images, prompt, generation):
        response = self.client.chat(VISUAL_CONTEXT + prompt, model=self.model, images=images,
                                    **self._generation_kwargs(generation))
        return {**response, "metadata": {"backend": self.signature(), "multi_image_strategy": "native",
                                          "image_count": len(images), "generation": generation.signature()}}


class OpenAITextBackend(OpenAIAdapter):
    def generate(self, prompt, schema, generation):
        response = self.client.chat(prompt, model=self.model, schema=schema, **self._generation_kwargs(generation))
        return {**response, "metadata": {"backend": self.signature(), "generation": generation.signature()}}


class OpenAIEmbeddingBackend(OpenAIAdapter):
    def embed(self, texts):
        return {**self.client.embed(texts, model=self.model), "metadata": {"backend": self.signature()}}
