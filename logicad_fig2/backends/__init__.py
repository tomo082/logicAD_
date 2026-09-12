"""Lazy, interchangeable model adapters. Importing this package loads no weights."""
from .base import BackendBundle, EmbeddingBackend, GenerationConfig, TextGenerationBackend, VisionLanguageBackend

__all__ = ["BackendBundle", "EmbeddingBackend", "GenerationConfig", "TextGenerationBackend", "VisionLanguageBackend"]
