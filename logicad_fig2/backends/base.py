from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, Sequence, TypedDict


class GenerationResult(TypedDict, total=False):
    text: str
    raw: dict
    metadata: dict
    finish_reason: str
    refusal: str | None


class EmbeddingResult(TypedDict):
    vectors: list[list[float]]
    metadata: dict
    raw: dict


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1600
    do_sample: bool = False
    seed: int | None = None

    def signature(self):
        return asdict(self)


class ModelBackend(Protocol):
    def signature(self) -> dict[str, Any]: ...

    def unload(self) -> None: ...


class VisionLanguageBackend(ModelBackend, Protocol):
    def describe(self, images: Sequence, prompt: str, generation: GenerationConfig) -> GenerationResult: ...


class TextGenerationBackend(ModelBackend, Protocol):
    def generate(self, prompt: str, schema: dict | None, generation: GenerationConfig) -> GenerationResult: ...


class EmbeddingBackend(ModelBackend, Protocol):
    def embed(self, texts: list[str]) -> EmbeddingResult: ...


@dataclass
class BackendBundle:
    vision: VisionLanguageBackend
    formatter: TextGenerationBackend
    embedding: EmbeddingBackend
    logic: TextGenerationBackend | None = None
    unload_on_switch: bool = False
    _active: ModelBackend | None = field(default=None, init=False, repr=False)

    def activate(self, role: str):
        backend = getattr(self, role)
        if backend is None:
            raise ValueError(f"Backend role {role} is not configured")
        if self.unload_on_switch and self._active is not None and self._active is not backend:
            self._active.unload()
        self._active = backend
        return backend

    def signature(self):
        return {role: getattr(self, role).signature() if getattr(self, role) else None
                for role in ("vision", "formatter", "embedding", "logic")}

    def unload(self):
        seen = set()
        for backend in (self.vision, self.formatter, self.embedding, self.logic):
            if backend is not None and id(backend) not in seen:
                backend.unload()
                seen.add(id(backend))
        self._active = None


def generation_for(config, role):
    """Resolve per-stage overrides while retaining historical common flags."""
    is_vision = role == "vlm"
    temperature = getattr(config, role + "_temperature", None)
    top_p = getattr(config, role + "_top_p", None)
    max_tokens = getattr(config, role + "_max_tokens", None)
    do_sample = getattr(config, role + "_do_sample", None)
    temperature = temperature if temperature is not None else (config.temperature if is_vision else 0.0)
    top_p = top_p if top_p is not None else (config.top_p if is_vision else 1.0)
    do_sample = do_sample if do_sample is not None else is_vision and temperature > 0
    return GenerationConfig(temperature, top_p, max_tokens or config.max_tokens, do_sample, config.seed)
