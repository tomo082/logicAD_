from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    vlm_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-large"
    format_model: str = "gpt-4o"
    logic_model: str = "gpt-4o"
    k: int = 3
    temperature: float = 0.05
    top_p: float = 0.1
    max_tokens: int = 1600
    seed: int = 42
    timeout: float = 90.0
    max_retries: int = 4
    json_retries: int = 2
    lof_neighbors: int = 2
    gdino_checkpoint: str | None = None
    gdino_config: str = "swint"
    device: str = "auto"
    disable_roi: bool = False
    feature_prompt: str | None = None
    box_threshold: float | None = None
    text_threshold: float | None = None
    roi_padding: float = 1.5
    max_rois: int = 32
    enable_reasoner: bool = False
    prover9_path: str | None = None
    mace4_path: str | None = None
    prover_timeout: int = 10
    max_mis_checks: int = 64
    normal_rules: str = "reference"

    vlm_backend: str = "openai"
    vlm_revision: str | None = None
    formatter_backend: str = "openai"
    formatter_revision: str | None = None
    embedding_backend: str = "openai"
    embedding_revision: str | None = None
    logic_backend: str = "openai"
    logic_revision: str | None = None
    vlm_temperature: float | None = None
    vlm_top_p: float | None = None
    vlm_max_tokens: int | None = None
    vlm_do_sample: bool | None = None
    formatter_temperature: float | None = None
    formatter_top_p: float | None = None
    formatter_max_tokens: int | None = None
    formatter_do_sample: bool | None = None
    logic_temperature: float | None = None
    logic_top_p: float | None = None
    logic_max_tokens: int | None = None
    logic_do_sample: bool | None = None
    formatter_model: str | None = None
    dtype: str = "auto"
    device_map: str = "auto"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    local_files_only: bool = False
    offload_folder: str | None = None
    max_memory: dict | None = None
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32
    vlm_image_strategy: str = "sequential"
    unload_on_switch: bool = False

    def __post_init__(self):
        if self.formatter_model is not None:
            if self.format_model != "gpt-4o" and self.format_model != self.formatter_model:
                raise ValueError("Conflicting format_model and formatter_model")
            object.__setattr__(self, "format_model", self.formatter_model)
        object.__setattr__(self, "formatter_model", self.format_model)
        for role in ("vlm", "formatter", "embedding", "logic"):
            if getattr(self, role + "_backend") not in ("openai", "hf"):
                raise ValueError(f"Unknown {role} backend")
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Choose either 4-bit or 8-bit quantization")
        if self.dtype not in ("auto", "float16", "bfloat16", "float32"):
            raise ValueError("Unsupported dtype")
        if self.vlm_image_strategy not in ("sequential", "native"):
            raise ValueError("Unsupported VLM image strategy")
        if self.embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be positive")
        if self.max_memory is not None and not isinstance(self.max_memory, dict):
            raise ValueError("max_memory must be a JSON object")
        if self.device_map not in ("auto", "balanced", "balanced_low_0", "sequential", "none"):
            raise ValueError("Unsupported device_map strategy")
        for role in ("vlm", "formatter", "logic"):
            from .backends.base import generation_for
            gen = generation_for(self, role)
            tokens = getattr(self, role + "_max_tokens")
            if (tokens is not None and tokens < 1) or not 0 <= gen.temperature <= 2 or not 0 < gen.top_p <= 1:
                raise ValueError(f"Invalid {role} generation settings")
            if gen.do_sample and gen.temperature <= 0:
                raise ValueError(f"{role} sampling requires positive temperature")
        for name in ("k", "max_tokens", "lof_neighbors", "max_rois", "prover_timeout"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.timeout <= 0 or self.roi_padding <= 0:
            raise ValueError("timeout and roi_padding must be positive")
        if min(self.max_retries, self.json_retries, self.max_mis_checks) < 0:
            raise ValueError("retry/check counts must be nonnegative")
        if not 0 <= self.temperature <= 2 or not 0 < self.top_p <= 1:
            raise ValueError("Invalid temperature/top_p")
        for value in (self.box_threshold, self.text_threshold):
            if value is not None and not 0 <= value <= 1:
                raise ValueError("ROI thresholds must be between 0 and 1")
