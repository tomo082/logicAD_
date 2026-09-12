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

    def __post_init__(self):
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
