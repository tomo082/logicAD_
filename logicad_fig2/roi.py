from __future__ import annotations

import importlib.util
import logging
import math
from pathlib import Path

from PIL import Image

from .cache import file_digest
from .config import Config


def crop_regions(image, boxes, padding=1.5, limit=32):
    """Convert normalized cx,cy,w,h boxes to clamped crops (original retained elsewhere)."""
    crops, pixels = [], []
    w, h = image.size
    for box in boxes:
        cx, cy, bw, bh = map(float, box)
        if not all(math.isfinite(x) for x in (cx, cy, bw, bh)) or bw <= 0 or bh <= 0:
            continue
        bounds = (max(0, math.floor((cx - bw * padding / 2) * w)),
                  max(0, math.floor((cy - bh * padding / 2) * h)),
                  min(w, math.ceil((cx + bw * padding / 2) * w)),
                  min(h, math.ceil((cy + bh * padding / 2) * h)))
        if bounds[2] > bounds[0] and bounds[3] > bounds[1] and list(bounds) not in pixels:
            crops.append(image.crop(bounds))
            pixels.append(list(bounds))
            if len(crops) == limit:
                break
    return crops, pixels


class ROIExtractor:
    def __init__(self, config: Config):
        self.config = config
        self._model = None
        self._load_failed = False
        self._checkpoint_digest = None

    def signature(self):
        c = self.config
        checkpoint = Path(c.gdino_checkpoint) if c.gdino_checkpoint else None
        if checkpoint and checkpoint.is_file() and self._checkpoint_digest is None:
            self._checkpoint_digest = file_digest(checkpoint)
        return {"checkpoint": str(checkpoint), "checkpoint_hash": self._checkpoint_digest,
                "config": c.gdino_config, "device": c.device, "disabled": c.disable_roi,
                "padding": c.roi_padding, "limit": c.max_rois,
                "box": c.box_threshold, "text": c.text_threshold, "version": 1}

    def _load(self):
        import torch

        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if self.config.device == "auto" else self.config.device
        # Reuse the existing loader without importing anomalib/__init__ or its models.
        path = Path(__file__).resolve().parents[1] / "src/anomalib/utils/llm/grounddino.py"
        spec = importlib.util.spec_from_file_location("logicad_legacy_gdino", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._model = module.load_gdino_model(ckpt_path=self.config.gdino_checkpoint,
                                             cfg=self.config.gdino_config, device="cpu")
        self._model.to(self.device).eval()

    def extract(self, path: Path, feature_prompt: str, category: str):
        with Image.open(path) as source:
            original = source.convert("RGB")
        metadata = {"status": "original_only", "boxes": [], "phrases": []}
        if self.config.disable_roi:
            metadata["reason"] = "disabled"
            return [original], metadata
        if not self.config.gdino_checkpoint:
            metadata["reason"] = "no_checkpoint"
            return [original], metadata
        if self._load_failed:
            metadata["reason"] = "model_unavailable"
            return [original], metadata
        try:
            if self._model is None:
                self._load()
            from groundingdino.util.inference import load_image, predict

            _, tensor = load_image(str(path))
            box_threshold = self.config.box_threshold
            text_threshold = self.config.text_threshold
            if box_threshold is None:
                box_threshold = 0.1 if category == "pushpins" else 0.2
            if text_threshold is None:
                text_threshold = 0.1 if category == "pushpins" else 0.3
            boxes, logits, phrases = predict(model=self._model, image=tensor, caption=feature_prompt,
                                             box_threshold=box_threshold, text_threshold=text_threshold,
                                             device=self.device)
            order = logits.argsort(descending=True).tolist()
            boxes = boxes.detach().cpu().tolist()
            patches, pixels = crop_regions(original, [boxes[i] for i in order], self.config.roi_padding,
                                            self.config.max_rois)
            metadata.update(status="ok" if patches else "no_detections", boxes=pixels,
                            phrases=[phrases[i] for i in order], device=self.device,
                            thresholds={"box": box_threshold, "text": text_threshold})
            return [original] + patches, metadata
        except Exception as exc:
            if self._model is None:
                self._load_failed = True
            logging.warning("ROI failed for %s (%s); using original image", path, exc)
            metadata.update(reason="roi_error", error_type=type(exc).__name__, error=str(exc))
            return [original], metadata
