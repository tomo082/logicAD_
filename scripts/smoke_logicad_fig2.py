#!/usr/bin/env python3
"""Offline integration demo. Synthetic images/responses; NEVER a benchmark result."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from logicad_fig2.cache import Cache, atomic_json
from logicad_fig2.config import Config
from logicad_fig2.dataset import discover_category
from logicad_fig2.evaluator import evaluate
from logicad_fig2.format_embedding import FeaturePipeline, FormatEmbedder
from logicad_fig2.prompts import CATEGORIES
from logicad_fig2.roi import ROIExtractor
from logicad_fig2.text_extraction import TextExtractor


class SyntheticClient:
    """Deterministic transport fixture; not an alternative detection model."""
    def __init__(self):
        self.calls = 0

    def chat(self, prompt, *, images=(), schema=None, **kwargs):
        self.calls += 1
        if images:
            text = f"Synthetic observation with color {images[0].getpixel((0, 0))}."
        else:
            def fill(node):
                if node["type"] == "object":
                    return {key: fill(value) for key, value in node["properties"].items()}
                if node["type"] == "array":
                    return []
                if node["type"] == "string":
                    return "synthetic_unknown"
                return None

            text = json.dumps(fill(schema))
        return {"text": text, "raw": {"synthetic_test_only": True}}

    def embed(self, texts, **kwargs):
        self.calls += 1
        return {"vectors": [[int(byte) + 1 for byte in hashlib.sha256(t.encode()).digest()] for t in texts],
                "raw": {"synthetic_test_only": True}}


def smoke(root):
    root = Path(root)
    for category in CATEGORIES:
        for split, count in (("train/good", 5), ("test/good", 2), ("test/logical_anomalies", 2)):
            folder = root / "synthetic_dataset" / category / split
            folder.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                Image.new("RGB", (32, 32), (index * 30, 60, 90)).save(folder / f"{index:03d}.png")
    config = Config(disable_roi=True)
    cache = Cache(root / "cache")
    client = SyntheticClient()
    extractor = TextExtractor(client, ROIExtractor(config), cache, config)
    pipeline = FeaturePipeline(extractor, FormatEmbedder(client, cache, config), cache, config)
    datasets = [discover_category(root / "synthetic_dataset", cat) for cat in CATEGORIES]
    result = evaluate(datasets, pipeline, config, root, num_runs=5)
    calls = client.calls
    resumed = evaluate(datasets, pipeline, config, root, num_runs=5, retry_errors=True)
    assert client.calls == calls
    assert all(run["complete"] for run in resumed["runs"])
    result["synthetic_test_only"] = True
    result["mock_api_calls"] = calls
    result["resume_additional_calls"] = 0
    atomic_json(root / "SMOKE_TEST_ONLY.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/offline_smoke"))
    print(json.dumps(smoke(parser.parse_args().output_dir), indent=2))
