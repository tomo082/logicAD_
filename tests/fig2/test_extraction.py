from PIL import Image

from logicad_fig2.cache import Cache
from logicad_fig2.config import Config
from logicad_fig2.roi import ROIExtractor, crop_regions
from logicad_fig2.text_extraction import TextExtractor
import pytest


class FakeClient:
    def __init__(self):
        self.calls = 0
        self.images = []

    def chat(self, prompt, **kwargs):
        self.calls += 1
        self.images = kwargs.get("images", [])
        return {"text": "On the left there is one apple; granola is on the right.", "raw": {"mock": True}}

    def embed(self, texts, **kwargs):
        self.calls += 1
        return {"vectors": [[1, 0] for _ in texts], "raw": {"mock": True}}


def test_single_image_description_without_groundingdino(tmp_path):
    path = tmp_path / "test.png"
    Image.new("RGB", (32, 24)).save(path)
    config = Config(k=1)
    client = FakeClient()
    extractor = TextExtractor(client, ROIExtractor(config), Cache(tmp_path / "cache"), config)
    result = extractor.describe(path, "breakfast_box")
    assert result["texts"] and client.calls == 1
    assert len(client.images) == 1 and client.images[0].size == (32, 24)
    assert result["roi"]["reason"] == "no_checkpoint"
    extractor.describe(path, "breakfast_box")
    assert client.calls == 1


def test_clamped_roi_keeps_valid_patches():
    image = Image.new("RGB", (100, 80))
    patches, boxes = crop_regions(image, [[0, 0, .4, .4], [.5, .5, 0, .1], [5, 5, .1, .1]])
    assert len(patches) == 1
    assert boxes[0][:2] == [0, 0]
    assert 30 <= patches[0].width <= 31 and 24 <= patches[0].height <= 25


def test_k3_resume_across_processes(tmp_path):
    path = tmp_path / "test.png"
    Image.new("RGB", (32, 24)).save(path)
    config = Config()
    client = FakeClient()
    original_chat = client.chat

    def failing(prompt, **kwargs):
        if client.calls == 1:
            raise RuntimeError("simulated outage")
        return original_chat(prompt, **kwargs)

    client.chat = failing
    root = tmp_path / "cache"
    with pytest.raises(RuntimeError):
        TextExtractor(client, ROIExtractor(config), Cache(root), config).extract(path, "breakfast_box")
    client.chat = original_chat
    result = TextExtractor(client, ROIExtractor(config), Cache(root), config).extract(path, "breakfast_box")
    assert len(result["texts"]) == 3
    assert client.calls == 4  # three descriptions, one embedding request
    TextExtractor(client, ROIExtractor(config), Cache(root), config).extract(path, "breakfast_box")
    assert client.calls == 4
