import json

from logicad_fig2.cache import Cache
from logicad_fig2.config import Config
from logicad_fig2.dataset import discover_category
from logicad_fig2.evaluator import evaluate
from logicad_fig2.format_embedding import FeaturePipeline, FormatEmbedder
from logicad_fig2.roi import ROIExtractor
from logicad_fig2.text_extraction import TextExtractor
from test_dataset_metrics import make_dataset


class MockAPI:
    def __init__(self):
        self.calls = 0

    def chat(self, prompt, **kwargs):
        self.calls += 1
        if kwargs.get("images"):
            return {"text": "one apple on the left and granola on the right", "raw": {"mock": True}}
        return {"text": json.dumps({"left": [{"object": "apple", "count": 1, "present": True}],
                                    "right": [{"object": "granola", "count": None, "present": True}]})}

    def embed(self, texts, **kwargs):
        self.calls += 1
        return {"vectors": [[1., 2., 3.] for _ in texts]}


def test_five_run_end_to_end_resume_force_and_errors(tmp_path):
    data_root = make_dataset(tmp_path / "data")
    config = Config()
    client = MockAPI()
    output = tmp_path / "outputs"
    data = [discover_category(data_root, "breakfast_box")]

    def pipeline(force=False):
        cache = Cache(output / "cache", force=force)
        return FeaturePipeline(TextExtractor(client, ROIExtractor(config), cache, config),
                               FormatEmbedder(client, cache, config), cache, config)

    result = evaluate(data, pipeline(), config, output, num_runs=5)
    assert len(result["runs"]) == 5 and all(row["complete"] for row in result["runs"])
    assert result["summary"]["categories"]["breakfast_box"]["auroc"]["mean"] == .5
    calls = client.calls
    evaluate(data, pipeline(), config, output, num_runs=5, retry_errors=True)
    assert client.calls == calls
    evaluate(data, pipeline(force=True), config, output, num_runs=5, force=True)
    assert client.calls == 2 * calls  # forced query extraction still happens once, not once per reference
    from pathlib import Path

    destination = Path(result["output"])
    assert len(json.loads((destination / "predictions.json").read_text())) == 20
    assert (destination / "summary.csv").is_file()


def test_failed_image_does_not_bias_reported_full_metrics(tmp_path):
    data_root = make_dataset(tmp_path / "data")
    data = [discover_category(data_root, "breakfast_box")]
    config = Config()
    output = tmp_path / "out"
    cache = Cache(output / "cache")
    api = MockAPI()
    pipeline = FeaturePipeline(TextExtractor(api, ROIExtractor(config), cache, config),
                               FormatEmbedder(api, cache, config), cache, config)
    original = pipeline.image
    failed = data[0].queries[0].path

    def fail_once(path, *args, **kwargs):
        if path == failed:
            raise ValueError("Simulated query failure")
        return original(path, *args, **kwargs)

    pipeline.image = fail_once
    result = evaluate(data, pipeline, config, output)
    assert result["runs"][0]["failed"] == 1
    assert result["summary"]["macro_average"]["auroc"]["mean"] is None
    pipeline.image = original
    result = evaluate(data, pipeline, config, output, retry_errors=True)
    assert result["runs"][0]["failed"] == 0
