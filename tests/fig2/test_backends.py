import builtins
import json
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from logicad_fig2.backends.base import BackendBundle, GenerationConfig, generation_for
from logicad_fig2.backends.factory import build_backends
from logicad_fig2.backends.hf_common import HFOptions, pretrained_kwargs
from logicad_fig2.backends.hf_embedding import HFEmbeddingBackend
from logicad_fig2.backends.hf_text import HFTextBackend
from logicad_fig2.backends.hf_vlm import HFVisionBackend
from logicad_fig2.cache import Cache
from logicad_fig2.config import Config
from logicad_fig2.dataset import discover_category
from logicad_fig2.embeddings import cached_embeddings
from logicad_fig2.evaluator import evaluate
from logicad_fig2.format_embedding import FeaturePipeline, FormatEmbedder
from logicad_fig2.roi import ROIExtractor
from logicad_fig2.structured import cached_structured
from logicad_fig2.text_extraction import TextExtractor
from test_dataset_metrics import make_dataset


class FakeVisionBackend:
    def __init__(self):
        self.calls, self.unloads = [], 0

    def signature(self):
        return {"backend": "fake", "role": "vision"}

    def describe(self, images, prompt, generation):
        self.calls.append(generation)
        return {"text": "one apple on the left", "metadata": {"images": len(images)}}

    def unload(self):
        self.unloads += 1


class FakeTextBackend(FakeVisionBackend):
    def signature(self):
        return {"backend": "fake", "role": "text"}

    def generate(self, prompt, schema, generation):
        self.calls.append(prompt)
        return {"text": json.dumps({"left": [{"object": "apple", "count": 1, "present": True}], "right": []})}


class FakeEmbeddingBackend(FakeVisionBackend):
    def signature(self):
        return {"backend": "fake", "role": "embedding"}

    def embed(self, texts):
        self.calls.append(texts)
        return {"vectors": [[1., 2., 3., 4., 5.] for text in texts]}


def test_role_protocol_e2e_duplicates_k_and_resume(tmp_path):
    data = [discover_category(make_dataset(tmp_path / "data"), "breakfast_box")]
    config, cache = Config(disable_roi=True), Cache(tmp_path / "cache")
    bundle = BackendBundle(FakeVisionBackend(), FakeTextBackend(), FakeEmbeddingBackend(), unload_on_switch=True)
    pipeline = FeaturePipeline(TextExtractor(bundle, ROIExtractor(config), cache, config),
                               FormatEmbedder(bundle, cache, config), cache, config)
    result = evaluate(data, pipeline, config, tmp_path / "output")
    assert all(row["complete"] for row in result["runs"])
    assert result["summary"]["categories"]["breakfast_box"]["auroc"]["mean"] == .5
    calls = len(bundle.vision.calls)
    assert calls >= 6 and calls % 3 == 0
    assert len({g.seed for g in bundle.vision.calls[:3]}) == 3
    evaluate(data, pipeline, config, tmp_path / "output", retry_errors=True)
    assert len(bundle.vision.calls) == calls
    assert bundle.vision.unloads > 0


def test_factory_hybrid_then_local_without_sdk_or_heavy_imports(monkeypatch):
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        level = args[3] if len(args) > 3 else kwargs.get("level", 0)
        if not level:
            assert name.split(".")[0] not in {"torch", "transformers", "sentence_transformers", "openai"}
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = Config(vlm_backend="hf", vlm_model="llava-hf/llava-v1.6-vicuna-13b-hf")
    mixed = build_backends(config, openai_client=object())
    assert isinstance(mixed.vision, HFVisionBackend)
    assert mixed.formatter.signature()["backend"] == "openai"
    assert mixed.embedding.signature()["model"] == "text-embedding-3-large"
    assert mixed.logic is None and mixed.vision._model is None
    local = build_backends(replace(config, formatter_backend="hf", formatter_model="local/text",
                                  embedding_backend="hf", embedding_model="local/embed",
                                  logic_backend="hf", logic_model="local/logic", enable_reasoner=True))
    assert isinstance(local.formatter, HFTextBackend)
    assert isinstance(local.logic, HFTextBackend)
    assert isinstance(local.embedding, HFEmbeddingBackend)
    assert all(b._model is None for b in (local.vision, local.formatter, local.embedding, local.logic))


@pytest.fixture
def fake_runtime(monkeypatch):
    seeds, generated, decoded = [], [], []
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                            random=SimpleNamespace(fork_rng=lambda **kw: nullcontext()),
                            inference_mode=nullcontext, manual_seed=seeds.append,
                            float32="float32", float16="float16")
    monkeypatch.setitem(__import__("sys").modules, "torch", torch)
    class Tensor:
        shape = (1, 2)
        def to(self, *args, **kwargs):
            return self
    class Processor:
        chat_template = "template"
        prompts = []
        def apply_chat_template(self, messages, **kwargs):
            self.prompts.append(messages)
            return "rendered"
        def __call__(self, *args, **kwargs):
            return {"input_ids": Tensor()}
        def decode(self, ids, **kwargs):
            decoded.append(ids.tolist())
            return '{"left": [], "right": []}'
    class Model:
        device, dtype = "cpu", "float32"
        config = SimpleNamespace(is_encoder_decoder=False)
        generation_config = SimpleNamespace(eos_token_id=99)
        def eval(self):
            pass
        def get_input_embeddings(self):
            return SimpleNamespace(weight=SimpleNamespace(device="cpu"))
        def generate(self, **kwargs):
            generated.append(kwargs)
            return np.array([[101, 102, 7, 99]])
    loads = []
    def loader(model_id, options):
        loads.append((model_id, options))
        return Model(), Processor()
    return SimpleNamespace(loader=loader, loads=loads, seeds=seeds, generated=generated, decoded=decoded,
                           torch=torch)


@pytest.mark.parametrize("strategy,count", [("sequential", 3), ("native", 1)])
def test_hf_images_lazy_generation_and_unload(fake_runtime, strategy, count):
    backend = HFVisionBackend("test/vlm", loader=fake_runtime.loader, image_strategy=strategy)
    assert not fake_runtime.loads
    response = backend.describe([object(), object(), object()], "describe", GenerationConfig(seed=12))
    assert len(fake_runtime.loads) == 1 and len(fake_runtime.generated) == count
    assert fake_runtime.decoded == [[7, 99]] * count  # prompt tokens are never decoded
    assert response["metadata"]["multi_image_strategy"] == strategy
    assert response["metadata"]["image_count"] == 3
    assert all("temperature" not in call for call in fake_runtime.generated)
    assert len(set(fake_runtime.seeds)) == count
    backend.unload()
    assert backend._model is None
    backend.describe([object()], "again", GenerationConfig(.4, .8, 20, True, 99))
    assert len(fake_runtime.loads) == 2
    assert fake_runtime.generated[-1]["temperature"] == .4


def test_hf_text_schema_and_repair(fake_runtime, monkeypatch, tmp_path):
    backend = HFTextBackend("test/text", loader=fake_runtime.loader)
    backend._load()
    answers = iter(["bad json", '{"x": 1}'])
    monkeypatch.setattr(backend._processor, "decode", lambda *args, **kwargs: next(answers))
    schema = {"type": "object", "required": ["x"], "properties": {"x": {"type": "integer"}}}
    result = cached_structured(backend, Cache(tmp_path), "text.json", prompt="observations",
                               schema=schema, generation=GenerationConfig())
    assert result == {"x": 1}
    prompts = backend._processor.prompts
    assert "Return only JSON" in prompts[-1][0]["content"]
    assert "Repair the JSON" in prompts[-1][0]["content"]


def test_embedding_dimension_and_cache_separation(tmp_path):
    calls = []
    class Encoder:
        max_seq_length = 256
        def encode(self, texts, **kwargs):
            calls.append(texts)
            return np.ones((len(texts), 7))
    loader = lambda *args, **kwargs: Encoder()
    a, b = HFEmbeddingBackend("a", loader=loader), HFEmbeddingBackend("b", loader=loader)
    cache = Cache(tmp_path)
    vectors, key_a = cached_embeddings(a, cache, ["text"])
    _, key_b = cached_embeddings(b, cache, ["text"])
    cached_embeddings(a, cache, ["text"])
    assert vectors.shape == (1, 7) and key_a != key_b and len(calls) == 2


def test_quantization_errors_and_cpu_dtype(fake_runtime, monkeypatch):
    kwargs, device = pretrained_kwargs(HFOptions(), fake_runtime.torch, object())
    assert device == "cpu" and kwargs["dtype"] == "float32" and kwargs["device_map"] == {"": "cpu"}
    monkeypatch.setattr("logicad_fig2.backends.hf_common.importlib.util.find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="bitsandbytes is missing"):
        pretrained_kwargs(HFOptions(load_in_4bit=True), fake_runtime.torch, object())


def test_config_and_cli():
    from scripts.run_logicad_fig2 import parser
    args = parser().parse_args(["--data-root", "data", "--vlm-backend", "hf", "--load-in-4bit",
                               "--formatter-model", "custom", "--no-vlm-do-sample", "--vlm-max-tokens", "88"])
    config = Config(**{key: getattr(args, key) for key in Config.__dataclass_fields__})
    assert config.format_model == "custom" and config.vlm_backend == "hf" and config.load_in_4bit
    assert generation_for(config, "vlm").max_tokens == 88
    assert not generation_for(config, "vlm").do_sample
    assert Config(format_model="old").formatter_model == "old"
    with pytest.raises(ValueError):
        Config(formatter_do_sample=True)
    with pytest.raises(ValueError):
        Config(vlm_max_tokens=0)


@pytest.mark.parametrize("change", [dict(load_in_4bit=True), dict(dtype="float16"), dict(revision="v2")])
def test_hf_option_fingerprints(change):
    a = HFVisionBackend("model")
    b = HFVisionBackend("model", HFOptions(**change))
    assert a.signature() != b.signature()


def test_full_local_factory_e2e_with_fake_weights(fake_runtime, monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = Config(vlm_backend="hf", vlm_model="test/vlm", formatter_backend="hf",
                    formatter_model="test/text", embedding_backend="hf", embedding_model="test/embed",
                    disable_roi=True)
    bundle = build_backends(config)
    bundle.vision._loader = bundle.formatter._loader = fake_runtime.loader
    bundle.embedding._loader = lambda *a, **kw: SimpleNamespace(
        encode=lambda texts, **kw: np.ones((len(texts), 11)))
    data = [discover_category(make_dataset(tmp_path / "data"), "breakfast_box")]
    cache = Cache(tmp_path / "cache")
    pipeline = FeaturePipeline(TextExtractor(bundle, ROIExtractor(config), cache, config),
                               FormatEmbedder(bundle, cache, config), cache, config)
    result = evaluate(data, pipeline, config, tmp_path / "out")
    assert all(row["complete"] for row in result["runs"])
    assert len(fake_runtime.loads) == 2
    assert bundle.logic is None


def test_llava_pretrained_dispatch(fake_runtime, monkeypatch):
    calls = []
    model, processor = fake_runtime.loader("stub", HFOptions())
    model.config = SimpleNamespace(model_type="llava_next", vision_config=SimpleNamespace(patch_size=14),
                                   vision_feature_select_strategy="default")
    def load_model(name, **kwargs):
        calls.append((name, kwargs))
        return model
    transformers = SimpleNamespace(AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **kw: processor),
                                    AutoModelForImageTextToText=SimpleNamespace(from_pretrained=load_model))
    monkeypatch.setattr("logicad_fig2.backends.hf_vlm.dependencies", lambda: (fake_runtime.torch, transformers))
    HFVisionBackend._load_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf", HFOptions(revision="fixed"))
    assert calls[0][0] == "llava-hf/llava-v1.6-vicuna-13b-hf"
    assert calls[0][1]["revision"] == "fixed" and calls[0][1]["trust_remote_code"] is False
    assert processor.patch_size == 14 and processor.num_additional_image_tokens == 1


def test_description_cache_generation_and_roi_separation(tmp_path):
    from PIL import Image
    path = tmp_path / "image.png"
    Image.new("RGB", (4, 4)).save(path)
    config, cache = Config(disable_roi=True), Cache(tmp_path / "cache")
    vision = FakeVisionBackend()
    bundle = BackendBundle(vision, FakeTextBackend(), FakeEmbeddingBackend())
    keys = []
    for c in (config, replace(config, vlm_temperature=.7), replace(config, roi_padding=2)):
        extractor = TextExtractor(bundle, ROIExtractor(c), cache, c)
        keys.append(extractor.describe(path, "breakfast_box")["descriptions_cache"])
        extractor.describe(path, "breakfast_box")
    assert len(set(keys)) == 3 and len(vision.calls) == 9


def test_logic_uses_its_own_role(tmp_path):
    from logicad_fig2.logic_reasoner import LogicReasoner
    class Logic(FakeTextBackend):
        def generate(self, prompt, schema, generation):
            self.calls.append(prompt)
            return {"text": '{"formulas": ["left(apple,1)"]}'}
    formatter, logic = FakeTextBackend(), Logic()
    bundle = BackendBundle(FakeVisionBackend(), formatter, FakeEmbeddingBackend(), logic)
    reasoner = LogicReasoner(bundle, Cache(tmp_path), Config(enable_reasoner=True))
    formulas, key = reasoner.formalize("one apple on the left", "breakfast_box")
    assert formulas and key.startswith("formal/")
    assert len(logic.calls) == 1 and not formatter.calls
