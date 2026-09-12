import json

import numpy as np
import pytest
from jsonschema import ValidationError

from logicad_fig2.cache import Cache, canonical_json
from logicad_fig2.config import Config
from logicad_fig2.embeddings import cached_embeddings, cosine_anomaly_score, normalize
from logicad_fig2.format_embedding import FormatEmbedder, canonical_features
from logicad_fig2.prompts import get_prompts
from logicad_fig2.structured import parse_json


@pytest.mark.parametrize("a,b,expected", [([2, 0], [3, 0], 0), ([1, 0], [0, 2], 1), ([1, 0], [-1, 0], 2)])
def test_cosine_score(a, b, expected):
    assert cosine_anomaly_score(a, b) == pytest.approx(expected)


def test_normalization():
    assert np.linalg.norm(normalize([3, 4])) == pytest.approx(1)
    with pytest.raises(ValueError):
        normalize([0, 0])
    with pytest.raises(ValueError):
        normalize([float("inf"), 1])
    with pytest.raises(ValueError):
        cosine_anomaly_score([1, 2], [1, 2, 3])


def test_json_parsing():
    assert parse_json('```json\n{"x":1}\n```') == {"x": 1}
    for text in ('{"x":1,"x":2}', '[1,2]', '{"x":NaN}', 'prefix {"x":1}'):
        with pytest.raises(ValueError):
            parse_json(text)
    with pytest.raises(ValidationError):
        parse_json('{"left":[],"right":[{"object":"apple","count":-1,"present":true}]}',
                   get_prompts("breakfast_box").schema)


def test_format_repair_and_cache(tmp_path):
    class Client:
        calls = 0

        def chat(self, prompt, **kwargs):
            self.calls += 1
            return {"text": "bad json" if self.calls == 1 else json.dumps({"left": [], "right": []})}

        def embed(self, texts, **kwargs):
            self.calls += 1
            assert texts == ['{"left":[],"right":[]}']
            return {"vectors": [[3, 4]]}

    client = Client()
    cache = Cache(tmp_path)
    formatter = FormatEmbedder(client, cache, Config())
    result = formatter.format_and_embed("empty compartments", "breakfast_box")
    assert result["embedding"] == pytest.approx([.6, .8])
    formatter.format_and_embed("empty compartments", "breakfast_box")
    assert client.calls == 3
    attempts = cache.get(result["formatted_cache"].removesuffix(".json") + ".attempts.json")
    assert attempts[0]["text"] == "bad json"


def test_canonical_order():
    assert canonical_json(canonical_features({"items": ["b", "a"]})) == '{"items":["a","b"]}'


def test_bad_embedding_can_be_retried_without_force(tmp_path):
    class Client:
        calls = 0

        def embed(self, texts, **kwargs):
            self.calls += 1
            return {"vectors": [[0, 0]] if self.calls == 1 else [[1, 2]]}

    client, cache = Client(), Cache(tmp_path)
    with pytest.raises(ValueError):
        cached_embeddings(client, cache, ["text"], "model")
    vectors, key = cached_embeddings(client, Cache(tmp_path), ["text"], "model")
    assert vectors.tolist() == [[1, 2]] and client.calls == 2
