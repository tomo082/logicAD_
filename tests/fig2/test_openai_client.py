from types import SimpleNamespace

import pytest
from PIL import Image

from logicad_fig2.config import Config
from logicad_fig2.openai_client import OpenAIClient


class Response:
    def __init__(self, value):
        self.value = value

    def model_dump(self, **kwargs):
        return self.value


def sdk(chat=None, embed=None):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=chat)),
                           embeddings=SimpleNamespace(create=embed))


def chat_response(text="ok"):
    return Response({"id": "fake", "usage": {"total_tokens": 10}, "choices": [
        {"message": {"content": text}, "finish_reason": "stop"}]})


def test_multimodal_message_keeps_original_and_patches(tmp_path):
    captured = []

    def create(**kwargs):
        captured.append(kwargs)
        return chat_response()

    client = OpenAIClient(Config(), tmp_path / "usage.jsonl", sdk=sdk(chat=create))
    images = [Image.new("RGB", (10, 10)), Image.new("RGB", (5, 5))]
    result = client.chat("inspect", model="gpt-4o", images=images, temperature=.05, top_p=.1)
    content = captured[0]["messages"][0]["content"]
    assert len(content) == 3
    assert all(image["image_url"]["url"].startswith("data:image/png;base64,") for image in content[1:])
    assert captured[0]["model"] == "gpt-4o" and captured[0]["temperature"] == .05
    assert result["raw"]["id"] == "fake" and client.tokens == 10
    assert "base64" not in (tmp_path / "usage.jsonl").read_text()


def test_rate_limit_exponential_backoff():
    class RateLimit(Exception):
        status_code = 429
        response = SimpleNamespace(headers={"retry-after": "3"})

    calls, delays = [], []

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise RateLimit()
        return chat_response()

    client = OpenAIClient(Config(), sdk=sdk(chat=create), sleep=delays.append)
    client.chat("test", model="gpt-4o")
    assert client.attempts == 3 and len(delays) == 2 and min(delays) >= 3


def test_json_mode_fallback_for_unsupported_snapshot():
    class Unsupported(Exception):
        status_code = 400

    calls = []

    def create(**kwargs):
        calls.append(kwargs["response_format"]["type"])
        if len(calls) == 1:
            raise Unsupported("response_format json_schema is not supported with this model")
        return chat_response('{"value":1}')

    client = OpenAIClient(Config(), sdk=sdk(chat=create))
    result = client.chat("JSON please", model="gpt-4o-2024-05-13", schema={"type": "object"})
    assert calls == ["json_schema", "json_object"] and result["mode"] == "json_object"


def test_auth_errors_do_not_retry():
    class AuthError(Exception):
        status_code = 401

    def create(**kwargs):
        raise AuthError()

    client = OpenAIClient(Config(), sdk=sdk(chat=create))
    with pytest.raises(AuthError):
        client.chat("test", model="gpt-4o")
    assert client.attempts == 1


def test_embedding_indices_sorted_and_validated():
    def create(**kwargs):
        return Response({"data": [{"index": 1, "embedding": [0, 1]}, {"index": 0, "embedding": [1, 0]}]})

    client = OpenAIClient(Config(), sdk=sdk(embed=create))
    assert client.embed(["a", "b"], model="text-embedding-3-large")["vectors"] == [[1, 0], [0, 1]]
