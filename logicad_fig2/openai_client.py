from __future__ import annotations

import base64
import io
import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from PIL import Image

from .config import Config


class Client(Protocol):
    def chat(self, prompt: str, *, model: str, images=(), schema=None, temperature=0.0, top_p=1.0) -> dict: ...

    def embed(self, texts: list[str], *, model: str) -> dict: ...


def image_url(image: Image.Image) -> dict:
    """Adapted from base.py:image2base64; correct MIME and no eager torch import."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}", "detail": "high"}}


class OpenAIClient:
    """Lazy SDK transport; inject sdk and sleep for unit tests, no API at import time."""

    def __init__(self, config: Config, usage_path: Path | None = None, sdk=None, sleep=time.sleep):
        self.config = config
        self.usage_path = usage_path
        self._sdk = sdk
        self._sleep = sleep
        self.calls = 0
        self.attempts = 0
        self.tokens = 0

    @property
    def sdk(self):
        if self._sdk is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("Set OPENAI_API_KEY to run uncached OpenAI stages")
            from openai import OpenAI

            self._sdk = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                               timeout=self.config.timeout, max_retries=0)
        return self._sdk

    def _log(self, record):
        record = {"time": datetime.now(timezone.utc).isoformat(), **record}
        if self.usage_path:
            self.usage_path.parent.mkdir(parents=True, exist_ok=True)
            with self.usage_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")

    def _request(self, call, kind, model):
        for attempt in range(self.config.max_retries + 1):
            self.attempts += 1
            try:
                response = call()
                raw = response.model_dump(mode="json")
                usage = raw.get("usage") or {}
                self.calls += 1
                self.tokens += usage.get("total_tokens", 0)
                self._log({"kind": kind, "model": model, "attempt": attempt + 1, "usage": usage,
                           "response_id": raw.get("id"), "status": "ok"})
                logging.info("API %s: calls=%s attempts=%s tokens=%s", kind, self.calls, self.attempts, self.tokens)
                return raw
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                transient = status in (408, 409, 429) or (status is not None and status >= 500)
                transient |= type(exc).__name__ in ("APITimeoutError", "APIConnectionError", "TimeoutError")
                self._log({"kind": kind, "model": model, "attempt": attempt + 1,
                           "status": "error", "error_type": type(exc).__name__, "http_status": status})
                if not transient or attempt == self.config.max_retries:
                    raise
                delay = min(60.0, 2 ** attempt + random.random())
                headers = getattr(getattr(exc, "response", None), "headers", {})
                try:
                    delay = max(delay, float(headers.get("retry-after", 0)))
                except (ValueError, TypeError):
                    pass
                logging.warning("Transient API error %s; retry in %.1fs", type(exc).__name__, delay)
                self._sleep(delay)

    def chat(self, prompt, *, model, images=(), schema=None, temperature=0.0, top_p=1.0, max_tokens=None, seed=None):
        content = [{"type": "text", "text": prompt}] + [image_url(img) for img in images]
        kwargs = dict(model=model, messages=[{"role": "user", "content": content}],
                      temperature=temperature, top_p=top_p, max_tokens=max_tokens or self.config.max_tokens)
        if seed is not None:
            kwargs["seed"] = seed
        mode = "text"
        if schema:
            mode = "json_schema"
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "logicad_features", "strict": True, "schema": schema}}
        try:
            raw = self._request(lambda: self.sdk.chat.completions.create(**kwargs), "chat", model)
        except Exception as exc:
            # Older snapshots / compatible model names may only support JSON mode.
            # Do not hide unrelated 400 errors (invalid schema/model/image etc.).
            message = str(exc).lower()
            unsupported = "not supported" in message or "unsupported" in message
            if schema and getattr(exc, "status_code", None) == 400 and unsupported and (
                "response_format" in message or "json_schema" in message
            ):
                logging.warning("Model %s lacks structured output; using validated JSON mode", model)
                mode = "json_object"
                kwargs["response_format"] = {"type": "json_object"}
                kwargs["messages"][0]["content"][0]["text"] += "\nJSON schema: " + json.dumps(schema)
                raw = self._request(lambda: self.sdk.chat.completions.create(**kwargs), "chat", model)
            else:
                raise
        choice = raw["choices"][0]
        return {"text": choice["message"].get("content") or "", "raw": raw, "mode": mode,
                "finish_reason": choice.get("finish_reason"), "refusal": choice["message"].get("refusal")}

    def embed(self, texts, *, model):
        raw = self._request(lambda: self.sdk.embeddings.create(input=texts, model=model), "embedding", model)
        ordered = sorted(raw["data"], key=lambda row: row["index"])
        if [row["index"] for row in ordered] != list(range(len(texts))):
            raise ValueError("Embedding response has missing or duplicate indices")
        return {"vectors": [row["embedding"] for row in ordered], "raw": raw}
