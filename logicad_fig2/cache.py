from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:24]


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".writing-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class Cache:
    """Atomic, versioned, single-writer cache. Force invalidates once per process.

    Keys include content/model/prompt/settings at each stage. In-memory memoization
    also prevents --force from charging again for the reference on every query.
    """

    def __init__(self, root: Path, force: bool = False):
        self.root = Path(root)
        self.force = force
        self._memo = {}

    def path(self, key: str) -> Path:
        result = (self.root / key).resolve()
        if self.root.resolve() not in result.parents:
            raise ValueError("Cache key must be inside the cache directory")
        return result

    def get(self, key: str):
        if key in self._memo:
            return self._memo[key]
        if self.force:
            return None
        path = self.path(key)
        if not path.is_file():
            return None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            if envelope.get("version") != 1:
                return None
            self._memo[key] = envelope["value"]
            return envelope["value"]
        except (ValueError, KeyError, AttributeError):
            logging.warning("Corrupt cache record ignored: %s", path)
            return None

    def put(self, key: str, value):
        atomic_json(self.path(key), {"version": 1, "value": value})
        self._memo[key] = value
        return value
