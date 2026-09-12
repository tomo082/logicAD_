from __future__ import annotations

import json
import re

from jsonschema import ValidationError, validate

from .cache import canonical_json


def parse_json(text: str, schema=None):
    """Accept strict JSON or a single fenced object; reject prose and duplicate keys."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    text = text.strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if not match:
            raise ValueError("Invalid JSON code fence")
        text = match.group(1)
    value = json.loads(text, object_pairs_hook=pairs)
    canonical_json(value)  # reject NaN/Infinity
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    if schema:
        validate(value, schema)
    return value


def cached_structured(backend, cache, key, *, prompt, schema, generation, retries=2, validator=None):
    """Validate -> repair, preserving all raw answers, without fabricating fallback features."""
    value = cache.get(key)
    if value is not None:
        return value
    attempts_key = key.removesuffix(".json") + ".attempts.json"
    attempts = cache.get(attempts_key) or []
    error = ""

    def parse(response):
        if response.get("refusal") or response.get("finish_reason") == "length":
            raise ValueError("Refused or truncated structured response")
        parsed = parse_json(response["text"], schema)
        if validator:
            validator(parsed)
        return parsed

    # Recover an answer saved just before a crash, including raw parse failures.
    if attempts:
        try:
            return cache.put(key, parse(attempts[-1]))
        except (ValueError, KeyError, ValidationError) as exc:
            error = str(exc)[:700]
    for _ in range(retries + 1):
        repair = ""
        if error:
            repair = ("\nPrevious response failed validation: " + error +
                      "\nPrevious response: " + attempts[-1].get("text", "") +
                      "\nRepair the JSON using only the supplied observations. Return the complete JSON object.")
        response = backend.generate(prompt + repair, schema, generation)
        attempts.append(response)
        cache.put(attempts_key, attempts)
        try:
            return cache.put(key, parse(response))
        except (ValueError, KeyError, ValidationError) as exc:
            error = str(exc)[:700]
    raise ValueError(f"Structured output failed after {retries + 1} attempts: {error}")
