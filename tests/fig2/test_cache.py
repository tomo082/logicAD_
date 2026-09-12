from logicad_fig2.cache import Cache, fingerprint


def test_atomic_cache_and_force_once(tmp_path):
    Cache(tmp_path).put("a.json", {"value": 3})
    assert Cache(tmp_path).get("a.json") == {"value": 3}
    forced = Cache(tmp_path, force=True)
    assert forced.get("a.json") is None
    forced.put("a.json", {"value": 4})
    assert forced.get("a.json") == {"value": 4}
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})


def test_corrupt_cache_is_recoverable(tmp_path):
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    assert Cache(tmp_path).get("bad.json") is None
