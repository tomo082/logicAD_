import json
import subprocess
from types import SimpleNamespace

import pytest

from logicad_fig2.cache import Cache
from logicad_fig2.config import Config
from logicad_fig2.logic_reasoner import (LogicReasoner, atp_program, build_axioms, complete_query,
                                         explain_subset, minimal_inconsistent_subset)
from logicad_fig2.logic_syntax import parse_formulas
from logicad_fig2.prompts import ALIASES, PREDICATE_FEATURES

FEATURES = PREDICATE_FEATURES["breakfast_box"]


def formulas(lines):
    return parse_formulas(lines, FEATURES, ALIASES["breakfast_box"])


def test_missing_default_and_synonyms():
    normal = formulas(["left(orange,2)", "right(cereal,irrel)"])
    query = formulas(["left(tangerine,2)"])
    completed, defaults = complete_query(normal, query)
    assert [f.render() for f in defaults] == ["right(cereal,0)"]
    assert "missing granola/cereal" in explain_subset(defaults)
    axioms = build_axioms(normal, completed, FEATURES)
    assert all(axioms[key] for key in ("sigma_norm", "sigma_na", "sigma_fa", "sigma_dca"))
    assert not any("tangerine" in f for f in axioms["sigma_na"])
    assert any("p_right(x,c_0)" in f for f in axioms["sigma_dca"])
    program = atp_program(axioms, completed, 5)
    assert "formulas(goals)." in program and "c_2" in program
    assert "all x" in program


@pytest.mark.parametrize("text", ["left(apple,1). end_of_list.", "left(f(apple),1)", "evil(apple)",
                                  "left(apple)", "all x left(x,1)", "left(apple,1) trailing"])
def test_reject_bad_formulas(text):
    with pytest.raises(ValueError):
        formulas([text])


def test_legacy_normal_specifications_parse():
    from formal_prompts_spec import logical_spec_dict

    for category, spec in logical_spec_dict.items():
        lines = [line for line in spec["norm_spec"].splitlines() if line.strip()]
        assert parse_formulas(lines, PREDICATE_FEATURES[category], ALIASES.get(category), True)


def test_irreducible_subset_not_minimum_cardinality():
    query = formulas(["left(apple,1)", "left(nectarine,1)", "right(cereal,irrel)"])

    def inconsistent(subset):
        return {f.render() for f in query[:2]}.issubset({f.render() for f in subset})

    result, details = minimal_inconsistent_subset(query, inconsistent)
    assert result == query[:2] and details["minimality_certified"]
    _, details = minimal_inconsistent_subset(query, lambda _: None)
    assert not details["minimality_certified"]
    _, details = minimal_inconsistent_subset(query, inconsistent, max_checks=1)
    assert not details["minimality_certified"]


def reasoner(tmp_path, runner):
    prover = tmp_path / "prover9"
    mace = tmp_path / "mace4"
    prover.touch()
    mace.touch()
    config = Config(enable_reasoner=True, prover9_path=str(prover), mace4_path=str(mace))

    class API:
        def chat(self, prompt, **kwargs):
            value = ["right(cereal,irrel)"] if "sole normal reference" in prompt else ["right(cereal,0)"]
            return {"text": json.dumps({"formulas": value})}

    return LogicReasoner(API(), Cache(tmp_path / "cache"), config, runner)


def test_reasoner_complete_proof_and_explanation_with_mock_processes(tmp_path):
    def runner(command, **kwargs):
        assert kwargs["shell"] is False
        if command[0].endswith("mace4"):
            return SimpleNamespace(returncode=0, stdout="interpretation(4, [], []).", stderr="")
        if "goals).\n$F." in kwargs["input"]:
            return SimpleNamespace(returncode=2, stdout="SEARCH FAILED", stderr="")
        return SimpleNamespace(returncode=0, stdout="================ PROOF ================", stderr="")

    result = reasoner(tmp_path, runner).reason("granola present", "granola missing", "breakfast_box")
    assert result["status"] == "abnormal"
    assert "missing granola/cereal" in result["explanation"]
    assert result["subset_details"]["minimality_certified"]
    assert list((tmp_path / "cache/prover_calls").glob("*.in"))


def test_timeout_is_unknown_not_normal(tmp_path):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 10)

    result = reasoner(tmp_path, timeout).reason("present", "missing", "breakfast_box")
    assert result["status"] == "unknown"


def test_inconsistent_normal_is_not_an_anomaly(tmp_path):
    def runner(command, **kwargs):
        if command[0].endswith("mace4"):
            return SimpleNamespace(returncode=2, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="================ PROOF ================", stderr="")

    result = reasoner(tmp_path, runner).reason("contradiction", "query", "breakfast_box")
    assert result["status"] == "unknown" and result["reason"] == "inconsistent_normal_specification"


def test_missing_prover_skips_without_api(tmp_path, monkeypatch):
    monkeypatch.setattr("logicad_fig2.logic_reasoner.shutil.which", lambda _: None)
    result = LogicReasoner(None, Cache(tmp_path), Config()).reason("normal", "query", "breakfast_box")
    assert result == {"status": "skipped", "reason": "prover9_not_found"}
