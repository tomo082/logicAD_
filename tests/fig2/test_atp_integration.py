"""Optional real ATP checks; no OpenAI requests. Set paths or install on PATH."""
import os
import shutil

import pytest

from logicad_fig2.cache import Cache
from logicad_fig2.config import Config
from logicad_fig2.logic_reasoner import LogicReasoner, build_axioms
from logicad_fig2.logic_syntax import parse_formulas

PROVER = os.environ.get("PROVER9_TEST_PATH") or shutil.which("prover9")
MACE = os.environ.get("MACE4_TEST_PATH") or shutil.which("mace4")
pytestmark = pytest.mark.skipif(not (PROVER and MACE), reason="Real Prover9 and Mace4 binaries not configured")


@pytest.mark.parametrize("count,expected", [("0", True), ("irrel", False)])
def test_real_atp_missing_item_and_consistent_query(tmp_path, count, expected):
    features = {"right": "bfc"}
    normal = parse_formulas(["right(granola,irrel)"], features)
    query = parse_formulas([f"right(granola,{count})"], features)
    config = Config(prover9_path=PROVER, mace4_path=MACE, prover_timeout=10)
    reasoner = LogicReasoner(None, Cache(tmp_path), config)
    axioms = build_axioms(normal, query, features)
    baseline, evidence = reasoner.check(axioms, [], model_first=True)
    assert baseline is False, evidence
    verdict, evidence = reasoner.check(axioms, query)
    assert verdict is expected, evidence
