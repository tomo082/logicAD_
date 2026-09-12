from pathlib import Path

import pytest
from PIL import Image

from logicad_fig2.dataset import discover_category, select_references
from logicad_fig2.evaluator import aggregate_runs, classification_metrics


def make_dataset(root, category="breakfast_box", references=5):
    for split, count in (("train/good", references), ("test/good", 2), ("test/logical_anomalies", 2),
                         ("test/structural_anomalies", 1)):
        directory = root / category / split
        directory.mkdir(parents=True)
        for index in range(count):
            Image.new("RGB", (16, 16), color=(index * 30, 20, 40)).save(directory / f"{index:03d}.png")
    return root


def test_discovery_and_limit(tmp_path):
    make_dataset(tmp_path)
    data = discover_category(tmp_path, "breakfast_box", 3)
    assert [s.label for s in data.queries] == [0, 1, 0]
    assert len(data.references) == 5
    assert all(s.split != "structural_anomalies" for s in data.queries)
    assert discover_category(tmp_path / "breakfast_box", "breakfast_box").root == data.root


def test_missing_split(tmp_path):
    with pytest.raises(ValueError):
        discover_category(tmp_path, "breakfast_box")


def test_distinct_seeded_references():
    refs = [Path(str(i)) for i in range(10)]
    result = select_references(refs, num_runs=5, seed=123, reference_index=0)
    assert len(set(result)) == 5 and result[0] == refs[0]
    assert result == select_references(refs, num_runs=5, seed=123, reference_index=0)
    with pytest.raises(ValueError):
        select_references(refs, num_runs=11)


def test_f1_and_auroc():
    perfect = classification_metrics([0, 0, 1, 1], [.1, .2, .8, .9])
    assert perfect["auroc"] == 1 and perfect["f1_max"] == 1
    tied = classification_metrics([0, 0, 1, 1], [.5] * 4)
    assert tied["auroc"] == .5 and tied["f1_max"] == pytest.approx(2 / 3)
    assert classification_metrics([0, 0], [0, 1])["auroc"] is None
    assert classification_metrics([], [])["f1_max"] is None


def test_macro_std_and_partial_runs():
    runs = [{"category": cat, "run": run, "complete": True, "auroc": value, "f1_max": value}
            for cat in ("a", "b") for run, value in enumerate((.5, 1))]
    summary = aggregate_runs(runs, ["a", "b"], 2)
    assert summary["macro_average"]["auroc"] == {"mean": .75, "std": .25, "n": 2}
    runs[0]["complete"] = False
    summary = aggregate_runs(runs, ["a", "b"], 2)
    assert summary["macro_average"]["auroc"]["mean"] is None
