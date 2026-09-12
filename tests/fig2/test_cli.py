import json
import subprocess
import sys
from pathlib import Path

from test_dataset_metrics import make_dataset

REPO = Path(__file__).resolve().parents[2]


def test_cli_dry_run_from_other_directory(tmp_path):
    make_dataset(tmp_path / "data")
    result = subprocess.run([sys.executable, str(REPO / "scripts/run_logicad_fig2.py"),
                             "--data-root", str(tmp_path / "data"), "--category", "breakfast_box",
                             "--num-runs", "5", "--max-images", "2", "--dry-run"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)["plan"][0]
    assert plan["queries"] == 2 and len(set(plan["references"])) == 5


def test_five_category_offline_smoke(tmp_path):
    from scripts.smoke_logicad_fig2 import smoke

    result = smoke(tmp_path / "smoke")
    assert len(result["runs"]) == 25
    assert result["summary"]["is_five_category_average"]
    assert result["resume_additional_calls"] == 0
