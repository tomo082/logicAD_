from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score

from .cache import atomic_json, file_digest, fingerprint
from .dataset import select_references
from .embeddings import cosine_anomaly_score
from .prompts import get_prompts


def classification_metrics(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or labels.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Labels and finite scores must be matching one-dimensional arrays")
    if not set(labels.tolist()).issubset({0, 1}):
        raise ValueError("Labels must be 0 (good) or 1 (logical anomaly)")
    if len(labels) == 0:
        return {"auroc": None, "f1_max": None, "f1_threshold": None, "metric_warning": "no_predictions"}
    if not labels.any():
        return {"auroc": None, "f1_max": 0.0, "f1_threshold": float(scores.min()),
                "metric_warning": "only_one_label_present"}
    # Same precision/recall calculation as notebook/stable_check.py; correctly
    # omit the terminal PR point, which has no associated threshold.
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    denom = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denom, out=np.zeros_like(denom), where=denom != 0)
    index = int(np.argmax(f1))
    both = len(set(labels.tolist())) == 2
    return {"auroc": float(roc_auc_score(labels, scores)) if both else None,
            "f1_max": float(f1[index]), "f1_threshold": float(thresholds[index]),
            "metric_warning": None if both else "only_one_label_present"}


def _stats(values):
    values = [value for value in values if value is not None]
    return {"mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=0)) if values else None, "n": len(values)}


def aggregate_runs(runs, categories, num_runs):
    summary = {}
    for category in categories:
        group = [r for r in runs if r["category"] == category]
        complete = len(group) == num_runs and all(r["complete"] for r in group)
        summary[category] = {"complete": complete, "completed_runs": sum(r["complete"] for r in group),
                             **{metric: _stats([r[metric] for r in group]) if complete else _stats([])
                                for metric in ("auroc", "f1_max")}}
    macro = {}
    for metric in ("auroc", "f1_max"):
        means = []
        for run in range(num_runs):
            group = [r for r in runs if r["run"] == run]
            if (len(group) == len(categories) and all(r["complete"] and r[metric] is not None for r in group)):
                means.append(float(np.mean([r[metric] for r in group])))
        macro[metric] = _stats(means if len(means) == num_runs else [])
    return {"categories": summary, "macro_average": macro,
            "category_count": len(categories), "is_five_category_average": len(categories) == 5,
            "std_ddof": 0, "metric_scale": "0..1", "num_runs": num_runs}


def _csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def evaluate(datasets, pipeline, config, output_dir, *, num_runs=1, reference_index=None, reasoner=None,
             retry_errors=False, force=False):
    """Persist each job; successful jobs are reused and failed jobs are retried.

    With --retry-errors, previous successes remain part of the complete evaluation.
    This flag requires the same experiment manifest (settings/images/references).
    """
    plan = []
    for dataset in datasets:
        references = select_references(dataset.references, num_runs=num_runs, seed=config.seed,
                                       reference_index=reference_index)
        plan.append({"category": dataset.category, "root": str(dataset.root),
                     "references": [{"path": str(p), "sha256": file_digest(p)} for p in references],
                     "queries": [{"path": str(s.path), "label": s.label, "split": s.split,
                                  "sha256": file_digest(s.path)} for s in dataset.queries]})
    manifest = {"config": asdict(config), "plan": plan, "pipeline_version": 2, "backends": pipeline.signature(),
                "prompts": {d.category: asdict(get_prompts(d.category)) for d in datasets},
                "reasoner": reasoner.signature() if reasoner else None}
    destination = Path(output_dir) / "results" / fingerprint(manifest)
    if retry_errors and not (destination / "manifest.json").is_file():
        raise ValueError("--retry-errors requires a previous run with identical settings and image files")
    atomic_json(destination / "manifest.json", manifest)
    rows, runs = [], []
    for category_plan in plan:
        category = category_plan["category"]
        for run, ref in enumerate(category_plan["references"]):
            ref_feature = None
            current = []
            for sample in category_plan["queries"]:
                record_path = destination / "records" / category / str(run) / (fingerprint(sample) + ".json")
                row = None
                if record_path.is_file() and not force:
                    try:
                        saved = json.loads(record_path.read_text(encoding="utf-8"))
                        logic_done = not reasoner or saved.get("reasoner", {}).get("status") in ("normal", "abnormal")
                        if saved.get("status") == "ok" and (logic_done or retry_errors and not config.enable_reasoner):
                            row = saved
                    except (ValueError, AttributeError):
                        logging.warning("Invalid result record will be rebuilt: %s", record_path)
                if row is None:
                    row = {"category": category, "run": run, "reference": ref["path"],
                           "image": sample["path"], "label": sample["label"], "split": sample["split"]}
                    stage = "reference_features"
                    try:
                        if ref_feature is None:
                            ref_feature = pipeline.image(Path(ref["path"]), category, reference=True)
                        stage = "query_features"
                        query = pipeline.image(Path(sample["path"]), category)
                        row.update(status="ok", score=cosine_anomaly_score(ref_feature["embedding"], query["embedding"]),
                                   selected_index=query["selection"]["selected_index"],
                                   selected_text=query["selected_text"], roi=query["roi"],
                                   selection_cache=query["selection_cache"], formatted_cache=query["formatted_cache"],
                                   reference_cache=ref_feature["reference_cache"])
                        if reasoner:
                            try:
                                row["reasoner"] = reasoner.reason(ref_feature["selected_text"], query["selected_text"], category)
                            except Exception as exc:
                                row["reasoner"] = {"status": "error", "error_type": type(exc).__name__, "error": str(exc)}
                                logging.warning("Reasoner error for %s: %s", sample["path"], exc)
                    except Exception as exc:
                        row.update(status="error", stage=stage, error_type=type(exc).__name__, error=str(exc), score=None)
                        logging.error("%s failed on %s: %s", stage, sample["path"], exc)
                    atomic_json(record_path, row)
                current.append(row)
                rows.append(row)
            success = [row for row in current if row["status"] == "ok"]
            metrics = classification_metrics([r["label"] for r in success], [r["score"] for r in success])
            run_result = {"category": category, "run": run, "reference": ref["path"], "expected": len(current),
                          "succeeded": len(success), "failed": len(current) - len(success),
                          "complete": len(current) == len(success), **metrics}
            if reasoner:
                resolved = [r for r in success if r.get("reasoner", {}).get("status") in ("normal", "abnormal")]
                run_result["reasoner_resolved"] = len(resolved)
                run_result["reasoner_total"] = len(current)
            runs.append(run_result)
            logging.info("%s run %d: %s", category, run + 1, metrics)
    summary = aggregate_runs(runs, [d.category for d in datasets], num_runs)
    atomic_json(destination / "predictions.json", rows)
    atomic_json(destination / "errors.json", [r for r in rows if r["status"] != "ok" or
                                              r.get("reasoner", {}).get("status") in ("error", "unknown")])
    atomic_json(destination / "runs.json", runs)
    atomic_json(destination / "summary.json", summary)
    _csv(destination / "predictions.csv", rows)
    _csv(destination / "runs.csv", runs)
    summary_rows = [{"category": category, "complete": value["complete"],
                     **{f"{m}_{k}": value[m][k] for m in ("auroc", "f1_max") for k in ("mean", "std", "n")}}
                    for category, value in summary["categories"].items()]
    summary_rows.append({"category": "macro_average", **{f"{m}_{k}": summary["macro_average"][m][k]
                                                         for m in ("auroc", "f1_max") for k in ("mean", "std", "n")}})
    _csv(destination / "summary.csv", summary_rows)
    return {"output": str(destination), "summary": summary, "runs": runs}
