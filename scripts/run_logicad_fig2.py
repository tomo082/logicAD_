#!/usr/bin/env python3
"""Standalone Linux/DLBox entry point; run from any working directory."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from logicad_fig2.cache import Cache
from logicad_fig2.config import Config
from logicad_fig2.dataset import discover_category
from logicad_fig2.evaluator import evaluate
from logicad_fig2.format_embedding import FeaturePipeline, FormatEmbedder
from logicad_fig2.openai_client import OpenAIClient
from logicad_fig2.prompts import CATEGORIES
from logicad_fig2.roi import ROIExtractor
from logicad_fig2.text_extraction import TextExtractor


def parser():
    p = argparse.ArgumentParser(description="LogicAD Figure 2/3 one-shot logical anomaly classification")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--category", choices=(*CATEGORIES, "all"), default="all")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--num-runs", type=int, default=1)
    p.add_argument("--reference-index", type=int)
    p.add_argument("--max-images", type=int, help="Maximum test images per category (both labels interleaved)")
    rerun = p.add_mutually_exclusive_group()
    rerun.add_argument("--force", action="store_true", help="Regenerate once per stage within this invocation")
    rerun.add_argument("--retry-errors", action="store_true", help="Resume a previous identical experiment")
    p.add_argument("--dry-run", action="store_true", help="Validate dataset and reference selection without API calls")
    p.add_argument("--verbose", action="store_true")
    defaults = Config()
    strings = ("vlm_model", "embedding_model", "format_model", "logic_model", "gdino_checkpoint", "feature_prompt",
               "prover9_path", "mace4_path")
    integers = ("k", "seed", "max_tokens", "max_retries", "json_retries", "lof_neighbors", "max_rois",
                "prover_timeout", "max_mis_checks")
    floats = ("temperature", "top_p", "timeout", "box_threshold", "text_threshold", "roi_padding")
    for name, kind in [(n, str) for n in strings] + [(n, int) for n in integers] + [(n, float) for n in floats]:
        p.add_argument("--" + name.replace("_", "-"), type=kind, default=getattr(defaults, name))
    p.add_argument("--gdino-config", choices=("swint", "swinb"), default="swint")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N; GPU is used only for ROI")
    p.add_argument("--disable-roi", action="store_true")
    p.add_argument("--enable-reasoner", action="store_true")
    p.add_argument("--normal-rules", choices=("reference", "legacy"), default="reference")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    # Keep SDK/http transport from dumping request headers and image payloads.
    for name in ("openai", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        config = Config(**{key: getattr(args, key) for key in Config.__dataclass_fields__})
        categories = CATEGORIES if args.category == "all" else (args.category,)
        datasets = [discover_category(args.data_root, cat, args.max_images) for cat in categories]
        from logicad_fig2.dataset import select_references

        preview = [{"category": d.category, "queries": len(d.queries), "labels": [s.label for s in d.queries],
                    "references": [str(p) for p in select_references(d.references, num_runs=args.num_runs,
                                                                     seed=config.seed, reference_index=args.reference_index)]}
                   for d in datasets]
        if args.dry_run:
            print(json.dumps({"dry_run": True, "plan": preview}, indent=2))
            return 0
        if not config.disable_roi and not config.gdino_checkpoint:
            logging.warning("GroundingDINO checkpoint unspecified: running with original images only")
        cache = Cache(args.output_dir / "cache", force=args.force)
        client = OpenAIClient(config, args.output_dir / "api_usage.jsonl")
        extractor = TextExtractor(client, ROIExtractor(config), cache, config)
        pipeline = FeaturePipeline(extractor, FormatEmbedder(client, cache, config), cache, config)
        reasoner = None
        if config.enable_reasoner:
            from logicad_fig2.logic_reasoner import LogicReasoner

            reasoner = LogicReasoner(client, cache, config)
        result = evaluate(datasets, pipeline, config, args.output_dir, num_runs=args.num_runs,
                          reference_index=args.reference_index, reasoner=reasoner,
                          retry_errors=args.retry_errors, force=args.force)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if all(row["complete"] for row in result["runs"]) else 2
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
