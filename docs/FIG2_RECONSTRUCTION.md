# Figure 2/3 reconstruction audit (Phase 1, before implementation)

Baseline: `tomo082/logicAD_`, commit `25833812e98e50ea8fc61b0fdab00971479aca46`.
Inspected the complete file inventory (242 non-git files), model registration,
LLM utilities, notebooks, formal scripts, dependency/test configuration and every
`datasets/*/*.json` intermediate-output file. No repository AGENTS.md was present.

| Paper operation | Existing repository evidence | Reuse / required addition |
| --- | --- | --- |
| Figure 3 ROI extraction | `src/anomalib/utils/llm/grounddino.py:load_gdino_model`; `notebook/pushping_sliding_window_approach.md` | Reuse loader through isolated module loading; add portable inference, clamped crops, original-image preservation and missing-ROI fallback. Referenced `models/logicad/sliding_window.py` is absent. |
| Guided category observations | `notebook/image2text_demo.Rmd`, pushpin notebook, all five `datasets/*/img2text*.json` | Reconstruct centralized prompts using supplementary A.1/A.4 and observed outputs; missing `TEXT_EXTRACTOR_PROMPTS` cannot be recovered verbatim from this checkout. |
| GPT-4o image-to-text | `src/anomalib/utils/llm/base.py:img2text`, `vlm.py` | Adapt base64 multimodal messages in an independent SDK adapter. Direct import eagerly imports torch/transformers; key-file/Azure assumptions and swallowed errors preclude safe direct reuse. |
| K=3, embeddings, LOF selection | `base.py:txt2embedding`; no LOF implementation found | Add independent repetitions, raw-response/embedding cache, small-sample LOF and seeded random inlier selection. |
| JSON format embedding | `base.py:txt2sum`; `notebook/stable_check.py:generate_text_embedding` | Preserve LLM JSON -> text embedding order. Add category schemas, validation/repair and canonical serialization. Original `TEXT_SUMMATION_PROMPTS` is absent. |
| One-shot cosine score | `base.py:cos_sim`; `stable_check.py:evaluation` | Reuse mathematical operation with validated L2 normalization and cached reference features. |
| LOCO logical evaluation | `stable_check.py:get_dataset_per_category`, `calculate_f1_max`, `src/anomalib/data/mvtec_loco.py` | Preserve good/logical split and precision-recall F1-max; add portable discovery, seeded distinct references, repeated runs, incomplete-run reporting and JSON/CSV. |
| Formalization | `formal_trans_reason.py`, `formal_prompts_spec.py` | Reuse side-effect-free prompt constants and two-shot examples for breakfast, juice and connectors. New schemas/validation; reconstruct pushpins and screw bag vocabulary. |
| Normal rules and extra axioms | `formal_prompts_spec.py:logical_spec_dict`; `formal_reasoner_all.py:logical_spec_generator` | Reuse existing normal rules optionally; default derives rules from one reference via LLM. Reimplement naming, functionality, domain closure and missing-value completion without global mutable state or author paths. |
| ATP and explanations | `formal_reasoner_all.py` | Replace import-time execution, `shell=True`, author paths and substring success handling with bounded Prover9/Mace4 subprocesses, explicit unknown status, query-relative inconsistent subset extraction and saved evidence. |

## Intermediate outputs recovered

- Breakfast: 240 natural descriptions; 187 entries each in two formal versions.
  Left fruit counts, right cereal/banana/nut presence, disjunctions and `irrel`.
- Juice: 57 current descriptions; variants 57/58; formal variants 57 each.
  Fill level, sticker count/location, fruit/color compatibility and symmetry.
  Some early formal outputs contain unrelated cats/grass or incompatible arities:
  these files are evidence of format evolution, **not ground truth or reusable API cache**.
- Pushpins: 23 descriptions, sometimes only presence; compartment counts are
  recoverable from the notebook and supplementary prompt, not the JSON alone.
- Screw bag: 21 descriptions per version; bolt/washer/nut counts and relative lengths.
- Connectors: 23 current descriptions, alternative 25/23/32 outputs; whole-image
  counts plus patch-specific top/middle/bottom cable positions.

## Sources and decisions

- [Paper, Figure 2/3 and method](https://arxiv.org/html/2501.01767v2)
- [Author project](https://jasonjin34.github.io/logicad.github.io/)
- [Supplement A.1, A.3, A.4](https://jasonjin34.github.io/logicad.github.io/static/pdfs/LogicAD_Supplymentary_Matrials.pdf)
- [OpenAI vision inputs](https://developers.openai.com/api/docs/guides/images-vision)
- [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs)

Supplement A.4 supplies temperature 0.05, top_p 0.1 and ROI thresholds
(box/text 0.2/0.3, pushpins 0.1/0.1). Its connector detector keyword refers
to a fruit juice bottle; use a connector keyword as an explicitly documented
reconstruction assumption and allow overrides. The Azure 2024-05-13 deployment
is replaced with requested public OpenAI `gpt-4o`, configurable by CLI.

LOF neighbor count, canonical JSON schemas, crop padding, stable seed policy,
and unseen-category formal predicates are reconstruction assumptions. The paper
selects a random surviving description; use seeded random selection, not a new
anomaly detector. Preserve the exact cosine equation (mathematically [0,2]).

## Phase gates

1. This audit precedes new pipeline code.
2. Image -> multimodal SDK adapter with mock single-image execution. A live
   GPT-4o check requires OPENAI_API_KEY, which is absent in this environment.
3. K/cache/embedding/LOF mock execution and resume checks.
4. JSON -> normalized embedding -> cosine checks.
5. Synthetic LOCO directory end-to-end evaluation, repeated references and exports.
6. ATP input, outcome and explanation tests, real binaries if available.
7. Dedicated isolated test suite, CLI smoke checks and README limitations.

## Completed phase verification

| Phase | Added / checked | Result and remaining external dependency |
| --- | --- | --- |
| 1 | Repository inventory, intermediate JSON inspection, paper/supplement mapping | Completed before implementation; missing model/prompt code identified. |
| 2 | Multimodal SDK boundary, preserved original, ROI fallback | 2 offline tests passed; no live API key available. |
| 3 | K=3, raw-response resume, embeddings, bounded LOF | 8 tests passed; interrupted description generation resumes from saved successes. |
| 4 | Structured JSON repair, canonical embedding, normalized cosine | 17 tests passed, including invalid counts, zero vectors and negative cosine. |
| 5 | LOCO loading, distinct references, repeated evaluation, exports/retry | 24 tests passed, including five-run end-to-end and failed-image recovery. |
| 6 | Formal AST, legacy rules, Γ, ATP status/evidence, inconsistent subset | 42 tests passed after transport tests; real prover binaries unavailable. |
| 7 | Five-category integration, CLI, documentation, optional ATP tests, CI | 45 passed / 2 real-ATP tests skipped; dependency check and compile checks passed. |

Validation host: Windows, isolated Python 3.12.14 environment. Python 3.10 grammar
validated on all 17 implementation/entry-point modules; actual Linux 3.10 CI was
configured but not run here. Offline CLI smoke processed 5 categories × 5 runs ×
4 queries = 100 image/reference pairs, with 170 synthetic API calls and zero
additional calls on resume. Synthetic results are labeled `SMOKE_TEST_ONLY.json`.
No live GPT-4o, GroundingDINO checkpoint, LOCO images or Prover9/Mace4 binary was
available. The existing anomalib-dependent tests were not used as validation for
this isolated addition.
