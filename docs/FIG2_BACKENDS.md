# Backend refactor audit (Phase 1)

Baseline: `a545567`. Scope: a Figure 2/3-inspired interchangeable experimental
pipeline, not a claim to reproduce a particular GPT-4o deployment's performance.

| Current coupling | Change boundary |
| --- | --- |
| `TextExtractor.describe` calls `client.chat(images=..., model=...)` | `VisionLanguageBackend.describe(images, prompt, generation)` |
| `cached_structured` calls the same `client.chat` for JSON | `TextGenerationBackend.generate(prompt, schema, generation)`; keep parsing/repair |
| `cached_embeddings` assumes `client.embed(model=...)` | `EmbeddingBackend.embed(texts)`; preserve normalization, LOF and cosine |
| `LogicReasoner.formalize` uses the shared OpenAI client | Independent text backend for logic; keep ATP code |
| CLI constructs one `OpenAIClient` for all stages | Factory builds a bundle of four independently selected adapters |
| `Config` and cache keys record model names, but not providers/load options | Backend identity + generation settings in every relevant cache and experiment key |
| `vlm.py` imports torch/transformers and uses legacy pipeline output | New lazy HF adapter using model-owned chat templates and automatic image-text model class |

Design: three Protocols (vision, text generation, embedding), four injected roles.
Adapters return small serializable response records including text/vectors, raw
outputs and backend metadata. The existing client interface is accepted by a
compatibility wrapper for existing callers/tests, not used for provider dispatch
inside the core algorithm. OpenAI transport retains its retry/usage behavior.

HF VLM starts with LLaVA-NeXT. Its default is sequential original/ROI inference
with labeled observation aggregation; a native multi-image adapter is explicit.
Processor chat templates and AutoModel classes provide the extension boundary.
Local text generation uses a causal model; embeddings use SentenceTransformer.
Heavy libraries and weights load only on first inference. Backend unload methods
and optional unloading on stage switches allow VRAM to be reclaimed.

Reference sources checked before implementation:

- [LLaVA 1.6 Vicuna 13B model card](https://huggingface.co/llava-hf/llava-v1.6-vicuna-13b-hf)
- [Transformers LLaVA-NeXT](https://huggingface.co/docs/transformers/en/model_doc/llava_next)
- [Transformers chat templates](https://huggingface.co/docs/transformers/en/chat_templating)
- [SentenceTransformer API](https://sbert.net/docs/package_reference/sentence_transformer/model.html)

Validation order: existing OpenAI tests after interface migration; stubbed HF
loader/generation checks; LLaVA+OpenAI hybrid construction; complete local pipeline
with fake backends and no key; CLI/cache/lifecycle regression tests. No live
benchmark or multi-GB weight download is required by this request.

## Validation and limitations (Phases 2–7)

- Original 45 passing tests preserved after OpenAI interface migration.
- HF vision: lazy loading, LLaVA AutoModel dispatch, processor settings,
  sequential/native images, seeds, prompt-token removal, sampling and unload.
- Embedding: arbitrary dimensions and model cache separation.
- Text: existing JSON validation and repair through HF generation.
- Factory: LLaVA + OpenAI construction, then a fully local bundle without
  importing the OpenAI SDK or heavy libraries during construction.
- Complete local evaluation uses real HF adapters with fake weights/runtime.
  Separate role fakes cover K=3 duplicates, anomaly scores, lifecycle and resume.
- Windows Python 3.12 suite: 60 passed, 2 skipped (external ATP binaries absent).
- Real weights, quantized GPU execution, live APIs and DLBox benchmarks remain
  unverified; no weights were downloaded. Dependency ranges are not a tested
  lockfile. Future model compatibility needs adapter-level validation.
