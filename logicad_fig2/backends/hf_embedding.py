from __future__ import annotations

import gc
import sys


class HFEmbeddingBackend:
    def __init__(self, model_id, *, device="cpu", revision=None, local_files_only=False,
                 batch_size=32, loader=None):
        self.model_id, self.device, self.revision = model_id, device, revision
        self.local_files_only, self.batch_size = local_files_only, batch_size
        self._loader, self._model = loader, None

    def signature(self):
        return {"backend": "hf", "role": "embedding", "model": self.model_id,
                "device": self.device, "revision": self.revision,
                "local_files_only": self.local_files_only, "batch_size": self.batch_size,
                "adapter_version": 1}

    def embed(self, texts):
        if self._model is None:
            loader = self._loader
            if loader is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise RuntimeError("Install requirements_logicad_fig2_local.txt for local embeddings") from exc
                loader = SentenceTransformer
            self._model = loader(self.model_id, device=None if self.device == "auto" else self.device,
                                 revision=self.revision, local_files_only=self.local_files_only,
                                 trust_remote_code=False)
        vectors = self._model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True,
                                     normalize_embeddings=False, show_progress_bar=False).tolist()
        return {"vectors": vectors, "raw": {}, "metadata": {
            "backend": self.signature(), "dimension": len(vectors[0]) if vectors else 0,
            "max_seq_length": getattr(self._model, "max_seq_length", None)}}

    def unload(self):
        self._model = None
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
