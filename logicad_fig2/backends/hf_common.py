"""Shared HF loading, generation and lifecycle utilities; no eager heavy imports."""
from __future__ import annotations

import gc
import importlib.util
import sys
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HFOptions:
    device: str = "auto"
    dtype: str = "auto"
    device_map: str | dict = "auto"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    revision: str | None = None
    local_files_only: bool = False
    offload_folder: str | None = None
    max_memory: dict | None = None

    def signature(self):
        return asdict(self)


def dependencies():
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError("HF backend requires requirements_logicad_fig2_local.txt; install it in the local venv") from exc
    return torch, transformers


def pretrained_kwargs(options, torch, transformers):
    device = ("cuda" if torch.cuda.is_available() else "cpu") if options.device == "auto" else options.device
    dtype = options.dtype
    if dtype == "auto":
        dtype = "float16" if device.startswith("cuda") else "float32"
    dtype = getattr(torch, dtype)
    device_map = options.device_map
    if device_map == "auto" and (options.device != "auto" or device == "cpu"):
        device_map = {"": device}
    kwargs = {"dtype": dtype, "revision": options.revision, "local_files_only": options.local_files_only,
              "trust_remote_code": False}
    if device_map != "none":
        kwargs["device_map"] = device_map
    if options.offload_folder:
        kwargs["offload_folder"] = options.offload_folder
    if options.max_memory:
        kwargs["max_memory"] = {int(k) if str(k).isdigit() else k: v for k, v in options.max_memory.items()}
    if options.load_in_4bit and options.load_in_8bit:
        raise ValueError("Choose either 4-bit or 8-bit quantization")
    if options.load_in_4bit or options.load_in_8bit:
        if importlib.util.find_spec("bitsandbytes") is None:
            raise RuntimeError("Quantization requested but bitsandbytes is missing; install bitsandbytes")
        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("This adapter's bitsandbytes quantization requires CUDA; use unquantized float32 for CPU")
        if device_map == "none":
            raise ValueError("Quantized models require a device_map (use auto)")
        kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=options.load_in_4bit, load_in_8bit=options.load_in_8bit,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
            llm_int8_enable_fp32_cpu_offload=bool(options.max_memory or options.offload_folder))
    return kwargs, device


class HFGenerativeModel:
    def __init__(self, model_id, options=None, *, loader=None):
        self.model_id, self.options = model_id, options or HFOptions()
        self._loader = loader
        self._model = self._processor = None

    def signature(self):
        return {"backend": "hf", "model": self.model_id, "options": self.options.signature(), "adapter_version": 1}

    def _load(self):
        if self._model is None:
            model, processor = (self._loader or self._load_pretrained)(self.model_id, self.options)
            self._model, self._processor = model, processor
            self._model.eval()

    def unload(self):
        self._model = self._processor = None
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _generate(self, inputs, generation):
        import torch

        # Accelerate handles offloaded layers. Feed the input embedding's device;
        # never call model.to() on an already dispatched/quantized model.
        device = self._model.device
        embedding = self._model.get_input_embeddings()
        if getattr(getattr(embedding, "weight", None), "device", None) is not None:
            if str(embedding.weight.device) != "meta":
                device = embedding.weight.device
        if str(device) == "meta":
            raise RuntimeError("Cannot determine HF input device for this dispatch map")
        moved = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                value = value.to(device)
                if key == "pixel_values" and value.is_floating_point():
                    value = value.to(dtype=self._model.dtype)
            moved[key] = value
        kwargs = {"max_new_tokens": generation.max_tokens, "do_sample": generation.do_sample}
        if generation.do_sample:
            if generation.temperature <= 0:
                raise ValueError("Sampling requires a positive temperature")
            kwargs.update(temperature=generation.temperature, top_p=generation.top_p)
        # Keep other code's RNG intact, and vary seeds by image/K/ROI upstream.
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            if generation.seed is not None:
                torch.manual_seed(generation.seed)
            output = self._model.generate(**moved, **kwargs)
        if getattr(self._model.config, "is_encoder_decoder", False):
            generated = output[0]
        else:
            generated = output[0][moved["input_ids"].shape[-1]:]
        ids = generated.tolist()
        text = self._processor.decode(generated, skip_special_tokens=True).strip()
        eos = getattr(self._model.generation_config, "eos_token_id", None)
        eos_ids = eos if isinstance(eos, (list, tuple)) else [eos]
        truncated = len(ids) >= generation.max_tokens and (not ids or ids[-1] not in eos_ids)
        return {"text": text, "raw": {"generated_token_ids": ids},
                "finish_reason": "length" if truncated else "stop",
                "metadata": {"backend": self.signature(), "generation": generation.signature(),
                             "input_device": str(device), "dtype": str(self._model.dtype),
                             "input_tokens": moved["input_ids"].shape[-1], "output_tokens": len(ids)}}
