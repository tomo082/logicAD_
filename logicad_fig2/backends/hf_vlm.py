from __future__ import annotations

from dataclasses import replace

from .hf_common import HFGenerativeModel, dependencies, pretrained_kwargs

NATIVE_CONTEXT = (
    "The first image is the complete image. All subsequent images are overlapping ROI crops of it. "
    "Use the original for counts and spatial relationships; crops are detail views, not extra objects.\n"
)
ROI_CONTEXT = (
    "This single image is ROI crop {index} of the original, not a separate scene. "
    "Report visible details, local counts and positions only. Do not infer whole-image counts.\n"
)


class HFVisionBackend(HFGenerativeModel):
    """HF image-text adapter. LLaVA-NeXT defaults to single-image calls + aggregation.

    native mode uses a processor-owned multimodal chat template in one call. It
    is opt-in: accepting image tokens does not imply a checkpoint was trained for
    multiple images. Subclass _infer/_load_pretrained for model-specific adapters.
    """
    def __init__(self, model_id, options=None, *, image_strategy="sequential", loader=None):
        super().__init__(model_id, options, loader=loader)
        if image_strategy not in ("sequential", "native"):
            raise ValueError("image_strategy must be sequential or native")
        self.image_strategy = image_strategy

    def signature(self):
        return {**super().signature(), "role": "vision", "multi_image_strategy": self.image_strategy,
                "visual_context": NATIVE_CONTEXT, "roi_context": ROI_CONTEXT}

    @staticmethod
    def _load_pretrained(model_id, options):
        torch, transformers = dependencies()
        kwargs, device = pretrained_kwargs(options, torch, transformers)
        processor = transformers.AutoProcessor.from_pretrained(
            model_id, revision=options.revision, local_files_only=options.local_files_only, trust_remote_code=False)
        model = transformers.AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        if options.device_map == "none":
            model.to(device)
        # Older LLaVA-NeXT processor snapshots need image-token expansion info.
        if getattr(model.config, "model_type", None) == "llava_next":
            for attr, value in (("patch_size", model.config.vision_config.patch_size),
                                ("vision_feature_select_strategy", model.config.vision_feature_select_strategy),
                                ("num_additional_image_tokens", 1)):
                if getattr(processor, attr, None) is None:
                    setattr(processor, attr, value)
        return model, processor

    def _infer(self, images, prompt, generation):
        self._load()
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        rendered = self._processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
        inputs = self._processor(text=rendered, images=list(images), return_tensors="pt")
        return self._generate(inputs, generation)

    def describe(self, images, prompt, generation):
        if not images:
            raise ValueError("Vision backend requires at least the original image")
        if self.image_strategy == "native":
            result = self._infer(images, NATIVE_CONTEXT + prompt, generation)
        else:
            responses = []
            for index, image in enumerate(images):
                context = "This is the complete original image.\n" if index == 0 else ROI_CONTEXT.format(index=index)
                roi_seed = None if generation.seed is None else (generation.seed + index * 1000003) % (2**32)
                response = self._infer([image], context + prompt, replace(generation, seed=roi_seed))
                responses.append(response)
            pieces = ["Original image observations:\n" + responses[0]["text"]]
            if len(responses) > 1:
                pieces.append("ROI detail observations below describe overlapping parts of the SAME original. "
                              "Do not sum repeated objects; retain original-image counts and spatial context.")
                pieces.extend(f"ROI {index}:\n{response['text']}" for index, response in enumerate(responses[1:], 1))
            result = {"text": "\n\n".join(pieces), "raw": {"per_image_responses": responses},
                      "finish_reason": "length" if any(r.get("finish_reason") == "length" for r in responses) else "stop"}
            if any(not r["text"].strip() for r in responses):
                result["text"] = ""  # Saved as a failed raw response by the extraction stage.
        result["metadata"] = {**result.get("metadata", {}), "backend": self.signature(),
                              "multi_image_strategy": self.image_strategy, "image_count": len(images),
                              "generation": generation.signature()}
        return result
