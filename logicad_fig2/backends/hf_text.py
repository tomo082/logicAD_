from __future__ import annotations

import json

from .hf_common import HFGenerativeModel, dependencies, pretrained_kwargs


class HFTextBackend(HFGenerativeModel):
    def signature(self):
        return {**super().signature(), "role": "text", "prompt_mode": "chat_template_or_plain_v3"}

    @staticmethod
    def _load_pretrained(model_id, options):
        torch, transformers = dependencies()
        kwargs, device = pretrained_kwargs(options, torch, transformers)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id, revision=options.revision, local_files_only=options.local_files_only,
            trust_remote_code=False)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if options.device_map == "none":
            model.to(device)
        return model, tokenizer

    def generate(self, prompt, schema, generation):
        self._load()
        if schema is not None:
            prompt += (
                "\nReturn only JSON data conforming to this schema."
                " For unknown values, use JSON null when the schema permits null; "
                "never use strings such as 'unknown', 'null', or 'N/A'. "
                "Counts must be nonnegative JSON integers or null, never strings. "
                "Presence values must be JSON true, false, or null, never strings."
                " The schema describes the output; do not output the schema itself. "
                "Array-valued fields must contain arrays directly, never objects "
                "wrapping an array in an items key. "
                "The items keyword describes each array element, not an extra output field. "
                "Use only the supplied observations; do not invent or discard observations "
                "to satisfy the schema. Schema:\n"
            ) + json.dumps(schema, sort_keys=True)
        chat = bool(getattr(self._processor, "chat_template", None))
        if chat:
            prompt = self._processor.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        inputs = self._processor(prompt, return_tensors="pt", add_special_tokens=not chat)
        return self._generate(inputs, generation)
