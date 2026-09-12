from __future__ import annotations

import json

from .hf_common import HFGenerativeModel, dependencies, pretrained_kwargs


class HFTextBackend(HFGenerativeModel):
    def signature(self):
        return {**super().signature(), "role": "text", "prompt_mode": "chat_template_or_plain_v1"}

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
            prompt += "\nReturn only JSON conforming to this schema:\n" + json.dumps(schema, sort_keys=True)
        chat = bool(getattr(self._processor, "chat_template", None))
        if chat:
            prompt = self._processor.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        inputs = self._processor(prompt, return_tensors="pt", add_special_tokens=not chat)
        return self._generate(inputs, generation)
