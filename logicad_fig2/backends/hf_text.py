from __future__ import annotations

import json

from .hf_common import HFGenerativeModel, dependencies, pretrained_kwargs


class HFTextBackend(HFGenerativeModel):
    def signature(self):
        return {**super().signature(), "role": "text", "prompt_mode": "chat_template_or_plain_v5"}

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
                " Identity uncertainty rules: Alternative names for the SAME observed object must remain in ONE "
                "entry; never create an entry for each possible identity. One peach or nectarine means one entry "
                "with object nectarine_or_peach, count 1, present true; it does not mean one peach plus one "
                "nectarine. One apple/nectarine/peach means object apple_or_nectarine_or_peach, count 1, present "
                "true; do not choose apple alone or replace the known count with null. Use lowercase snake_case "
                "object names, and sort alternative names alphabetically joined by _or_. Preserve all stated "
                "identity alternatives without adding new ones. Keep separately observed objects separate; two "
                "oranges and one nectarine remain two entries with counts 2 and 1. Before answering, check that no "
                "object or count was added, duplicated, or dropped."
                " Quantity preservation rules: Set count to an integer only when the observations explicitly state "
                "an exact count for that object in that compartment. Presence alone does not mean count 1. Words "
                "such as present, some, several, multiple, a portion, or a mixture do not provide an exact count; "
                "use count null. Do not use numbered-list labels or the number of food types as object counts. For "
                "example, banana chips present with several slices means count null and present true, not count 1. "
                "Preserve explicit counts such as two oranges as count 2. If absence is explicit and an object entry "
                "is included, use count 0 and present false. Preserve an explicitly empty compartment as an empty "
                "array. Do not omit explicitly observed objects."
                " For unknown values, use JSON null when the schema permits null; never use strings such as "
                "'unknown', 'null', or 'N/A'. Counts must be nonnegative JSON integers or null, never strings. "
                "Presence values must be JSON true, false, or null, never strings."
                " The schema describes the output; do not output the schema itself. Array-valued fields must contain "
                "arrays directly, never objects wrapping an array in an items key. The items keyword describes each "
                "array element, not an extra output field. Use only the supplied observations; do not invent or "
                "discard observations to satisfy the schema. Schema:\n"
            ) + json.dumps(schema, sort_keys=True)
        chat = bool(getattr(self._processor, "chat_template", None))
        if chat:
            prompt = self._processor.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        inputs = self._processor(prompt, return_tensors="pt", add_special_tokens=not chat)
        return self._generate(inputs, generation)
