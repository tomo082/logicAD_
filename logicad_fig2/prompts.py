"""All four prompt roles and category schemas. Provenance: docs/FIG2_RECONSTRUCTION.md.

Extraction wording and schemas are reconstructions; legacy formal examples are
imported verbatim from the repository's side-effect-free prompt module.
"""
from __future__ import annotations

from dataclasses import dataclass

from formal_prompts_spec import PROMPT0, RULE, lang_rules_dict, two_shot_dict, logical_spec_dict

CATEGORIES = ("breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors")


def obj(**properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def array(items):
    return {"type": "array", "items": items}


STRING = {"type": "string"}
COUNT = {"type": ["integer", "null"], "minimum": 0}
BOOL = {"type": ["boolean", "null"]}
ITEMS = array(obj(object=STRING, count=COUNT, present=BOOL))

SCHEMAS = {
    "breakfast_box": obj(left=ITEMS, right=ITEMS),
    "juice_bottle": obj(juice_color=STRING, fruit=STRING, fill_level=STRING, sticker_count=COUNT,
                        top_sticker_correct=BOOL, bottom_sticker_correct=BOOL,
                        fruit_matches_juice=BOOL, label_image_position=STRING, symmetrical=BOOL),
    "pushpins": obj(compartments=array(obj(position=STRING, pushpin_count=COUNT, walls_intact=BOOL)),
                    total_pushpins=COUNT),
    "screw_bag": obj(bolt_count=COUNT, washer_count=COUNT, nut_count=COUNT,
                     shorter_bolt_ratios=array(STRING), bolts_longer_than_three_washer_diameters=BOOL),
    "splicing_connectors": obj(connector_count=COUNT, cable_count=COUNT, cable_broken=BOOL,
                               same_size=BOOL, cable_positions=array(obj(block=STRING, position=STRING)),
                               same_slot=BOOL),
}

# Thresholds follow supplementary A.4. Connector keyword corrects an apparent
# copy/paste error in that table; --feature-prompt can reproduce its literal value.
FEATURE_PROMPTS = {
    "breakfast_box": "breakfast box",
    "juice_bottle": "juice bottle",
    "pushpins": "black square compartment",
    "screw_bag": "metal circle",
    "splicing_connectors": "connector block",
}

EXTRACTION_PROMPTS = {
    "breakfast_box": (
        "Inspect the left and right compartments separately. List each food and its count or presence. "
        "Distinguish apple/nectarine/peach; report citrus names consistently and state uncertainty. "
        "Inspect granola/cereal, banana chips and nuts on the right. Report missing and unexpected foods."
    ),
    "juice_bottle": (
        "Record juice color and pictured fruit; assess whether they agree. Inspect fill level relative to "
        "halfway up the neck, sticker count, the upper fruit sticker and centered fruit picture, "
        "the lower 100% juice sticker and its alignment, and overall sticker symmetry. "
        "Describe each observed feature, including absences."
    ),
    "pushpins": (
        "Inspect each compartment in reading order, assigning a row/column position. Count pushpins "
        "inside each, including empty compartments. Check dividing walls and multiple pins sharing "
        "one compartment. Reconcile the total with the original image."
    ),
    "screw_bag": (
        "Inventory bolts, washers and nuts separately. For closeups distinguish washers from nuts. "
        "Compare shorter bolts, including heads, to the longest using fifths of its length; also "
        "compare bolt length to three washer diameters. Report counts, relative sizes and uncertainty."
    ),
    "splicing_connectors": (
        "Count separate connector blocks and cables. Check cable continuity and equal block size. "
        "For each block inspect its cable insertion slot (top, middle or bottom); compare both ends "
        "of each cable. Report positions per block, including any mismatch and uncertain visibility."
    ),
}

VISUAL_CONTEXT = (
    "Image 1 is the complete inspected image. Subsequent images are overlapping ROI crops of that "
    "same image, not extra objects. Use crops for detail and the original for counts and locations. "
    "Give a concise factual observation record in English following the inspection steps. "
    "Do not invent hidden objects or classify the image as normal/abnormal. "
)

OBSERVATION_INSTRUCTION = (
    "Give a concise factual observation record in English following the inspection steps. "
    "Do not invent hidden objects or classify the image as normal/abnormal. "
)

FORMAT_PROMPT = (
    "Normalize the supplied observations into the category JSON schema. Preserve actual observations "
    "and uncertainty. Never repair an anomaly to match a normal pattern. Use null for unknown counts "
    "or booleans, 'unknown' for unknown strings, 0/false only for observed absence. Use lowercase "
    "singular object names. Sort item arrays by object name and spatial arrays by reading order. "
    "Uncountable food may have count=null and present=true. Do not add prose outside JSON."
)

LOGIC_EXTRA = {
    "pushpins": (
        "Predicates: count(object,number), pins_at(compartment,number), walls_intact(compartment). "
        "Name compartments row_1_col_1 etc. Examples: 'First compartment empty' -> "
        "pins_at(row_1_col_1,0); 'Two pins in first compartment' -> pins_at(row_1_col_1,2)."
    ),
    "screw_bag": (
        "Predicates: count(object,number), length_ratio(bolt,ratio), long_enough(bolt). "
        "Objects: bolt, washer, nut. Name bolts longest, shorter_1 etc ordered by length. "
        "Use ratio constants one_fifth, two_fifths, three_fifths, four_fifths, one. "
        "Examples: 'No washers' -> count(washer,0); 'Two nuts' -> count(nut,2)."
    ),
}

LOGIC_OUTPUT_PROMPT = (
    "Return JSON with a nonempty formulas array, one ground formula per string. Allowed syntax: "
    "lowercase predicate(constant,...), NOT, AND, OR, parentheses. No quantifiers, functions, "
    "comments or program directives. Retain disjunction for uncertain alternatives; do not guess. "
    "Use unknown for explicitly uncertain values. Use irrel for present uncountable quantities. "
)
NORMAL_LOGIC_PROMPT = (
    "This is the sole normal reference observation. Convert it to ground constraints defining "
    "normality using the given vocabulary. Preserve only supported observations and alternatives. "
    "Do not infer unobserved normal variants or use query images. "
)

# Predicate arity, functional-last-argument flag, domain-closure flag. Reuses
# formal_prompts_spec.py's u/b/f/c convention with two reconstructed categories.
PREDICATE_FEATURES = {
    category: dict(spec["predicate_feat"]) for category, spec in logical_spec_dict.items()
}
PREDICATE_FEATURES["juice_bottle"].update(color="bf", fruit_label="u")
PREDICATE_FEATURES["pushpins"] = {"count": "bfc", "pins_at": "bfc", "walls_intact": "u"}
PREDICATE_FEATURES["screw_bag"] = {"count": "bfc", "length_ratio": "bf", "long_enough": "u"}
ZERO_DEFAULT_PREDICATES = {"left", "right", "count", "patch_count", "pins_at"}

# Category-scoped canonical names adapted from formal_reasoner_all.py:reset_data.
# These task-level equivalences are not universal linguistic synonyms.
ALIASES = {
    "breakfast_box": {"mandarin": "orange", "tangerine": "orange", "clementine": "orange",
                      "mandarin_orange": "orange", "granola": "cereal", "oat": "cereal",
                      "almond": "nut", "amond": "nut", "whole_almond": "nut", "mixed_nut": "nut",
                      "dried_banana": "banana_chip", "dried_banana_chip": "banana_chip",
                      "dried_banana_slice": "banana_chip", "peach": "nectarine", "red_apple": "apple"},
    "juice_bottle": {"half_neck": "around_half_neck", "around_half_the_neck": "around_half_neck",
                     "horizontally_centered": "middle", "horizontally_centred": "middle"},
    "splicing_connectors": {"cabel": "cable", "connector": "connector_block"},
}


@dataclass(frozen=True)
class CategoryPrompts:
    category: str
    feature: str
    extraction: str
    format: str
    schema: dict
    logic: str


def get_prompts(category: str) -> CategoryPrompts:
    if category not in CATEGORIES:
        raise ValueError(f"Unsupported category: {category}")
    logic = PROMPT0 + RULE + lang_rules_dict.get(category, LOGIC_EXTRA.get(category, ""))
    logic += two_shot_dict.get(category, "") + "\n" + LOGIC_OUTPUT_PROMPT
    return CategoryPrompts(category, FEATURE_PROMPTS[category], OBSERVATION_INSTRUCTION + EXTRACTION_PROMPTS[category],
                           FORMAT_PROMPT, SCHEMAS[category], logic)
