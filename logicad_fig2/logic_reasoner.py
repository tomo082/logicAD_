from __future__ import annotations

import itertools
import logging
import re
import shutil
import subprocess
from pathlib import Path

from formal_prompts_spec import logical_spec_dict

from .cache import fingerprint
from .logic_syntax import Formula, parse_formulas
from .prompts import (ALIASES, NORMAL_LOGIC_PROMPT, PREDICATE_FEATURES, STRING,
                      ZERO_DEFAULT_PREDICATES, array, get_prompts, obj)
from .structured import cached_structured

FORMAL_SCHEMA = obj(formulas=array(STRING))


def resolve_executable(value, name):
    if value:
        path = Path(value).expanduser()
        if path.is_dir():
            path /= name
        if path.is_file():
            return str(path.resolve())
        return shutil.which(value)
    return shutil.which(name)


def complete_query(normal, query):
    """Reuse legacy zero defaults, but match complete atoms rather than substrings."""
    mentioned = {(a.name, a.args[:-1]) for f in query for a in f.atoms()}
    defaults = []
    for formula in normal:
        for atom in formula.atoms():
            key = (atom.name, atom.args[:-1])
            if atom.name in ZERO_DEFAULT_PREDICATES and key not in mentioned:
                defaults.append(Formula("atom", atom.name, (*atom.args[:-1], "0")))
                mentioned.add(key)
    return [*query, *defaults], defaults


def build_axioms(normal, query, predicate_features):
    """Γ components: normal constraints, unique names, functionality and domain closure.

    Canonical synonyms share a symbol before naming inequalities are generated.
    c_ prefixes keep numeric and u-z constants from being interpreted as variables.
    """
    constants = sorted(set().union(*(f.constants() for f in [*normal, *query])) | {"0", "irrel"})
    naming = [f"c_{a} != c_{b}." for a, b in itertools.combinations(constants, 2)]
    functional, closure = [], []
    for name, features in predicate_features.items():
        pred = "p_" + name
        if "f" in features:
            if "b" in features:
                functional.append(f"all x (exists y {pred}(x,y)).")
                functional.append(f"all x all y all z (({pred}(x,y) & {pred}(x,z)) -> y=z).")
            else:
                functional.append(f"exists x {pred}(x).")
                functional.append(f"all x all y (({pred}(x) & {pred}(y)) -> x=y).")
        if "c" in features:
            objects = sorted({a.args[0] for f in normal for a in f.atoms() if a.name == name})
            rhs = " | ".join("x=c_" + name for name in objects)
            closure.append(f"all x ({rhs} | {pred}(x,c_0))." if rhs else f"all x {pred}(x,c_0).")
    return {"sigma_norm": [f.render(encoded=True) + "." for f in normal], "sigma_na": naming,
            "sigma_fa": functional, "sigma_dca": closure, "constants": constants}


def atp_program(axioms, query, timeout, *, prover=True):
    assumptions = "\n".join(line for key in ("sigma_norm", "sigma_na", "sigma_fa", "sigma_dca") for line in axioms[key])
    description = " & ".join("(" + f.render(encoded=True) + ")" for f in query)
    if prover:
        # Empty query checks Γ itself for inconsistency; otherwise prove Γ |= ¬Σ0.
        goal = "-(" + description + ")" if description else "$F"
        tail = f"end_of_list.\nformulas(goals).\n{goal}.\nend_of_list.\n"
    else:
        # Finite model for Γ ∪ Σ0 certifies non-entailment of ¬Σ0.
        if description:
            assumptions += "\n" + description + "."
        tail = "end_of_list.\n"
    preamble = f"assign(max_seconds,{timeout}).\n"
    if not prover:
        # Unique-name axioms require at least this many domain elements.
        preamble += f"assign(start_size,{max(2, len(axioms['constants']))}).\n"
    return preamble + "formulas(assumptions).\n" + assumptions + "\n" + tail


def minimal_inconsistent_subset(query, check, max_checks=64):
    """Deletion-based query-relative irreducible subset, never claim minimum cardinality.

    check(subset) returns True (proof), False (countermodel), None (unknown).
    Unknown outcomes prevent certification of minimality.
    """
    subset = list(query)
    checks, certified = 0, True
    for formula in list(query):
        if checks >= max_checks:
            certified = False
            break
        trial = list(subset)
        trial.remove(formula)
        answer = check(trial)
        checks += 1
        if answer is True:
            subset = trial
        elif answer is None:
            certified = False
    return subset, {"minimality_certified": certified, "checks": checks, "kind": "query_relative_deletion_subset"}


def explain_subset(subset):
    explanations = []
    for formula in subset:
        if formula.op == "atom" and len(formula.args) == 2 and formula.name in ZERO_DEFAULT_PREDICATES:
            item, number = formula.args
            display = {"cereal": "granola/cereal"}.get(item, item.replace("_", " "))
            location = f" on the {formula.name}" if formula.name in ("left", "right") else ""
            explanations.append(f"missing {display}{location}" if number == "0" else f"{display}: {number}{location}")
        else:
            explanations.append(formula.render())
    return "These observations conflict with the normal specification: " + "; ".join(explanations)


class LogicReasoner:
    def __init__(self, client, cache, config, runner=subprocess.run):
        self.client, self.cache, self.config, self.runner = client, cache, config, runner
        self.prover = resolve_executable(config.prover9_path, "prover9")
        self.mace = resolve_executable(config.mace4_path, "mace4")
        if not self.prover:
            logging.warning("Prover9 not installed/found: skipping Logic Reasoner; Format Embedding remains enabled")
        elif not self.mace:
            logging.warning("Mace4 not found: normal consistency/countermodels cannot be certified; verdicts may be unknown")

    def signature(self):
        return {"prover9": self.prover, "mace4": self.mace, "timeout": self.config.prover_timeout,
                "normal_rules": self.config.normal_rules, "max_mis_checks": self.config.max_mis_checks,
                "logic_model": self.config.logic_model, "version": 1}

    def formalize(self, text, category, *, normal=False):
        prompts = get_prompts(category)
        prompt = prompts.logic + (NORMAL_LOGIC_PROMPT if normal else "Convert only the observed query facts. ")
        if category == "splicing_connectors":
            # Existing position(cable, slot) + functionality enforces same slots;
            # preserve *both* observed slots, never collapse conflicting facts.
            prompt += " Preserve every observed position(cable,slot), including multiple differing slot values. "
        prompt += "\nObservations:\n" + text
        features, aliases = PREDICATE_FEATURES[category], ALIASES.get(category, {})
        settings = {"prompt": prompt, "model": self.config.logic_model, "schema": FORMAL_SCHEMA,
                    "max_tokens": self.config.max_tokens, "features": features, "aliases": aliases}
        key = f"formal/{category}/{fingerprint(settings)}.json"
        value = cached_structured(self.client, self.cache, key, prompt=prompt, schema=FORMAL_SCHEMA,
                                  model=self.config.logic_model, retries=self.config.json_retries,
                                  validator=lambda value: parse_formulas(value["formulas"], features, aliases))
        return parse_formulas(value["formulas"], features, aliases), key

    def _run(self, program, executable, kind):
        key = f"prover_calls/{fingerprint({'program': program, 'executable': executable})}.json"
        record = self.cache.get(key)
        if record is not None and record.get("status") in ("proof", "model"):
            return record, key
        # Actual readable .in/.out files accompany raw JSON subprocess evidence.
        base = self.cache.path(key).with_suffix("")
        base.parent.mkdir(parents=True, exist_ok=True)
        base.with_suffix(".in").write_text(program, encoding="utf-8")
        try:
            completed = self.runner([executable], input=program, capture_output=True, text=True,
                                    timeout=self.config.prover_timeout + 3, shell=False)
            stdout, stderr = completed.stdout or "", completed.stderr or ""
            status = "unknown"
            if kind == "prover9" and completed.returncode == 0 and re.search(r"=+ PROOF =+", stdout):
                status = "proof"
            if kind == "mace4" and completed.returncode == 0 and "interpretation(" in stdout:
                status = "model"
            record = {"status": status, "returncode": completed.returncode, "stdout": stdout, "stderr": stderr,
                      "program": program, "executable": executable}
        except subprocess.TimeoutExpired:
            record = {"status": "unknown", "reason": "timeout", "program": program, "executable": executable}
        except OSError as exc:
            record = {"status": "unknown", "reason": "execution_error", "error": str(exc), "program": program}
        base.with_suffix(".out").write_text(record.get("stdout", "") + "\n" + record.get("stderr", ""), encoding="utf-8")
        return self.cache.put(key, record), key

    def check(self, axioms, query, *, model_first=False):
        evidence = []
        kinds = ("mace4", "prover9") if model_first else ("prover9", "mace4")
        for kind in kinds:
            executable = self.prover if kind == "prover9" else self.mace
            if not executable:
                continue
            program = atp_program(axioms, query, self.config.prover_timeout, prover=kind == "prover9")
            record, key = self._run(program, executable, kind)
            evidence.append(key)
            if record["status"] == "proof":
                return True, evidence
            if record["status"] == "model":
                return False, evidence
        return None, evidence

    def reason(self, normal_text, query_text, category):
        if not self.prover:
            return {"status": "skipped", "reason": "prover9_not_found"}
        settings = {"normal": normal_text, "query": query_text, "settings": self.signature(),
                    "prompts": get_prompts(category).logic, "features": PREDICATE_FEATURES[category],
                    "aliases": ALIASES.get(category, {})}
        key = f"reasoning/{category}/{fingerprint(settings)}.json"
        saved = self.cache.get(key)
        if saved and saved.get("status") in ("normal", "abnormal"):
            return saved
        if self.config.normal_rules == "legacy":
            if category not in logical_spec_dict:
                return {"status": "skipped", "reason": "legacy_normal_rules_unavailable_for_category"}
            normal = parse_formulas([line.strip() for line in logical_spec_dict[category]["norm_spec"].splitlines()
                                     if line.strip()], PREDICATE_FEATURES[category], ALIASES.get(category), True)
            normal_key = "formal_prompts_spec.py:logical_spec_dict/" + category
        else:
            normal, normal_key = self.formalize(normal_text, category, normal=True)
        query, query_key = self.formalize(query_text, category)
        if any("unknown" in f.constants() for f in [*normal, *query]):
            return {"status": "unknown", "reason": "explicitly_uncertain_formal_description",
                    "normal_formal_cache": normal_key, "query_formal_cache": query_key}
        query, defaults = complete_query(normal, query)
        axioms = build_axioms(normal, query, PREDICATE_FEATURES[category])
        record = {"normal_formal_cache": normal_key, "query_formal_cache": query_key,
                  "sigma0": [f.render() for f in query], "default_facts": [f.render() for f in defaults],
                  "gamma": axioms, "normal_rule_source": self.config.normal_rules}
        # An inconsistent Γ proves everything. Never attribute its contradictions
        # to a query anomaly; require a model witnessing baseline consistency.
        baseline, baseline_evidence = self.check(axioms, [], model_first=True)
        record["baseline_evidence"] = baseline_evidence
        if baseline is not False:
            record.update(status="unknown", reason="inconsistent_normal_specification" if baseline else "normal_consistency_unverified")
            return self.cache.put(key, record)
        verdict, evidence = self.check(axioms, query)
        record["evidence"] = evidence
        if verdict is None:
            record.update(status="unknown", reason="no_proof_or_countermodel_within_limits")
        elif verdict is False:
            record.update(status="normal", explanation="A finite model satisfies the normal specification and query facts.")
        else:
            evidence_by_subset = []

            def check_subset(subset):
                answer, proofs = self.check(axioms, subset)
                evidence_by_subset.append({"facts": [f.render() for f in subset], "result": answer, "evidence": proofs})
                return answer

            subset, details = minimal_inconsistent_subset(query, check_subset, self.config.max_mis_checks)
            record.update(status="abnormal", explanation=explain_subset(subset),
                          inconsistent_subset=[f.render() for f in subset], subset_details=details,
                          subset_evidence=evidence_by_subset)
        return self.cache.put(key, record)
