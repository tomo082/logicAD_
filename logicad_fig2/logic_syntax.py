"""A small validated formula AST; never execute or paste LLM text as an ATP program."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Formula:
    op: str
    name: str = ""
    args: tuple = ()
    children: tuple = ()

    def atoms(self):
        if self.op == "atom":
            return [self]
        return [atom for child in self.children for atom in child.atoms()]

    def constants(self, bound=()):
        if self.op == "atom":
            return set(self.args) - set(bound)
        if self.op in ("all", "exists"):
            bound = (*bound, self.name)
        return set().union(*(child.constants(bound) for child in self.children))

    def render(self, *, encoded=False, bound=()):
        if self.op == "atom":
            pred = "p_" + self.name if encoded else self.name
            args = [("c_" + arg if encoded and arg not in bound else arg) for arg in self.args]
            return pred + "(" + ",".join(args) + ")"
        if self.op == "not":
            return "-(" + self.children[0].render(encoded=encoded, bound=bound) + ")"
        if self.op in ("all", "exists"):
            return self.op + " " + self.name + " (" + self.children[0].render(encoded=encoded, bound=(*bound, self.name)) + ")"
        separator = " & " if self.op == "and" else " | "
        return "(" + separator.join(c.render(encoded=encoded, bound=bound) for c in self.children) + ")"


class FormulaParser:
    def __init__(self, text, predicates, aliases=None, allow_quantifiers=False):
        if len(text) > 10000:
            raise ValueError("Formal statement is too long")
        text = text.strip().removesuffix(".").replace("%", "percent")
        for word, symbol in (("AND", "&"), ("OR", "|"), ("NOT", "-")):
            text = re.sub(r"\b" + word + r"\b", symbol, text)
        self.tokens = re.findall(r"[a-z0-9_]+|[(),&|~-]", text)
        if "".join(self.tokens) != re.sub(r"\s+", "", text):
            raise ValueError("Unsupported formal syntax")
        self.index, self.predicates, self.aliases = 0, predicates, aliases or {}
        self.allow_quantifiers = allow_quantifiers

    def peek(self):
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def take(self, expected=None):
        token = self.peek()
        if token is None or expected is not None and token != expected:
            raise ValueError(f"Expected {expected or 'token'}, got {token}")
        self.index += 1
        return token

    def parse(self):
        formula = self.disjunction()
        if self.peek() is not None:
            raise ValueError("Trailing tokens in formula")
        return formula

    def disjunction(self):
        formula = self.conjunction()
        while self.peek() == "|":
            self.take()
            formula = Formula("or", children=(formula, self.conjunction()))
        return formula

    def conjunction(self):
        formula = self.unary()
        while self.peek() == "&":
            self.take()
            formula = Formula("and", children=(formula, self.unary()))
        return formula

    def unary(self):
        token = self.take()
        if token in ("-", "~"):
            return Formula("not", children=(self.unary(),))
        if token == "(":
            result = self.disjunction()
            self.take(")")
            return result
        if token in ("all", "exists"):
            if not self.allow_quantifiers:
                raise ValueError("Generated formalizations must be ground formulas")
            var = self.take()
            if var not in ("x", "y", "z"):
                raise ValueError("Invalid quantified variable")
            return Formula(token, name=var, children=(self.unary(),))
        if token not in self.predicates:
            raise ValueError(f"Unknown predicate: {token}")
        self.take("(")
        args = [self.take()]
        while self.peek() == ",":
            self.take()
            args.append(self.take())
        self.take(")")
        if not all(re.fullmatch(r"[a-z0-9][a-z0-9_]*", arg) for arg in args):
            raise ValueError("Invalid constant or nested function")
        expected = 2 if "b" in self.predicates[token] else 1
        if len(args) != expected:
            raise ValueError(f"Wrong arity for {token}: expected {expected}")
        return Formula("atom", name=token, args=tuple(self.aliases.get(arg, arg) for arg in args))


def parse_formulas(lines, predicates, aliases=None, allow_quantifiers=False):
    if not lines or len(lines) > 128:
        raise ValueError("Expected 1..128 formal statements")
    return [FormulaParser(line, predicates, aliases, allow_quantifiers).parse() for line in lines]
