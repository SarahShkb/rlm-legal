"""Parse a model answer into a LexGLUE prediction, recording how it was parsed.

parse_status values (stored per record so failures can be audited/reported):
  "ok"              JSON with the expected key, every label copied exactly
  "normalized"      JSON ok, but a label only matched after case/space/punctuation
                    normalisation or an ECtHR article alias ("Article 3", "P1-1")
  "no_json"         no JSON object, but the whole answer is a valid label / index
  "partial_invalid" multi-label: some labels are not candidates (kept in `invalid`)
  "invalid_label"   the answer names no candidate label / index
  "unparseable"     nothing usable in the answer (prediction is None)
Only "invalid_label" and "unparseable" leave the prediction empty (None).
"""

import ast
import json
import re
from dataclasses import dataclass, field

from lexglue.config import MULTI_CLASS, MULTI_LABEL, TASKS

_FENCE = re.compile(r"```(?:json|python|repl)?\s*(.*?)```", re.DOTALL)


@dataclass
class ParseResult:
    prediction: str | list[str] | int | None
    status: str
    invalid: list[str] = field(default_factory=list)


def _norm(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip().strip(".,;:\"'`").strip()
    return s.lower()


def _json_candidates(text: str) -> list:
    """Every JSON/Python-literal object found in the text, last one first."""
    chunks = [text.strip()] + [m.strip() for m in _FENCE.findall(text)]
    # balanced {...} spans
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                chunks.append(text[start:i + 1])
    found = []
    for chunk in chunks:
        for loader in (json.loads, ast.literal_eval):
            try:
                obj = loader(chunk)
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                continue
            # answer["content"] is sometimes a JSON string holding the JSON object
            if isinstance(obj, str) and obj.strip().startswith("{"):
                try:
                    obj = json.loads(obj)
                except ValueError:
                    continue
            if isinstance(obj, (dict, list, int)) and not isinstance(obj, bool):
                found.append(obj)
            break
    return found[::-1]


class LabelMatcher:
    def __init__(self, task: str, labels: list[str]):
        self.labels = labels
        self.exact = set(labels)
        self.loose = {_norm(l): l for l in labels}
        if task.startswith("ecthr"):
            # "Article 3 - Prohibition of torture" is also accepted as "Article 3" / "3"
            for l in labels:
                code = l.split(" - ")[0].replace("Article ", "")
                self.loose.setdefault(_norm(code), l)
                self.loose.setdefault(_norm(f"Article {code}"), l)

    def match(self, s) -> tuple[str | None, bool]:
        """(label, exact?) or (None, False)."""
        if not isinstance(s, str):
            return None, False
        if s in self.exact:
            return s, True
        return self.loose.get(_norm(s)), False


def parse_prediction(task: str, labels: list[str] | None, n_options: int, text: str | None) -> ParseResult:
    if not text or not str(text).strip():
        return ParseResult(None, "unparseable")
    text = str(text)
    task_type = TASKS[task]["task_type"]
    candidates = _json_candidates(text)

    if task_type == MULTI_CLASS:
        matcher = LabelMatcher(task, labels)
        for obj in candidates:
            if isinstance(obj, dict) and "label" in obj:
                value = obj["label"]
                if isinstance(value, list) and len(value) == 1:
                    value = value[0]
                label, exact = matcher.match(value)
                if label is None:
                    return ParseResult(None, "invalid_label", [str(value)])
                return ParseResult(label, "ok" if exact else "normalized")
        label, _ = matcher.match(text)
        if label is not None:
            return ParseResult(label, "no_json")
        return ParseResult(None, "unparseable")

    if task_type == MULTI_LABEL:
        matcher = LabelMatcher(task, labels)
        found, values, status = False, None, "unparseable"
        for obj in candidates:
            if isinstance(obj, dict) and "labels" in obj:
                found, values, status = True, obj["labels"], "ok"
                break
            if isinstance(obj, list):
                found, values, status = True, obj, "no_json"
                break
        if not found:
            if _norm(text) in ("none", "[]", "no labels"):
                return ParseResult([], "no_json")
            return ParseResult(None, "unparseable")
        if values is None or values == "NONE":
            values = []
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            return ParseResult(None, "unparseable")
        valid, invalid, all_exact = set(), [], True
        for v in values:
            if isinstance(v, str) and _norm(v) == "none":
                continue
            label, exact = matcher.match(v)
            if label is None:
                invalid.append(str(v))
            else:
                valid.add(label)
                all_exact &= exact
        if invalid:
            status = "partial_invalid" if valid else "invalid_label"
            # an answer made only of invalid labels is a wrong (empty-handed) answer
            return ParseResult(sorted(valid) if valid else None, status, invalid)
        if status == "ok" and not all_exact:
            status = "normalized"
        return ParseResult(sorted(valid), status)

    # multiple choice (CaseHOLD)
    for obj in candidates:
        value = obj.get("answer") if isinstance(obj, dict) else None
        if value is None:
            continue
        try:
            idx = int(str(value).strip())
        except ValueError:
            return ParseResult(None, "invalid_label", [str(value)])
        if 0 <= idx < n_options and str(value).strip() == str(idx):
            return ParseResult(idx, "ok")
        return ParseResult(None, "invalid_label", [str(value)])
    m = re.fullmatch(r"\s*(\d)\s*", text)
    if m and int(m.group(1)) < n_options:
        return ParseResult(int(m.group(1)), "no_json")
    return ParseResult(None, "unparseable")


def is_correct(task: str, prediction, gold) -> bool:
    """Exact match (exact label set for multi-label tasks)."""
    if prediction is None:
        return False
    if TASKS[task]["task_type"] == MULTI_LABEL:
        return set(prediction) == set(gold)
    return prediction == gold
