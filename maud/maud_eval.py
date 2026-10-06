"""
maud_eval.py -- score the vanilla / RLM log against the MAUD `sample` dataframe.

Notebook usage
--------------
    from maud_eval import add_gold_letters, build_results, summarize, paired_table

    add_gold_letters(sample, question_answer_map_unique, gold_to_letters)   # adds sample["gold_letters"]
    res = build_results(sample, question_answer_map_unique,
                        "maud_152_different_prompts_per_tier.csv")
    print(res.filter(like="_status").apply(pd.Series.value_counts))         # parsing audit
    print(summarize(res))
    print(paired_table(res, policy="lenient"))

Audit the log alone (no `sample` needed):
    python maud_eval.py maud_152_different_prompts_per_tier.csv
"""
from __future__ import annotations

import ast
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator, Optional

import pandas as pd

from maud.maud_data_sampling import get_sample_rows_for_maud_152, gold_to_letters

# ----------------------------------------------------------------------------
# 1. Reading the log
# ----------------------------------------------------------------------------
_ROW_ID_RE = re.compile(r"^contract_(\d+)_(\d+)$")


def read_log(path: str) -> pd.DataFrame:
    """Read the log CSV.

    * Skips the leading '# ...' comment line(s) -- but ONLY leading ones. Do not use
      pd.read_csv(comment="#"): the raw_result cells contain multi-line Python code
      whose '#' comments would be truncated/corrupted.
    * Reads everything as str (no NaN coercion) so answers are never altered.
    * Multi-line quoted cells are handled by the CSV parser itself.
    * Keeps the last record per (row_id, method) if a run was repeated.
    * row_id 'contract_<contract_no>_<pos>'  ->  `pos` = 0-based row position in `sample`.
    """
    n_skip = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.lstrip().startswith("#") or not line.strip():
                n_skip += 1
            else:
                break

    log = pd.read_csv(path, skiprows=n_skip, dtype=str, keep_default_na=False)
    missing = {"row_id", "method", "raw_result"} - set(log.columns)
    if missing:
        raise ValueError(f"Log is missing required columns: {missing}")

    ids = log["row_id"].str.extract(_ROW_ID_RE)
    if ids.isna().any().any():
        bad = log.loc[ids[0].isna(), "row_id"].tolist()[:5]
        raise ValueError(f"Unexpected row_id format, e.g. {bad}")
    log["contract_no"] = ids[0].astype(int)
    log["pos"] = ids[1].astype(int)

    if "timestamp" in log.columns:
        log = log.sort_values("timestamp", kind="stable")
    n_before = len(log)
    log = log.drop_duplicates(["row_id", "method"], keep="last")
    if len(log) < n_before:
        print(f"[read_log] dropped {n_before - len(log)} duplicate (row_id, method) records (kept latest)")
    return log.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. Normalising "an answer" into a set of letters
# ----------------------------------------------------------------------------
_LETTER_LIST_RE = re.compile(r"[A-Z](?:\s*(?:,|;|&|\band\b)\s*[A-Z])*")
_ANSWER_PREFIX_RE = re.compile(r"^\s*(?:final\s+)?answer\s*[:\-]\s*", re.I)
_ANSWER_LINE_RE = re.compile(r"^\s*Answer:\s*([A-Z](?:\s*,\s*[A-Z])*)\s*\.?\s*$", re.I | re.M)


def _letters_from_str(s: str) -> Optional[frozenset]:
    s = s.strip().strip("`").strip().strip("\"'").strip()
    s = _ANSWER_PREFIX_RE.sub("", s).strip().strip("\"'").rstrip(".").strip()
    if _LETTER_LIST_RE.fullmatch(s):
        return frozenset(re.findall(r"[A-Z]", s))
    return None


def letters_from_any(x) -> Optional[frozenset]:
    """'B' | 'Answer: B, D' | '"B"' | '{"answer": "B"}' | {"Answer": ["A","C"]} -> frozenset({...})
    Returns None unless the WHOLE value is a letter answer (so prose is never mis-read)."""
    if isinstance(x, dict):
        for k, v in x.items():
            if str(k).lower() == "answer":
                return letters_from_any(v)
        return None
    if isinstance(x, (list, tuple, set, frozenset)):
        out: set = set()
        for item in x:
            part = letters_from_any(item)
            if part is None:
                return None
            out |= part
        return frozenset(out) if out else None
    if isinstance(x, str):
        s = x.strip()
        if s and s[0] in "{[":
            try:
                return letters_from_any(json.loads(s))
            except ValueError:
                pass
        return _letters_from_str(s)
    return None


# ----------------------------------------------------------------------------
# 3. Recovering answers from REPL code (static analysis -- the code is NEVER executed)
# ----------------------------------------------------------------------------
class _Unresolved:
    def __repr__(self):
        return "UNRESOLVED"


UNRESOLVED = _Unresolved()
_FENCE_RE = re.compile(r"```[A-Za-z]*[ \t]*\n(.*?)(?:```|\Z)", re.S)


def _dotted(n) -> str:
    if isinstance(n, ast.Name):
        return n.id
    if isinstance(n, ast.Attribute):
        return f"{_dotted(n.value)}.{n.attr}"
    return ""


def _names_in(node) -> set:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _literal_env(tree: ast.AST) -> dict:
    """name -> str, only for variables whose every assignment is the same string literal."""
    values, tainted = defaultdict(list), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    values[t.id].append(node.value)
                else:
                    tainted |= _names_in(t)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            values[node.target.id].append(node.value)
        elif isinstance(node, (ast.AugAssign, ast.For)):
            tainted |= _names_in(node.target)
    env = {}
    for name, nodes in values.items():
        if name in tainted:
            continue
        lits = {n.value for n in nodes if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        if len(lits) == 1 and all(isinstance(n, ast.Constant) for n in nodes):
            env[name] = next(iter(lits))
    return env


def _eval(node, env):
    """Tiny safe evaluator: literals, known variables, 'a' + 'b', f-strings, dict/list, json.dumps(x)."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id, UNRESOLVED)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a, b = _eval(node.left, env), _eval(node.right, env)
        return a + b if isinstance(a, str) and isinstance(b, str) else UNRESOLVED
    if isinstance(node, ast.JoinedStr):
        parts = []
        for v in node.values:
            r = v.value if isinstance(v, ast.Constant) else (
                _eval(v.value, env) if isinstance(v, ast.FormattedValue) else UNRESOLVED)
            if not isinstance(r, str):
                return UNRESOLVED
            parts.append(r)
        return "".join(parts)
    if isinstance(node, ast.Dict):
        out = {}
        for k, v in zip(node.keys, node.values):
            key = _eval(k, env) if k is not None else UNRESOLVED
            if isinstance(key, str):
                out[key] = _eval(v, env)
        return out
    if isinstance(node, (ast.List, ast.Tuple)):
        items = [_eval(e, env) for e in node.elts]
        return UNRESOLVED if any(i is UNRESOLVED for i in items) else items
    if isinstance(node, ast.Call) and _dotted(node.func) in {"json.dumps", "dumps"} and node.args:
        return _eval(node.args[0], env)
    return UNRESOLVED


def _is_answer_content(target) -> bool:
    if not (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
            and target.value.id == "answer"):
        return False
    s = target.slice
    if s.__class__.__name__ == "Index":  # Python < 3.9
        s = s.value
    return isinstance(s, ast.Constant) and s.value == "content"


# regex fallback, used only if a code block does not parse (e.g. truncated output)
_FALLBACK_RES = [
    re.compile(r"""answer\[['"]content['"]\]\s*=\s*json\.dumps\(\s*\{\s*['"]answer['"]\s*:\s*['"]([^'"]*)['"]"""),
    re.compile(r"""answer\[['"]content['"]\]\s*=\s*['"]([^'"]*)['"]"""),
]


def _answer_assignments(code: str) -> Iterator[object]:
    """Yield the (statically resolved) value of every `answer["content"] = ...` in the code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        for rx in _FALLBACK_RES:
            for m in rx.finditer(code):
                yield m.group(1)
        return
    env = _literal_env(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(_is_answer_content(t) for t in node.targets):
            yield _eval(node.value, env)


# ----------------------------------------------------------------------------
# 4. Parsing one raw_result cell
# ----------------------------------------------------------------------------
@dataclass
class Parsed:
    letters: Optional[frozenset]
    status: str   # see STATUS_DOC
    detail: str = ""


STATUS_DOC = {
    "final":           "a real final answer: JSON / 'Answer: X' / bare letters / trailing 'Answer:' line",
    "repl_recovered":  "no final answer, but the REPL code hard-codes it in answer['content'] (all branches agree)",
    "repl_unresolved": "REPL code assigns answer['content'] from a runtime value (e.g. an llm_query result) or branches disagree",
    "repl_no_answer":  "REPL code / prose only (exploration, no answer submitted)",
    "unparsed":        "non-code text with no recognisable answer",
    "empty":           "empty cell",
    "missing":         "no log record for this sample",
}


def parse_raw_result(raw: str) -> Parsed:
    text = (raw or "").strip()
    if not text:
        return Parsed(None, "empty")

    # (a) the whole cell is an answer (JSON, "Answer: B", "B", "\"Answer: A\"", "A, B, C")
    letters = letters_from_any(text)
    if letters:
        return Parsed(letters, "final", "json_or_plain")

    blocks = _FENCE_RE.findall(text)

    # (b) the cell contains REPL code -> look only at answer["content"] = ... assignments
    if blocks:
        cands = [letters_from_any(v) if v is not UNRESOLVED else None
                 for code in blocks for v in _answer_assignments(code)]
        if not cands:
            return Parsed(None, "repl_no_answer")
        if any(c is None for c in cands):
            return Parsed(None, "repl_unresolved")
        uniq = set(cands)
        if len(uniq) == 1:
            return Parsed(next(iter(uniq)), "repl_recovered", "answer['content'] literal")
        return Parsed(None, "repl_unresolved", "branches disagree")

    # (c) free text ending with an "Answer: X" line (only when there is no code in the cell)
    m = _ANSWER_LINE_RE.findall(text)
    if m:
        return Parsed(frozenset(re.findall(r"[A-Z]", m[-1].upper())), "final", "answer_line")

    return Parsed(None, "unparsed")


# ----------------------------------------------------------------------------
# 5. Gold letters + merge + scoring
# ----------------------------------------------------------------------------
def add_gold_letters(sample: pd.DataFrame, question_answer_map_unique: dict, gold_to_letters) -> pd.DataFrame:
    """Adds sample['gold_letters'] in place (same logic as your snippet)."""
    sample["gold_letters"] = [
        gold_to_letters(ans, question_answer_map_unique[q])
        for ans, q in zip(sample["answer"], sample["question"])
    ]
    return sample


def _gold_set(g) -> frozenset:
    s = letters_from_any(g)
    if s is None:
        raise ValueError(f"Cannot interpret gold_letters value: {g!r}")
    return s


def build_results(sample: pd.DataFrame, question_answer_map_unique: dict, log_path: str,
                  methods: Optional[dict] = None) -> pd.DataFrame:
    """One row per sample row, with gold + (pred, status, correctness) for each method.

    methods: {short_name: method string in the log}
    Log row_id 'contract_<n>_<pos>' is matched to sample.iloc[pos].
    """
    methods = methods or {"vanilla": "vanilla", "rlm": "RLM-customized"}
    log = read_log(log_path)
    if log["pos"].max() >= len(sample):
        raise ValueError(f"Log refers to position {log['pos'].max()} but sample has only {len(sample)} rows")

    res = sample.reset_index(drop=True).copy()
    res.index.name = "pos"
    gold = res["gold_letters"].map(_gold_set)
    n_opts = res["question"].map(lambda q: len(question_answer_map_unique[q]))
    res["gold"] = gold.map(lambda s: ",".join(sorted(s)))
    res["multi_gold"] = gold.map(len) > 1

    for short, method_name in methods.items():
        sub = log[log["method"] == method_name]
        if sub.empty:
            raise ValueError(f"No log records for method {method_name!r}; found {sorted(log['method'].unique())}")
        sub = sub.set_index("pos")
        recs = []
        for pos in res.index:
            if pos in sub.index:
                raw = sub.at[pos, "raw_result"]
                p = parse_raw_result(raw)
            else:
                raw, p = "", Parsed(None, "missing")
            pred = p.letters
            exact = bool(pred) and pred == gold[pos]
            recs.append({
                f"{short}_pred": ",".join(sorted(pred)) if pred else "",
                f"{short}_status": p.status,
                f"{short}_correct_strict": exact and p.status == "final",
                f"{short}_correct_lenient": exact,
                f"{short}_jaccard": len(pred & gold[pos]) / len(pred | gold[pos]) if pred else 0.0,
                # sanity check: letters beyond the number of options => wrong row mapping or a bad parse
                f"{short}_out_of_range": bool(pred) and any(ord(c) - 65 >= n_opts[pos] for c in pred),
                f"{short}_raw": raw,
            })
        res = res.join(pd.DataFrame(recs, index=res.index))
    return res


def summarize(res: pd.DataFrame, methods=("vanilla", "rlm")) -> pd.DataFrame:
    """Accuracy per method, overall and split by single-/multi-letter gold.

    strict  : only real final answers count as correct
    lenient : strict + answers recovered from hard-coded REPL code
    Un-answered rows stay in the denominator (counted wrong). acc_answered_only is shown
    separately because it flatters a method that skips its hard questions.
    """
    rows = []
    for subset, mask in [("all", res.index == res.index),
                         ("gold single-letter", ~res["multi_gold"]),
                         ("gold multi-letter", res["multi_gold"])]:
        sub = res[mask]
        for m in methods:
            answered = sub[f"{m}_pred"].ne("")
            rows.append({
                "subset": subset, "method": m, "n": len(sub),
                "n_final": int((sub[f"{m}_status"] == "final").sum()),
                "n_recovered": int((sub[f"{m}_status"] == "repl_recovered").sum()),
                "n_no_answer": int((~answered).sum()),
                "acc_strict": sub[f"{m}_correct_strict"].mean(),
                "acc_lenient": sub[f"{m}_correct_lenient"].mean(),
                "acc_answered_only": sub.loc[answered, f"{m}_correct_lenient"].mean() if answered.any() else float("nan"),
                "mean_jaccard": sub[f"{m}_jaccard"].mean(),
            })
    return pd.DataFrame(rows).set_index(["subset", "method"]).round(3)


def paired_table(res: pd.DataFrame, policy: str = "lenient", a: str = "vanilla", b: str = "rlm") -> pd.DataFrame:
    """2x2 table of per-sample agreement between two methods (policy: 'strict' | 'lenient')."""
    ca, cb = res[f"{a}_correct_{policy}"], res[f"{b}_correct_{policy}"]
    t = pd.crosstab(ca.map({True: f"{a} correct", False: f"{a} wrong"}),
                    cb.map({True: f"{b} correct", False: f"{b} wrong"}), margins=True)
    return t


# ----------------------------------------------------------------------------
# CLI: audit the log alone
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "maud/maud_152_different_prompts_per_tier.csv"
    log = read_log(path)
    log["status"] = [parse_raw_result(r).status for r in log["raw_result"]]

    sample = get_sample_rows_for_maud_152(split="train")
    with open("maud/question_answer_map_unique.json", "r", encoding="utf-8") as f:
        question_answer_map_unique = json.load(f)
    res = build_results(sample, question_answer_map_unique, "maud/maud_152_different_prompts_per_tier.csv")
    summary = summarize(res)
    print(summary)
    summary.to_csv("summary_152_maud.csv", index=False)                    # strict / lenient / answered-only, split by single vs multi-letter gold
    print(paired_table(res, "lenient"))      # vanilla vs RLM, sample by sample

    print(pd.crosstab(log["status"], log["method"], margins=True))
    print()
    for k, v in STATUS_DOC.items():
        print(f"  {k:16s} {v}")