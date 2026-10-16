"""
tools_util.py — shared helpers for the RLM legal tools
======================================================

Everything that more than one tool needs lives here, so each tool file
(tool1_section_locator.py, tool2_definitions.py, future tool3_..., tool4_...)
only contains its own logic.

Contents
--------
Text helpers      QUOTE_CHARS, Q_OPEN, Q_CLOSE, norm_term, roman_to_int, num_key,
                  STOPWORDS, tokens, SECTION_NUM, SECTION_REF
Tool plumbing     ToolLog, clip, wrap_tool, wrap_tools
RLM adapter       context_offset, as_rlm_custom_tools, merge_custom_tools
MAUD data         contracts_source, maud_contract_ids, load_maud_contract
Testing           DEMO_CONTRACT

Writing a new tool (pattern used by tools 1 and 2):
    from tools_util import ToolLog, wrap_tools, as_rlm_custom_tools, context_offset

    def make_tools(index, log=None, *, max_chars=6000):
        def my_tool(arg: str) -> str: ...          # always return a str
        return wrap_tools({"my_tool": my_tool}, log, max_chars)

    def rlm_custom_tools(contract_text, log=None, *, context=None):
        idx = MyIndex(contract_text, base=context_offset(contract_text, context))
        return as_rlm_custom_tools(make_tools(idx, log), TOOL_DESCRIPTIONS)

Standard library only.
"""

from __future__ import annotations

import json
import os
import re
import time
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

# ----------------------------------------------------------------------------
# Text helpers
# ----------------------------------------------------------------------------

# Quote characters seen in EDGAR contract text (straight and curly).
QUOTE_CHARS = "\"“”‘’'"
Q_OPEN = r"[\"“]"          # opening double quote, straight or curly
Q_CLOSE = r"[\"”]"         # closing double quote, straight or curly

# A section number such as 7.1, 5.02, 10.12
SECTION_NUM = r"\d{1,2}\.\d{1,2}"
# A reference such as "Section 7.1(b)(ii)", "Sections 5.2", "Article VIII"
SECTION_REF = re.compile(
    r"\b(Section|Sections|Article)\s+((?:" + SECTION_NUM + r"(?:\([a-zA-Z0-9]+\))*)|[IVXLC]+)\b")

_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def roman_to_int(s: str) -> int:
    """'VIII' -> 8; digits are returned as int ('8' -> 8)."""
    s = s.upper()
    if s.isdigit():
        return int(s)
    total, prev = 0, 0
    for ch in reversed(s):
        v = _ROMAN.get(ch, 0)
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total


def num_key(num: str) -> tuple:
    """Section number -> sortable key: '5.02' -> (5, 2); '10.1' -> (10, 1)."""
    return tuple(int(p) for p in num.split(".") if p.isdigit())


def norm_term(t: str) -> str:
    """Normalize a defined term for lookup: strip quotes, lowercase, collapse spaces."""
    t = t.strip(QUOTE_CHARS + " ").lower()
    return re.sub(r"\s+", " ", t)


STOPWORDS = frozenset(["a", "an", "the", "of", "and", "or", "to", "in", "on", "for", "by", "with", "as", "at", "be", "is", "are", "was", "were", "this", "that", "these", "those", "any", "all", "such", "each", "other", "its", "it", "from", "into", "under", "upon", "shall", "will", "may", "not", "no", "if", "than", "then", "which", "who", "whom", "whose", "hereof", "herein", "hereto", "hereunder", "thereof", "therein", "thereto", "party", "parties", "agreement", "section", "article"])

_TOK = re.compile(r"[a-z0-9]+")


def tokens(s: str) -> list[str]:
    """Lowercased word tokens without stopwords / 1-char tokens (for BM25 etc.)."""
    return [t for t in _TOK.findall(s.lower()) if t not in STOPWORDS and len(t) > 1]


# ----------------------------------------------------------------------------
# Tool plumbing: logging, output limits, error safety
# ----------------------------------------------------------------------------

@dataclass
class ToolLog:
    """One log per question; share it across all tools of a run."""
    calls: list[dict] = field(default_factory=list)

    def record(self, name: str, args: dict, out: str, secs: float) -> None:
        self.calls.append({"tool": name, "args": args, "out_chars": len(out), "secs": round(secs, 4)})

    def summary(self) -> dict:
        c = Counter(x["tool"] for x in self.calls)
        return {"n_tool_calls": len(self.calls), "by_tool": dict(c),
                "chars_returned": sum(x["out_chars"] for x in self.calls)}

    def to_json(self) -> str:
        return json.dumps({"summary": self.summary(), "calls": self.calls}, ensure_ascii=False)

    def reset(self) -> None:
        self.calls.clear()


def clip(s: str, limit: int) -> str:
    """Cut long tool output and say how much was left out."""
    return s if len(s) <= limit else s[:limit] + f"\n...[truncated; {len(s) - limit} more chars]"


def wrap_tool(name: str, fn: Callable, log: ToolLog | None, max_chars: int = 6000) -> Callable:
    """Wrap a tool function so that every call is logged, output is size-capped,
    and an exception becomes an error string instead of crashing the REPL."""
    def wrapped(*a, **kw):
        t0 = time.time()
        try:
            out = fn(*a, **kw)
            out = out if isinstance(out, str) else str(out)
        except Exception as e:
            out = f"[{name} error: {type(e).__name__}: {e}]"
        out = clip(out, max_chars + 600)
        if log is not None:
            log.record(name, {"args": [str(x)[:120] for x in a],
                              **{k: str(v)[:120] for k, v in kw.items()}},
                       out, time.time() - t0)
        return out
    wrapped.__name__, wrapped.__doc__ = name, fn.__doc__
    return wrapped


def wrap_tools(fns: dict[str, Callable], log: ToolLog | None, max_chars: int = 6000) -> dict[str, Callable]:
    """wrap_tool for a whole {name: function} dict."""
    return {n: wrap_tool(n, f, log, max_chars) for n, f in fns.items()}


# ----------------------------------------------------------------------------
# RLM adapter
# ----------------------------------------------------------------------------

def context_offset(contract_text: str, context: str | None) -> int:
    """Where the contract starts inside the RLM `context` (0 if context IS the contract).
    Tools add this to every char offset they report, so offsets point into `context`."""
    if context is None or context is contract_text or context == contract_text:
        return 0
    base = context.find(contract_text[:2000])
    if base < 0:
        raise ValueError("contract_text not found inside context; build the tools from the same text")
    return base


def as_rlm_custom_tools(fns: dict[str, Callable], descriptions: dict[str, str]) -> dict:
    """{name: fn} -> the format of rlm.RLM(custom_tools=...):
    {name: {"tool": fn, "description": "..."}}. The descriptions go into the
    system prompt (via the {custom_tools_section} placeholder)."""
    return {n: {"tool": f, "description": descriptions.get(n, "")} for n, f in fns.items()}


# Names the rlm package reserves in the REPL (custom tools must not reuse them).
RLM_RESERVED_NAMES = frozenset({"llm_query", "llm_query_batched", "rlm_query", "rlm_query_batched",
                                "SHOW_VARS", "answer", "context", "history", "FINAL", "FINAL_VAR"})


def merge_custom_tools(*tool_dicts: dict) -> dict:
    """Combine the custom_tools dicts of several tools into one for RLM(custom_tools=...).
    Raises on duplicate or reserved names, so two tools never silently shadow each other."""
    merged: dict = {}
    for d in tool_dicts:
        for name, entry in d.items():
            if name in merged:
                raise ValueError(f"tool name '{name}' defined twice")
            if name in RLM_RESERVED_NAMES:
                raise ValueError(f"tool name '{name}' is reserved by the RLM REPL")
            merged[name] = entry
    return merged


# ----------------------------------------------------------------------------
# MAUD data: contract_{i} from your cloned repository
# ----------------------------------------------------------------------------

def contracts_source(maud_dir: str) -> tuple[str, str]:
    """Find the MAUD contracts. Returns ("dir", path) or ("zip", path).
    Accepts the repo root, its data/ folder, or the contracts/ folder itself,
    and works whether or not data.zip has been unzipped."""
    maud_dir = os.path.expanduser(maud_dir)
    for d in (os.path.join(maud_dir, "data", "contracts"), os.path.join(maud_dir, "contracts"), maud_dir):
        if os.path.isfile(os.path.join(d, "contract_0.txt")):
            return "dir", d
    for z in (os.path.join(maud_dir, "data.zip"), os.path.join(maud_dir, "..", "data.zip")):
        if os.path.isfile(z):
            return "zip", z
    raise FileNotFoundError(
        f"No MAUD contracts under {maud_dir}. Expected data/contracts/contract_0.txt or data.zip "
        "(data.zip is stored with git LFS: run `git lfs pull` if it is only a few bytes).")


def maud_contract_ids(maud_dir: str) -> list[int]:
    """All contract numbers available, sorted: [0, 1, ..., 151]."""
    kind, path = contracts_source(maud_dir)
    if kind == "dir":
        names = os.listdir(path)
    else:
        with zipfile.ZipFile(path) as z:
            names = [os.path.basename(n) for n in z.namelist()]
    ids = [int(m.group(1)) for n in names if (m := re.fullmatch(r"contract_(\d+)\.txt", n))]
    return sorted(set(ids))


def load_maud_contract(maud_dir: str, i: int | str) -> str:
    """Text of f"contract_{i}". `i` may be 34, "34", "contract_34" or "contract_34.pdf"
    (as in MAUD_*.csv `contract_name` or raw/main.csv `Filename (anon)`).
    Decoded as UTF-8 exactly like open(path, encoding="utf-8").read() (a leading BOM
    is kept, line endings become \\n). Use the SAME string as the RLM context."""
    m = re.search(r"(\d+)", str(i))
    if not m:
        raise ValueError(f"Cannot read a contract number from {i!r}")
    name = f"contract_{int(m.group(1))}.txt"
    kind, path = contracts_source(maud_dir)
    if kind == "dir":
        with open(os.path.join(path, name), encoding="utf-8") as f:
            return f.read()
    with zipfile.ZipFile(path) as z:
        member = next((n for n in z.namelist() if n.endswith("/" + name) or n == name), None)
        if member is None:
            raise FileNotFoundError(f"{name} not in {path}")
        # same newline handling as text-mode open()
        return z.read(member).decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


# ----------------------------------------------------------------------------
# A tiny merger agreement for self-checks (no data needed)
# ----------------------------------------------------------------------------

DEMO_CONTRACT = """AGREEMENT AND PLAN OF MERGER

This AGREEMENT AND PLAN OF MERGER (this "Agreement") is entered into by Parent Corp. ("Parent") and Target Inc. (the "Company").

ARTICLE I
DEFINITIONS

Section 1.1 Definitions. For purposes of this Agreement:
"Acquisition Proposal" means any proposal or offer from any Person relating to the acquisition of 20% or more of the assets of the Company.
"Company Material Adverse Effect" means any event, change or effect that is materially adverse to the business of the Company, provided that none of the following shall constitute a Company Material Adverse Effect: (i) changes in general economic conditions; (ii) any pandemic, including COVID-19, except to the extent disproportionately affecting the Company; (iii) changes in Law.
"Superior Proposal" has the meaning set forth in Section 6.3(d).
"Intervening Event" means a material event that was not known to the Company Board as of the date hereof.

ARTICLE VI
COVENANTS

Section 6.1 Conduct of Business. Except as set forth herein, the Company shall conduct its business in the ordinary course consistent with past practice.
Section 6.3 No Solicitation.
(a) The Company shall not solicit, initiate or knowingly encourage any Acquisition Proposal.
(b) Notwithstanding Section 6.3(a), the Company may engage in discussions with a Person that makes an unsolicited Acquisition Proposal that the Company Board determines in good faith is or would reasonably be expected to lead to a Superior Proposal.
(c) The Company Board may make an Adverse Recommendation Change in response to an Intervening Event, subject to Section 8.3.
(d) As used herein, "Superior Proposal" means a bona fide written Acquisition Proposal for more than 50% of the Company that the Company Board determines is more favorable to stockholders than the Merger.

ARTICLE VIII
TERMINATION

Section 8.1 Termination. This Agreement may be terminated prior to the Effective Time as provided in this Article VIII.
Section 8.3 Termination Fee. If this Agreement is terminated pursuant to Section 8.1(c), the Company shall pay Parent a fee of $45,000,000 (the "Termination Fee").
"""