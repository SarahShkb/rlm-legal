"""
tool2_definitions.py — Tool 2: defined-term resolver for RLM
============================================================

Extracts every defined term of a contract once, with its defining text and
char offsets in `context`, and exposes:

    define(term)            definition of a capitalized term, with its source;
                            follows "has the meaning set forth in Section X";
                            resolves acronyms ("MAE") and party-prefix variants
                            ("Material Adverse Effect" -> "Company Material Adverse
                            Effect"); otherwise suggests the closest defined terms
    term_usage(term, k=8)   where a term is used outside its definition
    list_definitions(contains="", start=0)
                            names of all defined terms (filterable, paged)
    terms_in(ref)           defined terms used inside a section (needs tool 1's index)

Why (thesis framing)
--------------------
Many MAUD answers turn on carve-outs buried in definitions ("Company Material
Adverse Effect", "Superior Proposal", "Intervening Event"). This makes
ConReader's term–definition relation (TDR) explicit and callable.
ConReader encodes all relations for every input inside its encoder; here they
are extracted once (with patterns, as ConReader also does) and the RLM decides
at question time which terms to look up.
Compared with LexNLP's get_definitions(), which returns only term names,
define() returns the defining text plus its offsets, so the model can verify it
and the run log records exactly what was retrieved.

Needs tools_util.py next to it. Works without tool 1. If you also pass the SectionIndex from tool 1
(tool1_section_locator.py), definitions are labelled with their section,
"has the meaning set forth in Section X" is followed into that section, and
terms_in() is available. Without it, by-reference definitions are resolved by
searching the whole contract for the quoted term.

Offline check on all 152 MAUD contracts: median 166 defined terms per contract
(minimum 50); define("MAE") finds the MAE/MAC definition in 149/152 contracts;
definition boundaries with and without tool 1 agree for 92% of definitions.

Usage in the RLM runner (per contract):
    from tools_util import load_maud_contract, merge_custom_tools, ToolLog
    from tool1_section_locator import SectionIndex, rlm_custom_tools as tool1
    from tool2_definitions import rlm_custom_tools as tool2
    text = load_maud_contract("~/maud", 34)
    log = ToolLog()
    t2_only  = tool2(text, log)                                     # tool 2 alone
    sec      = SectionIndex(text)                                   # or with tool 1:
    t1_and_2 = merge_custom_tools(tool1(text, log, index=sec), tool2(text, log, sections=sec))
    rlm = RLM(..., custom_tools=t1_and_2)
"""

from __future__ import annotations

import difflib
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable

from legal_tools.tools_utils import (
    DEMO_CONTRACT,
    Q_CLOSE,
    Q_OPEN,
    SECTION_NUM,
    ToolLog,
    as_rlm_custom_tools,
    context_offset,
    load_maud_contract,
    maud_contract_ids,
    norm_term,
    wrap_tools,
)

# ----------------------------------------------------------------------------
# Data structure
# ----------------------------------------------------------------------------


@dataclass
class Definition:
    term: str
    text: str
    start: int
    end: int
    sid: str            # section where it is defined ("" without tool 1)
    kind: str           # "formal" | "inline" | "by_reference"
    ref: str = ""       # for by_reference: the section it points to, e.g. "6.3(d)"


# ----------------------------------------------------------------------------
# The index
# ----------------------------------------------------------------------------

class DefinitionIndex:
    """Parse definitions once per contract; the tool functions read from it.

    `sections` is optional: any object with section_at(pos) -> str,
    lookup(ref) -> (section | None, subref) and by_sid (tool 1's SectionIndex)."""

    # "Term" means ... / "Term" shall have the meaning ... / "Term" has the meaning set forth in Section X
    # (the opening quote may be a mis-typed closing quote in EDGAR text; a short qualifier
    #  such as "with respect to a Person" or "as used with respect to a party," may sit
    #  between the term and "means")
    _DEF_FORMAL = re.compile(
        r"[\"\u201c\u201d]\s?([A-Z][^\"\u201c\u201d\n]{1,90}?)" + Q_CLOSE +
        r"\s*(?:,\s*)?(?:(?:(?:as |when )?used\b|with respect to|as to|in respect of|or similar|"
        r"of (?:the )?(?:Company|Parent|Buyer|Purchaser|any Person)\b)"
        r"[^\"\u201c\u201d.;]{0,60}?,?\s+)?"
        r"(?:shall\s+)?(?:means?|mean,|has the meaning|have the meaning|"
        r"shall have the meaning|is defined|refers to|includes?)")
    # parenthetical definitions: (the "Company"), (each, a "Share" and collectively, the "Shares")
    _PAREN_Q = re.compile(r"\([^()]{0,300}?" + Q_OPEN + r"[^()]{0,300}?\)")
    _QTERM = re.compile(Q_OPEN + r"([A-Z][^\"”\n]{1,90}?)" + Q_CLOSE)
    # an ARTICLE / Section heading (used to end a definition when tool 1 is not available)
    _HEADING = re.compile(r"(?:\n[ \t]*|\s{2,})(?:ARTICLE\s+[IVXLC\d]+\b|"
                          r"(?:Section|SECTION|\u00a7)\s*" + SECTION_NUM + r"\.?\s+[A-Z])")
    _DEF_BYREF = re.compile(r"(?:Section|Sections)\s+(" + SECTION_NUM + r"(?:\([a-z0-9]+\))*)")

    def __init__(self, text: str, sections=None, max_def_chars: int = 12000, base: int | None = None):
        """base: where the contract starts inside the RLM `context` (default: the
        section index's base, else 0). Reported char offsets are shifted by it."""
        self.text = text
        self.sections = sections
        self.base = base if base is not None else getattr(sections, "base", 0)
        self.defs: dict[str, list[Definition]] = defaultdict(list)
        t0 = time.time()
        self._parse(max_def_chars)
        self.build_seconds = round(time.time() - t0, 3)

    def _sid(self, pos: int) -> str:
        return self.sections.section_at(pos) if self.sections is not None else ""

    def _sec_end(self, sid: str) -> int:
        if self.sections is None or sid not in getattr(self.sections, "by_sid", {}):
            return len(self.text)
        return self.sections.by_sid[sid].end

    def _parse(self, max_def_chars: int) -> None:
        text = self.text
        formal = list(self._DEF_FORMAL.finditer(text))
        for i, m in enumerate(formal):
            start = m.start()
            nxt = formal[i + 1].start() if i + 1 < len(formal) else len(text)
            sid = self._sid(start)
            end = min(nxt, start + max_def_chars, self._sec_end(sid))
            if self.sections is None:            # no section map: stop at the next heading
                h = self._HEADING.search(text, m.end(), end)
                if h:
                    end = h.start()
            body = text[start:end].strip()
            kind, ref = "formal", ""
            head = body[:250]
            if re.search(r"ha(?:s|ve) the meaning|is defined", head):
                r = self._DEF_BYREF.search(head)
                if r:
                    kind, ref = "by_reference", r.group(1)
            self._add(Definition(m.group(1).strip(), body, start, end, sid, kind, ref))

        for m in self._PAREN_Q.finditer(text):
            # the defining sentence: back to the previous sentence end, forward to the paren
            s = max(text.rfind(". ", 0, m.start()), text.rfind("\n", 0, m.start()), m.start() - 600)
            s = s + 1 if s > 0 else 0
            e = min(len(text), m.end() + 1)
            for q in self._QTERM.finditer(m.group(0)):
                term = q.group(1).strip()
                if len(term) >= 2:
                    self._add(Definition(term, text[s:e].strip(), s, e, self._sid(m.start()), "inline"))

    def _add(self, d: Definition) -> None:
        k = norm_term(d.term)
        if not any(abs(x.start - d.start) < 5 for x in self.defs[k]):
            self.defs[k].append(d)

    # party prefixes that vary between contracts: "Company Material Adverse Effect" vs "Material Adverse Effect"
    _PARTY = re.compile(r"^(?:the )?(?:company|parent|target|buyer|purchaser|acquiror|acquirer|seller)"
                        r"(?:['’]s)?\s+")
    _PREFERRED = re.compile(r"^(?:the )?(?:company|target)\b")

    def exact_keys(self, term: str) -> tuple[str, ...]:
        """Spellings of `term` that count as an exact match (plural / leading 'the')."""
        k = norm_term(term)
        return (k, k.rstrip("s"), k + "s", re.sub(r"^the ", "", k))

    def match(self, term: str) -> tuple[str | None, list[str]]:
        """Resolve a user/model-typed term to an index key.
        Returns (key or None, other candidate term names)."""
        k = norm_term(term)
        for cand in self.exact_keys(term):
            if cand in self.defs:
                return cand, []
        # same term with another party prefix (or none): prefer the Company / Target version
        core = self._PARTY.sub("", k)
        near = [t for t in self.defs if self._PARTY.sub("", t) == core
                and (t == core or self._PREFERRED.match(t))]
        if near:
            near.sort(key=lambda t: (not self._PREFERRED.match(t), len(t)))
            others = [t for t in self.defs if self._PARTY.sub("", t) == core and t != near[0]]
            return near[0], [self.defs[t][0].term for t in others[:4]]
        # acronyms: MAE -> (Company) Material Adverse Effect
        if k.isalpha() and term.strip().isupper() and 2 <= len(k) <= 6:
            def initials(t):
                return "".join(w[0] for w in t.split() if w not in {"of", "and", "the", "to", "for"})
            alts = {k} | ({"mae", "mac"} if k in {"mae", "mac"} else set())   # Effect / Change
            acr = [t for t in self.defs if any(initials(t).endswith(a) for a in alts)]
            if acr:
                acr.sort(key=lambda t: (not t.startswith("company"), len(t)))
                return acr[0], [self.defs[t][0].term for t in acr[1:4]]
        # "Superior Proposal" -> "Company Superior Proposal" if unique
        ends = [t for t in self.defs if t.endswith(" " + k)]
        if len(ends) == 1:
            return ends[0], []
        return None, [self.defs[s][0].term for s in self.suggest(k)]

    def suggest(self, k: str, n: int = 6) -> list[str]:
        """Closest index keys to a normalized term: substring matches first, then terms
        sharing content words, then near-identical spellings."""
        def words(t: str) -> set[str]:
            return {w for w in re.findall(r"[a-z0-9]+", t) if len(w) > 2 and w not in {"the", "and", "for"}}

        kw = words(k)

        def score(t: str) -> tuple[float, float]:
            tw = words(t)
            return (len(kw & tw) / len(kw | tw) if kw | tw else 0.0,
                    difflib.SequenceMatcher(None, k, t).ratio())

        subs = sorted((t for t in self.defs if len(t) > 3 and (k in t or t in k)), key=len)
        ranked = [t for t in sorted(self.defs, key=score, reverse=True)
                  if score(t)[0] >= 0.3 or score(t)[1] >= 0.8]
        return list(dict.fromkeys(subs + ranked))[:n]

    def names(self) -> list[str]:
        """Every defined term, as written in the contract, sorted."""
        return sorted((ds[0].term for ds in self.defs.values()), key=str.lower)

    def stats(self) -> dict:
        kinds = Counter(d.kind for ds in self.defs.values() for d in ds)
        return {"defined_terms": len(self.defs), **dict(kinds),
                "with_sections": self.sections is not None, "build_seconds": self.build_seconds}


# ----------------------------------------------------------------------------
# The tools (what the RLM calls)
# ----------------------------------------------------------------------------

def make_tools(index: DefinitionIndex, log: ToolLog | None = None, *,
               max_chars: int = 6000, def_chars: int = 5000) -> dict[str, Callable]:
    """Return {name: function}. Every function returns a str and never raises."""
    T = index.text
    S = index.sections
    B = index.base       # offsets reported to the model are offsets in `context`

    def _sentence_with_quoted(body: str, term: str) -> str | None:
        qm = re.search(Q_OPEN + re.escape(term) + Q_CLOSE, body)
        if not qm:
            return None
        a = max(body.rfind(". ", 0, qm.start()), body.rfind("\n", 0, qm.start()), 0)
        e = re.search(r"\.(?:\s|$)", body[qm.end():])
        b = qm.end() + (e.end() if e else 1200)
        return body[a:min(b, a + 2000)].strip()

    def define(term: str) -> str:
        """Definition of a defined term, with the section and chars it comes from.
        Follows 'has the meaning set forth in Section X' automatically."""
        k, sugg = index.match(term)
        if k is None:
            return (f"No definition found for '{term}'."
                    + (" Did you mean: " + ", ".join(f"'{s}'" for s in sugg) + "?" if sugg else "")
                    + " Call list_definitions() to see every defined term.")
        notes, out = [], []
        if k not in index.exact_keys(term):
            notes.append(f"'{term}' is not itself a defined term here; showing the closest: "
                         f"'{index.defs[k][0].term}'.")
        if sugg:
            notes.append("Other matching terms: " + ", ".join(sugg))
        ds = index.defs[k]
        if any(d.kind != "inline" for d in ds):      # quoted mentions in parentheses are noise then
            ds = [d for d in ds if d.kind != "inline"]
        for d in sorted(ds, key=lambda d: (d.kind != "formal", d.start))[:3]:
            text = d.text
            if d.kind == "by_reference" and d.ref:
                found = None
                if S is not None:
                    sec, _ = S.lookup(d.ref)
                    if sec is not None:
                        found = _sentence_with_quoted(T[sec.start:sec.end], d.term)
                        if found:
                            text += f"\n  -> in Section {sec.sid}: {found}"
                if found is None and not any(x.kind == "formal" for x in index.defs[k]):
                    # no section map: look for the quoted term anywhere else in the contract
                    for qm in re.finditer(Q_OPEN + re.escape(d.term) + Q_CLOSE, T):
                        if abs(qm.start() - d.start) > 5:
                            win = T[max(0, qm.start() - 800):qm.end() + 1500]
                            found = _sentence_with_quoted(win, d.term)
                            if found:
                                text += f"\n  -> at char {B + qm.start()}: {found}"
                                break
            if len(text) > def_chars:     # long definitions (MAE carve-outs): continue via context slice
                text = text[:def_chars] + f"\n...[definition continues; read context[{B + d.start + def_chars}:{B + d.end}]]"
            where = f"Section {d.sid}, " if d.sid else ""
            out.append(f"[{d.kind}, {where}chars {B + d.start}-{B + d.end}]\n" + text)
        return f'"{index.defs[k][0].term}":\n' + "".join(n + "\n" for n in notes) + "\n---\n".join(out)

    def term_usage(term: str, k: int = 8) -> str:
        """Where a term is used outside its own definition (by section if available)."""
        key, _ = index.match(term)
        name = index.defs[key][0].term if key else term
        def_spans = [(d.start, d.end) for d in index.defs.get(key, [])] if key else []
        hits = [m.start() for m in re.finditer(re.escape(name), T)
                if not any(a <= m.start() < b for a, b in def_spans)]
        if not hits:
            return f"'{name}' is not used outside its definition."
        if S is not None:
            c = Counter(S.section_at(p) for p in hits)
            rows = [f"  {sid} {S.by_sid[sid].title if sid in S.by_sid else ''} — {n}x"
                    for sid, n in c.most_common(k)]
            return f"'{name}' used {len(hits)}x in {len(c)} sections; top {min(k, len(c))}:\n" + "\n".join(rows)
        rows = [f"  char {B + p}: ...{' '.join(T[max(0, p - 80):p + 120].split())}..." for p in hits[:k]]
        return f"'{name}' used {len(hits)}x; first {min(k, len(hits))}:\n" + "\n".join(rows)

    def terms_in(ref: str) -> str:
        """Defined terms used inside one section — candidates to define() before deciding."""
        if S is None:
            return "terms_in needs the section index from tool 1 (inject(..., sections=...))."
        sec, _ = S.lookup(ref)
        if sec is None:
            return f"'{ref}' not found."
        body = T[sec.start:sec.end]
        terms = Counter()
        for ds in index.defs.values():
            name = ds[0].term
            if len(name) > 3 and name in body and not any(sec.start <= d.start < sec.end for d in ds):
                terms[name] = body.count(name)
        tl = ", ".join(f"{t} ({n}x)" for t, n in terms.most_common(25)) or "none"
        return f"Section {sec.sid} — {sec.title}\nDefined terms used: {tl}"

    def list_definitions(contains: str = "", start: int = 0) -> str:
        """Names of the defined terms in this contract (optionally only those containing
        `contains`), so define() is asked for terms that exist. Long lists are paged."""
        names = [n for n in index.names() if contains.lower() in n.lower()]
        if not names:
            return f"No defined term contains '{contains}'."
        shown, size = [], 0
        for n in names[start:]:
            if size + len(n) + 2 > max_chars - 200:
                break
            shown.append(n)
            size += len(n) + 2
        what = f" containing '{contains}'" if contains else ""
        nxt = start + len(shown)
        tail = (f"\n...[{len(names) - nxt} more: list_definitions({contains!r}, start={nxt})]"
                if nxt < len(names) else "")
        return f"{len(names)} defined terms{what}:\n" + "; ".join(shown) + tail

    tools = {"define": define, "term_usage": term_usage, "list_definitions": list_definitions}
    if S is not None:
        tools["terms_in"] = terms_in
    return wrap_tools(tools, log, max_chars)


# ----------------------------------------------------------------------------
# RLM adapter: the `custom_tools` dict expected by rlm.RLM(...)
# ----------------------------------------------------------------------------

TOOL_DESCRIPTIONS = {
    "define": "define(term: str) -> str. Definition of a capitalized defined term in the contract "
              "(e.g. 'Superior Proposal', 'Company Material Adverse Effect', or acronyms like 'MAE'), "
              "with where it is defined. Follows 'has the meaning set forth in Section X'. "
              "If the exact term is not defined it returns the closest one, or suggestions.",
    "term_usage": "term_usage(term: str, k: int = 8) -> str. Where a defined term is used in the "
                  "contract outside its own definition.",
    "list_definitions": "list_definitions(contains: str = '', start: int = 0) -> str. Names of all "
                        "defined terms in the contract (filter with e.g. contains='Adverse'), so you "
                        "define() terms that exist.",
    "terms_in": "terms_in(ref: str) -> str. Defined terms used inside a section (e.g. '6.3'): "
                "candidates to define() before choosing an answer.",
}


def rlm_custom_tools(contract_text: str, log: ToolLog | None = None, *,
                     context: str | None = None, sections=None,
                     max_chars: int = 6000, list_in_prompt: bool = False) -> dict:
    """Tool 2 for ONE contract, in the format of rlm.RLM(custom_tools=...).
    sections:       tool 1's SectionIndex for the same contract (optional; enables section
                    labels, by-reference following and terms_in). Its base offset is reused.
    context:        the RLM context if it is not exactly the contract (offsets get shifted).
    list_in_prompt: also put the defined-term names into define()'s description, i.e. into
                    the system prompt (~4-8k extra chars), so the model sees them up front."""
    base = None if sections is not None else context_offset(contract_text, context)
    idx = DefinitionIndex(contract_text, sections, base=base)
    descriptions = dict(TOOL_DESCRIPTIONS)
    if list_in_prompt:
        descriptions["define"] += " Defined terms in this contract: " + "; ".join(idx.names())
    return as_rlm_custom_tools(make_tools(idx, log, max_chars=max_chars), descriptions)


TOOL_PROMPT = """\
In the REPL you also have a definitions tool. It returns strings; print them.
  list_definitions("Adverse")      -> names of the defined terms (all of them if no filter)
  define("Superior Proposal")      -> the definition of a capitalized term and where it is;
                                      also accepts acronyms like "MAE"
  term_usage("Intervening Event")  -> where a term is used in the agreement
  terms_in("6.3")                  -> defined terms used inside a section (if available)
Before choosing between answer options, define() the capitalized terms that decide them.
"""

TOOL_PROMPT_NO_SECTIONS = TOOL_PROMPT.replace(
    '  terms_in("6.3")                  -> defined terms used inside a section (if available)\n', "")


def inject(namespace: dict, context_text: str, log: ToolLog | None = None,
           sections=None, **kw) -> DefinitionIndex:
    """Build the definition index for one contract and add tool 2 to a REPL namespace
    (the dict where `context` and `llm_query` live)."""
    idx = DefinitionIndex(context_text, sections)
    namespace.update(make_tools(idx, log, **kw))
    return idx


# ----------------------------------------------------------------------------
# Self-check:
#   python tool2_definitions.py                         (demo contract)
#   python tool2_definitions.py --maud ~/maud --contract 34 --term MAE
#   python tool2_definitions.py --maud ~/maud --all     (stats for every contract)
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Self-check of tool 2 (definitions).")
    ap.add_argument("--maud", help="path to your cloned MAUD repo (or its data/ folder)")
    ap.add_argument("--contract", help="contract number, e.g. 34 or contract_34")
    ap.add_argument("--term", default="Superior Proposal", help="term to define()")
    ap.add_argument("--all", action="store_true", help="index every MAUD contract and print stats")
    ap.add_argument("--no-sections", action="store_true", help="run tool 2 without tool 1's index")
    ap.add_argument("file", nargs="?", help="or: any contract .txt file")
    a = ap.parse_args()

    def _sections(text):
        if a.no_sections:
            return None
        try:
            from tool1_section_locator import SectionIndex
        except ImportError:
            print("(tool1_section_locator.py not found; running without sections)")
            return None
        return SectionIndex(text)

    if a.maud and a.all:
        for cid in maud_contract_ids(a.maud):
            t = load_maud_contract(a.maud, cid)
            idx = DefinitionIndex(t, _sections(t))
            mae = make_tools(idx)["define"]("MAE")
            print(f"contract_{cid:<4} terms={len(idx.defs):<4} MAE={'yes' if 'No definition' not in mae else 'NO'}")
        raise SystemExit

    if a.maud:
        text = load_maud_contract(a.maud, a.contract if a.contract is not None else 0)
    elif a.file:
        text = open(a.file, encoding="utf-8").read()
    else:
        text = DEMO_CONTRACT
    log, ns = ToolLog(), {}
    idx = inject(ns, text, log, sections=_sections(text))
    print("INDEX:", idx.stats(), "\n")
    print(ns["define"](a.term)[:2500], "\n")
    print(ns["term_usage"](a.term), "\n")
    if "terms_in" in ns:
        k, _ = idx.match(a.term)
        if k and idx.defs[k][0].sid:
            print(ns["terms_in"](idx.defs[k][0].sid)[:1500], "\n")
    print("LOG:", log.summary())