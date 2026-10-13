"""
tool2_definitions.py — Tool 2: defined-term resolver for RLM
============================================================

Extracts every defined term of a contract once, with its defining text and
char offsets in `context`, and exposes:

    define(term)            definition of a capitalized term, with its source;
                            follows "has the meaning set forth in Section X";
                            resolves acronyms ("MAE") and near-misses
    term_usage(term, k=8)   where a term is used outside its definition
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

Works on its own. If you also pass the SectionIndex from tool 1
(tool1_section_locator.py), definitions are labelled with their section,
"has the meaning set forth in Section X" is followed into that section, and
terms_in() is available. Without it, by-reference definitions are resolved by
searching the whole contract for the quoted term.

Offline check on all 152 MAUD contracts: median 166 defined terms per contract
(minimum 50); define("MAE") finds the MAE/MAC definition in 149/152 contracts;
definition boundaries with and without tool 1 agree for 92% of definitions.

Usage in the RLM runner (per contract):
    from tool2_definitions import inject, ToolLog, TOOL_PROMPT
    log = ToolLog()
    inject(repl_globals, contract_text, log)                       # tool 2 alone
    # or, with tool 1:  inject(repl_globals, contract_text, log, sections=sec_index)
    system_prompt += "\\n" + TOOL_PROMPT
"""

from __future__ import annotations

import difflib
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable

from tools_utils import SectionIndex, ToolLog, _wrap

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
# Helpers
# ----------------------------------------------------------------------------

_OQ = "\"“”‘’'"          # quote characters seen in EDGAR text
_Q_OPEN = r"[\"“]"
_Q_CLOSE = r"[\"”]"


def _norm_term(t: str) -> str:
    t = t.strip(_OQ + " ").lower()
    return re.sub(r"\s+", " ", t)


# ----------------------------------------------------------------------------
# The index
# ----------------------------------------------------------------------------

class DefinitionIndex:
    """Parse definitions once per contract; the tool functions read from it.

    `sections` is optional: any object with section_at(pos) -> str,
    lookup(ref) -> (section | None, subref) and by_sid (tool 1's SectionIndex)."""

    # "Term" means ... / "Term" shall have the meaning ... / "Term" has the meaning set forth in Section X
    # (the opening quote may be a mis-typed closing quote in EDGAR text; a short qualifier
    #  such as "with respect to a Person" may sit between the term and "means")
    _DEF_FORMAL = re.compile(
        r"[\"\u201c\u201d]\s?([A-Z][^\"\u201c\u201d\n]{1,90}?)" + _Q_CLOSE +
        r"\s*(?:,\s*)?(?:(?:with respect to|as to|when used)[^\"\u201c\u201d.;]{0,60}?,?\s+)?"
        r"(?:shall\s+)?(?:means?|mean,|has the meaning|have the meaning|"
        r"shall have the meaning|is defined|refers to|includes?)")
    # parenthetical definitions: (the "Company"), (each, a "Share" and collectively, the "Shares")
    _PAREN_Q = re.compile(r"\([^()]{0,300}?" + _Q_OPEN + r"[^()]{0,300}?\)")
    _QTERM = re.compile(_Q_OPEN + r"([A-Z][^\"”\n]{1,90}?)" + _Q_CLOSE)
    # an ARTICLE / Section heading (used to end a definition when tool 1 is not available)
    _HEADING = re.compile(r"(?:\n[ \t]*|\s{2,})(?:ARTICLE\s+[IVXLC\d]+\b|"
                          r"(?:Section|SECTION|§)\s*\d{1,2}\.\d{1,2}\.?\s+[A-Z])")
    _DEF_BYREF = re.compile(r"(?:Section|Sections)\s+(\d{1,2}\.\d{1,2}(?:\([a-z0-9]+\))*)")

    def __init__(self, text: str, sections=None, max_def_chars: int = 12000):
        self.text = text
        self.sections = sections
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
        k = _norm_term(d.term)
        if not any(abs(x.start - d.start) < 5 for x in self.defs[k]):
            self.defs[k].append(d)

    def match(self, term: str) -> tuple[str | None, list[str]]:
        """Resolve a user/model-typed term to an index key.
        Returns (key or None, other candidate term names)."""
        k = _norm_term(term)
        for cand in (k, k.rstrip("s"), k + "s", re.sub(r"^the ", "", k)):
            if cand in self.defs:
                return cand, []
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
        subs = [t for t in self.defs if k in t or t in k]
        close = difflib.get_close_matches(k, list(self.defs), n=5, cutoff=0.75)
        sugg = list(dict.fromkeys(close + sorted(subs, key=len)))[:6]
        return None, [self.defs[s][0].term for s in sugg]

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
    log = log if log is not None else ToolLog()
    T = index.text
    S = index.sections

    def _sentence_with_quoted(body: str, term: str) -> str | None:
        qm = re.search(_Q_OPEN + re.escape(term) + _Q_CLOSE, body)
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
                    + (f" Similar defined terms: {', '.join(sugg)}" if sugg else ""))
        out = []
        if sugg:
            out.append("Other matching terms: " + ", ".join(sugg))
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
                    for qm in re.finditer(_Q_OPEN + re.escape(d.term) + _Q_CLOSE, T):
                        if abs(qm.start() - d.start) > 5:
                            win = T[max(0, qm.start() - 800):qm.end() + 1500]
                            found = _sentence_with_quoted(win, d.term)
                            if found:
                                text += f"\n  -> at char {qm.start()}: {found}"
                                break
            if len(text) > def_chars:     # long definitions (MAE carve-outs): continue via context slice
                text = text[:def_chars] + f"\n...[definition continues; read context[{d.start + def_chars}:{d.end}]]"
            where = f"Section {d.sid}, " if d.sid else ""
            out.append(f"[{d.kind}, {where}chars {d.start}-{d.end}]\n" + text)
        return f'"{index.defs[k][0].term}":\n' + "\n---\n".join(out)

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
        rows = [f"  char {p}: ...{' '.join(T[max(0, p - 80):p + 120].split())}..." for p in hits[:k]]
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

    tools = {"define": define, "term_usage": term_usage}
    if S is not None:
        tools["terms_in"] = terms_in
    return {n: _wrap(n, f, log, max_chars) for n, f in tools.items()}


TOOL_PROMPT = """\
In the REPL you also have a definitions tool. It returns strings; print them.
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
    """Build the definition index for one contract and add tool 2 to the REPL namespace
    (the dict where `context` and `llm_query` live). Pass tool 1's SectionIndex as
    `sections` to enable section labels, by-reference following and terms_in()."""
    idx = DefinitionIndex(context_text, sections)
    namespace.update(make_tools(idx, log, **kw))
    return idx


# ----------------------------------------------------------------------------
# Self-check:  python tool2_definitions.py [contract.txt]
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        text = open(sys.argv[1], encoding="utf-8").read()
    else:
        from tool1_section_locator import DEMO as text   # demo contract lives in tool 1
    log = ToolLog()

    print("=== tool 2 alone ===")
    ns: dict = {}
    idx = inject(ns, text, log)
    print("INDEX:", idx.stats())
    print(ns["define"]("Superior Proposal"), "\n")

    print("=== tool 2 with tool 1's section index ===")
    try:
        from tool1_section_locator import SectionIndex
        ns2: dict = {}
        idx2 = inject(ns2, text, log, sections=SectionIndex(text))
        print("INDEX:", idx2.stats())
        print(ns2["define"]("Superior Proposal"), "\n")
        print(ns2["define"]("MAE"), "\n")
        print(ns2["term_usage"]("Acquisition Proposal"), "\n")
        print(ns2["terms_in"]("6.3"), "\n")
    except ImportError:
        print("(tool1_section_locator.py not found next to this file; skipped)")
    print("LOG:", log.summary())