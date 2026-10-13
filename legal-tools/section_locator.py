"""
tool1_section_locator.py — Tool 1: section locator for RLM
==========================================================

Parses a merger agreement once into a section map
(Article / Section number / heading / char offsets in `context`) and exposes
navigation functions for the RLM REPL:

    toc()                        table of contents (ids, headings, char spans)
    find_sections(query, k=5)    BM25 search over sections, with a snippet each
    get_section(ref, start=0)    text of "7.1", "Section 7.1(b)", "Article VIII", "chunk_3"
    refs_in(ref)                 cross-references ("Section 8.3", "Article VII") inside a section

Why (thesis framing)
--------------------
MAUD failure analysis: 8/10 RLM format failures hit the iteration cap while
still "searching or reading". This tool replaces ad-hoc regex navigation with
1–2 calls. It makes ConReader's long-range context relation (LCR) explicit:
the RLM can jump between distant, related parts of the agreement.

Standalone: standard library only. Tool 2 (tool2_definitions.py) can use the
SectionIndex built here to label definitions by section, but each tool works
without the other, so they can be ablated separately.

Offline validation on all 152 MAUD contracts (eval_locator.py, no API cost)
------------------------------------------------------------------------
Parsing: 151/152 contracts get a section map (median 89 sections);
contract_53 has unnumbered headings and falls back to fixed chunks.
Build time <= 0.21 s per contract.
Locator: gold = section(s) containing the MAUD deal-point excerpt,
query = deal-point name only:
                       R@1    R@3    R@5    R@10
    expand=False      0.29   0.49   0.59   0.70
    expand=True       0.73   0.96   0.98   0.99
Caveat: LEGAL_SYNONYMS was written after seeing which deal-point names failed,
so expand=True is optimistic. Report it as an ablation, not a held-out result.

Usage in the RLM runner (per contract):
    from tool1_section_locator import inject, ToolLog, TOOL_PROMPT
    log = ToolLog()
    sections = inject(repl_globals, contract_text, log)   # dict holding `context`
    system_prompt += "\\n" + TOOL_PROMPT
"""

from __future__ import annotations

import re
import time
from typing import Callable

from tools_utils import SectionIndex, ToolLog, _wrap

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def _roman_to_int(s: str) -> int:
    s = s.upper()
    if s.isdigit():
        return int(s)
    total, prev = 0, 0
    for ch in reversed(s):
        v = _ROMAN.get(ch, 0)
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total

# ----------------------------------------------------------------------------
# Optional legal query expansion (domain vocabulary, NOT answer labels).
# Ablate with expand=False.
# ----------------------------------------------------------------------------

LEGAL_SYNONYMS: dict[str, list[str]] = {
    "mae": ["material adverse effect", "material adverse change", "mac"],
    "material adverse effect": ["mae", "material adverse change", "carve-out"],
    "no-shop": ["solicitation", "acquisition proposal", "takeover proposal", "no solicitation"],
    "no shop": ["solicitation", "acquisition proposal", "takeover proposal"],
    "go-shop": ["window shop", "solicitation period", "excluded party"],
    "fiduciary out": ["superior proposal", "change of recommendation",
                      "adverse recommendation change", "intervening event"],
    "termination fee": ["break fee", "termination payment", "expenses", "reverse termination fee"],
    "matching right": ["match", "notice period", "negotiate in good faith", "superior proposal"],
    "covid": ["pandemic", "epidemic", "covid-19", "sars-cov-2"],
    "ordinary course": ["conduct of business", "consistent with past practice", "interim operating"],
    "outside date": ["end date", "termination date", "drop dead"],
    "financing": ["debt financing", "commitment letter", "financing cooperation"],
    "specific performance": ["equitable relief", "injunction", "enforcement"],
    "antitrust": ["hsr", "regulatory approval", "efforts", "burdensome condition"],
    "efforts": ["reasonable best efforts", "commercially reasonable efforts", "hell or high water"],
    "board recommendation": ["company board recommendation", "change of recommendation"],
    "closing conditions": ["conditions to obligations", "conditions precedent", "bring-down"],
    "ftr": ["terminate", "termination", "superior proposal", "fiduciary", "intervening event"],
    "fiduciary termination": ["terminate", "superior proposal", "termination fee"],
    "cor": ["change of recommendation", "adverse recommendation change", "withdraw", "modify", "qualify"],
    "change of recommendation": ["adverse recommendation change", "withdraw", "modify", "qualify"],
    "r&w": ["representations and warranties", "true and correct", "bring-down"],
    "accuracy of": ["representations and warranties", "true and correct", "conditions"],
    "compliance with covenant": ["performed", "complied", "all material respects", "obligations", "conditions"],
    "interim operating": ["conduct of business", "shall not", "prior to the effective time", "consent"],
    "consideration": ["merger consideration", "converted into the right to receive", "per share", "cash"],
    "litigation": ["injunction", "order", "restraint", "prohibit", "enjoin", "illegal", "conditions"],
    "knowledge": ["actual knowledge", "knowledge of the company", "individuals", "due inquiry"],
    "tail period": ["acquisition proposal", "termination fee", "months", "consummated"],
    "representations": ["representations and warranties", "accuracy of representations"],
}


# ----------------------------------------------------------------------------
# The tools (what the RLM calls)
# ----------------------------------------------------------------------------

def make_tools(index: SectionIndex, log: ToolLog | None = None, *,
               max_chars: int = 6000, expand: bool = True) -> dict[str, Callable]:
    """Return {name: function}. Every function returns a str and never raises."""
    log = log if log is not None else ToolLog()
    T = index.text

    def toc(max_items: int = 400) -> str:
        """Table of contents: section ids, headings and char spans in `context`."""
        if not index.structured:
            return (f"No section headings detected; document split into {len(index.sections)} "
                    f"chunks of ~{index._fallback_chunk} chars (ids chunk_0..). "
                    "Use find_sections(query) to locate content.")
        lines, cur = [], None
        for s in index.sections[:max_items]:
            if s.article and s.article != cur:
                cur = s.article
                lines.append(f"ARTICLE {cur} {index.articles[cur][0]}")
            lines.append(f"  {s.sid:<6} {s.title}  [chars {s.start}-{s.end}]")
        if len(index.sections) > max_items:
            lines.append(f"... {len(index.sections) - max_items} more sections")
        return "\n".join(lines)

    def find_sections(query: str, k: int = 5) -> str:
        """Rank sections for a deal point / topic. Returns id, heading, best snippet."""
        ranked = index.rank_sections(query, k, expand)
        if not ranked:
            return f"No section matched '{query}'. Try other wording or toc()."
        out = []
        for rank, (sid, score, ci) in enumerate(ranked, 1):
            sec = index.by_sid[sid]
            _, cs, ce = index.chunks[ci]
            snip = " ".join(T[cs:ce].split())[:350]
            out.append(f"{rank}. Section {sid} — {sec.title} (score {score:.1f}, "
                       f"section chars {sec.start}-{sec.end}, {sec.end - sec.start} chars)\n"
                       f"   ...{snip}...")
        return "\n".join(out)

    def get_section(ref: str, start: int = 0, limit: int = max_chars) -> str:
        """Text of 'Section 7.1', '7.1(b)', 'Article VIII' or 'chunk_3'.
        Long sections are paged: call again with start=<offset within section>."""
        r = ref.strip()
        am = re.fullmatch(r"(?:Article|ARTICLE)\s+([IVXLC]+|\d+)", r)
        if am:
            name = am.group(1).upper()
            if name.isdigit():
                name = next((a for a in index.articles if _roman_to_int(a) == int(name)), name)
            if name not in index.articles:
                return f"Article {am.group(1)} not found. Articles: {', '.join(index.articles) or 'none'}"
            title, a_s, a_e = index.articles[name]
            secs = [s for s in index.sections if s.article == name]
            listing = "\n".join(f"  {s.sid} {s.title} ({s.end - s.start} chars)" for s in secs)
            return (f"ARTICLE {name} {title} [chars {a_s}-{a_e}] contains:\n{listing}\n"
                    "Call get_section('<id>') for the text of one section.")
        sec, sub = index.lookup(r)
        if sec is None:
            return f"'{ref}' not found. Use toc() or find_sections()."
        body = T[sec.start:sec.end]
        offset = 0
        if sub:   # narrow to subsection (a), (b)(ii)...
            for part in re.findall(r"\([a-zA-Z0-9]+\)", sub):
                m = re.search(r"(?:^|\n|[.;:]\s+|\s{2,})" + re.escape(part) + r"\s", body[offset:])
                if not m:
                    break
                offset += m.end() - 1 - len(part)
            if offset:
                body = body[offset:]
                letter = sub[1]
                nxt = chr(ord(letter) + 1) if letter.isalpha() else None
                if nxt:
                    m2 = re.search(r"(?:\n|[.;:]\s+|\s{2,})\(" + nxt + r"\)\s", body[3:])
                    if m2:
                        body = body[:m2.end() - 1]
        piece = body[start:start + limit]
        more = len(body) - (start + len(piece))
        a = sec.start + offset + start
        head = f"Section {sec.sid}{sub} — {sec.title} [chars {a}-{a + len(piece)} of context]\n"
        tail = f"\n...[{more} more chars: get_section('{ref}', start={start + len(piece)})]" if more > 0 else ""
        return head + piece + tail

    def refs_in(ref: str) -> str:
        """Cross-references (other sections/articles) mentioned inside one section."""
        sec, _ = index.lookup(ref)
        if sec is None:
            return f"'{ref}' not found."
        body = T[sec.start:sec.end]
        xrefs = sorted({f"{m.group(1).rstrip('s')} {m.group(2)}" for m in index._REF.finditer(body)
                        if not m.group(2).startswith(sec.sid + "(") and m.group(2) != sec.sid})
        return f"Section {sec.sid} — {sec.title}\nCross-references: {', '.join(xrefs) or 'none'}"

    tools = {"toc": toc, "find_sections": find_sections, "get_section": get_section, "refs_in": refs_in}
    return {n: _wrap(n, f, log, max_chars) for n, f in tools.items()}


TOOL_PROMPT = """\
In the REPL you also have contract-navigation tools. They return strings; print them.
  toc()                          -> list of sections with headings and char spans
  find_sections("no-shop", k=5)  -> most relevant sections for a topic, with snippets
  get_section("7.1(b)")          -> text of a section/subsection; also "Article VIII".
                                    Long sections are paged: get_section("7.1", start=6000)
  refs_in("6.3")                 -> other sections referenced inside a section
Start with find_sections for the question topic, then get_section on the top 1-2 results.
Prefer these tools to scanning `context` with regex or loops; they save iterations.
"""


def inject(namespace: dict, context_text: str, log: ToolLog | None = None, **kw) -> SectionIndex:
    """Build the section index for one contract and add tool 1 to the REPL namespace
    (the dict where `context` and `llm_query` live). Returns the index, which you can
    pass to tool2_definitions.inject(..., sections=index)."""
    idx = SectionIndex(context_text)
    namespace.update(make_tools(idx, log, **kw))
    return idx


# ----------------------------------------------------------------------------
# Self-check:  python tool1_section_locator.py [contract.txt]
# ----------------------------------------------------------------------------

DEMO = """AGREEMENT AND PLAN OF MERGER

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

if __name__ == "__main__":
    import sys
    text = open(sys.argv[1], encoding="utf-8").read() if len(sys.argv) > 1 else DEMO
    log, ns = ToolLog(), {}
    idx = inject(ns, text, log)
    print("INDEX:", idx.stats(), "\n")
    print(ns["toc"](), "\n")
    print(ns["find_sections"]("no-shop fiduciary out", k=3), "\n")
    print(ns["get_section"]("Section 6.3(b)"), "\n")
    print(ns["refs_in"]("6.3"), "\n")
    print(ns["get_section"]("Article VIII"), "\n")
    print("LOG:", log.summary())