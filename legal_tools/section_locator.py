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

Needs tools_util.py next to it (shared helpers); standard library only. Tool 2 (tool2_definitions.py) can use the
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

Usage with the rlm package (per contract):
    from rlm import RLM
    from tools_util import load_maud_contract, ToolLog
    from tool1_section_locator import rlm_custom_tools
    text = load_maud_contract("~/maud", 34)            # contract_34
    log = ToolLog()
    rlm = RLM(..., custom_tools=rlm_custom_tools(text, log))
    result = rlm.completion(text, root_prompt=question)  # context == contract text

Command line:
    python tool1_section_locator.py --maud ~/maud --contract 34 --query "termination fee"
    python tool1_section_locator.py --maud ~/maud --all
"""

from __future__ import annotations

import bisect
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable

from legal_tools.tools_utils import (
    DEMO_CONTRACT,
    SECTION_REF,
    ToolLog,
    as_rlm_custom_tools,
    context_offset,
    load_maud_contract,
    maud_contract_ids,
    num_key,
    roman_to_int,
    tokens,
    wrap_tools,
)

# ----------------------------------------------------------------------------
# Data structure
# ----------------------------------------------------------------------------


@dataclass
class Section:
    sid: str            # "7.1"  (or "chunk_3" in fallback mode)
    key: tuple          # (7, 1) for ordering / lookup
    title: str
    start: int          # char offset in context (inclusive)
    end: int            # char offset in context (exclusive)
    article: str = ""   # "VII" if known


# ----------------------------------------------------------------------------
# Tool-1-specific helper
# ----------------------------------------------------------------------------

def _lis_by_key(cands: list[tuple]) -> list[tuple]:
    """Longest strictly increasing subsequence of candidates (sorted by position)
    w.r.t. their numeric key. Real headings form a long increasing chain;
    in-text references ('see Section 8.3') break it and get dropped."""
    if not cands:
        return []
    tails, tails_idx, prev = [], [], [-1] * len(cands)
    for i, c in enumerate(cands):
        j = bisect.bisect_left(tails, c[0])
        if j == len(tails):
            tails.append(c[0])
            tails_idx.append(i)
        else:
            tails[j] = c[0]
            tails_idx[j] = i
        prev[i] = tails_idx[j - 1] if j > 0 else -1
    out, i = [], tails_idx[-1]
    while i != -1:
        out.append(cands[i])
        i = prev[i]
    return out[::-1]


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


def expand_query(query: str) -> str:
    q = query.lower()
    extra = [" ".join(v) for k, v in LEGAL_SYNONYMS.items()
             if re.search(r"(?<![a-z])" + re.escape(k) + r"(?![a-z])", q)]
    return query + (" " + " ".join(extra) if extra else "")


# ----------------------------------------------------------------------------
# The index
# ----------------------------------------------------------------------------

class SectionIndex:
    """Parse once per contract; the tool functions read from it."""

    # "Section 7.1 Termination." / "SECTION 7.01. Termination" / "§ 7.1 Termination" at line start
    _SEC_WORD = re.compile(
        r"(?:(?<=\n)|^)[ \t]*(?:Section|SECTION|Sec\.|§)\s*(\d{1,2}\.\d{1,2})\.?\s+"
        r"([A-Z][^\n]{0,150})", re.M)
    # bare "7.1 Termination." at line start
    _SEC_BARE = re.compile(r"(?:(?<=\n)|^)[ \t]*(\d{1,2}\.\d{1,2})\.?[ \t]+([A-Z][^\n]{0,150})", re.M)
    # heading inside a flattened paragraph: "...  Section 5.01    Covenants of First Choice. (a) ..."
    _W = r"(?:[A-Z][\w’'\-]*|of|and|or|the|to|for|in|on|with|by|a|an|under|upon|its)"
    _SEC_FLAT = re.compile(r"(?:Section|SECTION|§)\s*(\d{1,2}\.\d{1,2})\.?\s+(" + _W +
                           r"(?:[ ,;&/\-]+" + _W + r"){0,14})\.(?=\s)")
    # bare number mid-line after a gap or sentence end: "... foregoing. 5.02 Parent Forbearance. ..."
    _SEC_FLAT_BARE = re.compile(r"(?:(?<=\s\s)|(?<=[.;:]\s))(\d{1,2}\.\d{1,2})\.?\s+(" + _W +
                                r"(?:[ ,;&/\-]+" + _W + r"){0,14})\.(?=\s)")
    _ARTICLE = re.compile(r"(?:(?<=\n)|^)[ \t]*ARTICLE\s+([IVXLC]+|\d+)\b[ .:\-–—]*([^\n]{0,100})", re.M)

    def __init__(self, text: str, chunk_chars: int = 1500, fallback_chunk: int = 4000,
                 base: int = 0):
        """text: the contract. base: where the contract starts inside the RLM's `context`
        (0 when the context IS the contract). Reported char offsets are shifted by base."""
        self.text = text
        self.base = base
        self.sections: list[Section] = []
        self.by_sid: dict[str, Section] = {}
        self.articles: dict[str, tuple[str, int, int]] = {}   # "VII" -> (title, start, end)
        self.structured = True
        self._fallback_chunk = fallback_chunk
        t0 = time.time()
        self._parse_sections()
        self._build_search(chunk_chars)
        self.build_seconds = round(time.time() - t0, 3)

    # ---------------- parsing ----------------
    def _is_toc(self, pos: int) -> bool:
        """Heading candidates inside a table of contents: the line ends in a page
        number, has dotted leaders, or is a dense list of 'Section x.y' entries."""
        ls = self.text.rfind("\n", 0, pos) + 1
        le = self.text.find("\n", pos)
        line = self.text[ls:le if le != -1 else len(self.text)].strip()
        if re.search(r"(?:\s{2,}|\.{4,}\s*)[ivx\d]{1,3}$", line) and len(line) < 120:
            return True
        if "......" in line or "……" in line:
            return True
        n = len(re.findall(r"(?:Section|SECTION)\s+\d{1,2}\.\d", line))
        return n >= 3 and len(line) / n < 150

    def _parse_sections(self) -> None:
        text = self.text
        cands = []
        for rx in (self._SEC_WORD, self._SEC_BARE):
            for m in rx.finditer(text):
                cands.append((num_key(m.group(1)), m.start(), m.group(1), m.group(2)))
        line_pos = {c[1] for c in cands}
        for rx in (self._SEC_FLAT, self._SEC_FLAT_BARE):
            for m in rx.finditer(text):
                if m.start() not in line_pos and m.group(2)[0].isupper():
                    cands.append((num_key(m.group(1)), m.start(), m.group(1), m.group(2)))
        cands = [c for c in cands if not self._is_toc(c[1])]
        seen, uniq = set(), []
        for c in sorted(cands, key=lambda c: c[1]):
            if c[1] not in seen:
                seen.add(c[1])
                uniq.append(c)
        chain = _lis_by_key(uniq)

        if len(chain) < 3:                       # unstructured: fixed chunks
            self.structured = False
            fc = self._fallback_chunk
            for i, s in enumerate(range(0, len(text), fc)):
                sid = f"chunk_{i}"
                sec = Section(sid, (i,), text[s:s + 80].replace("\n", " ").strip(), s,
                              min(len(text), s + fc))
                self.sections.append(sec)
                self.by_sid[sid] = sec
            self._starts = [s.start for s in self.sections]
            return

        def _art_title(m):
            t = m.group(2).strip()
            if not t:   # title on the following non-empty line
                nxt = re.search(r"\S[^\n]{0,100}", text[m.end():m.end() + 300])
                t = nxt.group(0).strip() if nxt else ""
            return t if t.isupper() or len(t) < 60 else ""
        art_c = [((roman_to_int(m.group(1)),), m.start(), m.group(1).upper(), _art_title(m))
                 for m in self._ARTICLE.finditer(text)]
        art_c = [c for c in art_c if not self._is_toc(c[1])]
        art_starts = [(a[1], a[2], a[3]) for a in _lis_by_key(art_c)]
        for i, (s, name, title) in enumerate(art_starts):
            e = art_starts[i + 1][0] if i + 1 < len(art_starts) else len(text)
            self.articles[name] = (title, s, e)

        for i, (key, start, num, title_line) in enumerate(chain):
            end = chain[i + 1][1] if i + 1 < len(chain) else len(text)
            for (a_s, _, _) in art_starts:      # an ARTICLE heading ends the earlier section
                if start < a_s < end:
                    end = a_s
                    break
            title = re.split(r"(?<=[a-z\)])\.\s|\.$", title_line.strip())[0][:100]
            art = ""
            for name, (_, a_s, a_e) in self.articles.items():
                if a_s <= start < a_e:
                    art = name
            sid = ".".join(str(k) for k in key)
            if sid in self.by_sid:
                sid = f"{sid}#{i}"
            sec = Section(sid, key, title.strip(), start, end, art)
            self.sections.append(sec)
            self.by_sid[sid] = sec
        self._starts = [s.start for s in self.sections]

    # ---------------- lookups (also used by tool 2) ----------------
    def section_at(self, pos: int) -> str:
        """Id of the section containing char offset `pos`."""
        if not self.sections:
            return ""
        i = bisect.bisect_right(self._starts, pos) - 1
        return self.sections[i].sid if i >= 0 else "preamble"

    def lookup(self, ref: str) -> tuple[Section | None, str]:
        """'Section 7.01(b)(ii)' -> (Section 7.1, '(b)(ii)')."""
        ref = ref.strip()
        if ref in self.by_sid:
            return self.by_sid[ref], ""
        m = re.search(r"(\d{1,2}\.\d{1,2})((?:\([a-zA-Z0-9]+\))*)", ref)
        if m:
            sid = ".".join(str(k) for k in num_key(m.group(1)))
            return self.by_sid.get(sid), m.group(2)
        return None, ""

    # ---------------- search ----------------
    def _build_search(self, chunk_chars: int) -> None:
        self.chunks: list[tuple[str, int, int]] = []     # (sid, start, end)
        docs: list[list[str]] = []
        for sec in self.sections:
            title_toks = tokens(sec.title) * 3          # heading words count more
            for s in range(sec.start, sec.end, chunk_chars):
                e = min(sec.end, s + chunk_chars + 200)
                self.chunks.append((sec.sid, s, e))
                docs.append(title_toks + tokens(self.text[s:e]))
        self._tf = [Counter(d) for d in docs]
        self._len = [len(d) for d in docs]
        self._avg = (sum(self._len) / len(self._len)) if self._len else 1.0
        df = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(docs)
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def bm25(self, query: str, k1: float = 1.4, b: float = 0.75) -> list[tuple[float, int]]:
        """Scores over chunks: [(score, chunk_index)], best first."""
        q = tokens(query)
        phrases = [p.strip().lower() for p in re.split(r"[,;]", query) if len(p.split()) > 1]
        scores = []
        for i, tf in enumerate(self._tf):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self._idf.get(t, 0) * f * (k1 + 1) / (f + k1 * (1 - b + b * self._len[i] / self._avg))
            if s and phrases:
                _, cs, ce = self.chunks[i]
                low = self.text[cs:ce].lower()
                s += sum(1.5 for p in phrases if p in low)
            if s:
                scores.append((s, i))
        scores.sort(reverse=True)
        return scores

    def rank_sections(self, query: str, k: int = 5, expand: bool = True) -> list[tuple[str, float, int]]:
        """[(section_id, score, best_chunk_index)] for the top-k distinct sections."""
        q = expand_query(query) if expand else query
        best: dict[str, tuple[float, int]] = {}
        for score, ci in self.bm25(q):
            sid = self.chunks[ci][0]
            if sid not in best:
                best[sid] = (score, ci)
            if len(best) >= k:
                break
        return [(sid, sc, ci) for sid, (sc, ci) in best.items()]

    def stats(self) -> dict:
        return {"chars": len(self.text), "structured": self.structured,
                "sections": len(self.sections), "articles": len(self.articles),
                "build_seconds": self.build_seconds}


# ----------------------------------------------------------------------------
# The tools (what the RLM calls)
# ----------------------------------------------------------------------------

def make_tools(index: SectionIndex, log: ToolLog | None = None, *,
               max_chars: int = 6000, expand: bool = True) -> dict[str, Callable]:
    """Return {name: function}. Every function returns a str and never raises."""
    T = index.text
    B = index.base       # offsets reported to the model are offsets in `context`

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
            lines.append(f"  {s.sid:<6} {s.title}  [chars {B + s.start}-{B + s.end}]")
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
                       f"section chars {B + sec.start}-{B + sec.end}, {sec.end - sec.start} chars)\n"
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
                name = next((a for a in index.articles if roman_to_int(a) == int(name)), name)
            if name not in index.articles:
                return f"Article {am.group(1)} not found. Articles: {', '.join(index.articles) or 'none'}"
            title, a_s, a_e = index.articles[name]
            secs = [s for s in index.sections if s.article == name]
            listing = "\n".join(f"  {s.sid} {s.title} ({s.end - s.start} chars)" for s in secs)
            return (f"ARTICLE {name} {title} [chars {B + a_s}-{B + a_e}] contains:\n{listing}\n"
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
        a = B + sec.start + offset + start
        head = f"Section {sec.sid}{sub} — {sec.title} [chars {a}-{a + len(piece)} of context]\n"
        tail = f"\n...[{more} more chars: get_section('{ref}', start={start + len(piece)})]" if more > 0 else ""
        return head + piece + tail

    def refs_in(ref: str) -> str:
        """Cross-references (other sections/articles) mentioned inside one section."""
        sec, _ = index.lookup(ref)
        if sec is None:
            return f"'{ref}' not found."
        body = T[sec.start:sec.end]
        xrefs = sorted({f"{m.group(1).rstrip('s')} {m.group(2)}" for m in SECTION_REF.finditer(body)
                        if not m.group(2).startswith(sec.sid + "(") and m.group(2) != sec.sid})
        return f"Section {sec.sid} — {sec.title}\nCross-references: {', '.join(xrefs) or 'none'}"

    tools = {"toc": toc, "find_sections": find_sections, "get_section": get_section, "refs_in": refs_in}
    return wrap_tools(tools, log, max_chars)


# ----------------------------------------------------------------------------
# RLM adapter: the `custom_tools` dict expected by rlm.RLM(...)
# ----------------------------------------------------------------------------

TOOL_DESCRIPTIONS = {
    "toc": "toc() -> str. Table of contents of the contract in `context`: section ids, headings, "
           "char spans. Use it to see the structure.",
    "find_sections": "find_sections(query: str, k: int = 5) -> str. Ranks the contract's sections for a "
                     "topic or deal point (e.g. 'no-shop', 'MAE definition', 'termination fee') and "
                     "returns id, heading and a snippet for each. Start here.",
    "get_section": "get_section(ref: str, start: int = 0) -> str. Text of a section or subsection: "
                   "'7.1', 'Section 7.1(b)', 'Article VIII'. Long sections are paged with start=.",
    "refs_in": "refs_in(ref: str) -> str. Other sections referenced inside a section "
               "(follow them with get_section).",
}


def rlm_custom_tools(contract_text: str, log: ToolLog | None = None, *,
                     context: str | None = None, index: "SectionIndex | None" = None,
                     expand: bool = True, max_chars: int = 6000) -> dict:
    """Tool 1 for ONE contract, in the format of rlm.RLM(custom_tools=...).
    context: the RLM context if it is not exactly the contract (offsets get shifted).
    index:   an existing SectionIndex to reuse (e.g. the one you also give tool 2)."""
    idx = index or SectionIndex(contract_text, base=context_offset(contract_text, context))
    return as_rlm_custom_tools(make_tools(idx, log, max_chars=max_chars, expand=expand), TOOL_DESCRIPTIONS)


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


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Self-check of tool 1 (section locator).")
    ap.add_argument("--maud", help="path to your cloned MAUD repo (or its data/ folder)")
    ap.add_argument("--contract", help="contract number, e.g. 34 or contract_34")
    ap.add_argument("--query", default="no-shop fiduciary out", help="find_sections query")
    ap.add_argument("--all", action="store_true", help="parse every MAUD contract and print stats")
    ap.add_argument("file", nargs="?", help="or: any contract .txt file")
    a = ap.parse_args()

    if a.maud and a.all:
        bad = []
        for cid in maud_contract_ids(a.maud):
            st = SectionIndex(load_maud_contract(a.maud, cid)).stats()
            print(f"contract_{cid:<4} sections={st['sections']:<4} articles={st['articles']:<3} "
                  f"structured={st['structured']}  {st['build_seconds']}s")
            if not st["structured"]:
                bad.append(cid)
        print("unstructured (chunk fallback):", bad or "none")
        raise SystemExit

    if a.maud:
        text = load_maud_contract(a.maud, a.contract if a.contract is not None else 0)
    elif a.file:
        text = open(a.file, encoding="utf-8").read()
    else:
        text = DEMO_CONTRACT
    log, ns = ToolLog(), {}
    idx = inject(ns, text, log)
    print("INDEX:", idx.stats(), "\n")
    print(ns["toc"](max_items=25), "\n")
    print(ns["find_sections"](a.query, k=3), "\n")
    top = idx.rank_sections(a.query, k=1)
    if top:
        print(ns["get_section"](top[0][0], limit=800), "\n")
        print(ns["refs_in"](top[0][0]), "\n")
    print("LOG:", log.summary())