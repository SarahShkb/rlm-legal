from __future__ import annotations

import bisect
from collections.abc import Callable
import math
import re
import time
from collections import Counter
from dataclasses import dataclass

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
# The index
# ----------------------------------------------------------------------------

class SectionIndex:
    """Parse once per contract; the tool functions read from it."""

    # "Section 7.1 Termination." / "SECTION 7.01. Termination" / "§ 7.1 Termination" at line start
    _SEC_WORD = re.compile(
        r"(?:(?<=\n)|^)[ \t]*(?:Section|SECTION|Sec\.|§)\s*(\d{1,2}\.\d{1,2})\.?\s+"
        r"([A-Z][^\n]{0,150})", re.MULTILINE)
    # bare "7.1 Termination." at line start
    _SEC_BARE = re.compile(r"(?:(?<=\n)|^)[ \t]*(\d{1,2}\.\d{1,2})\.?[ \t]+([A-Z][^\n]{0,150})", re.MULTILINE)
    # heading inside a flattened paragraph: "...  Section 5.01    Covenants of First Choice. (a) ..."
    _W = r"(?:[A-Z][\w’'\-]*|of|and|or|the|to|for|in|on|with|by|a|an|under|upon|its)"
    _SEC_FLAT = re.compile(r"(?:Section|SECTION|§)\s*(\d{1,2}\.\d{1,2})\.?\s+(" + _W +
                           r"(?:[ ,;&/\-]+" + _W + r"){0,14})\.(?=\s)")
    # bare number mid-line after a gap or sentence end: "... foregoing. 5.02 Parent Forbearance. ..."
    _SEC_FLAT_BARE = re.compile(r"(?:(?<=\s\s)|(?<=[.;:]\s))(\d{1,2}\.\d{1,2})\.?\s+(" + _W +
                                r"(?:[ ,;&/\-]+" + _W + r"){0,14})\.(?=\s)")
    _ARTICLE = re.compile(r"(?:(?<=\n)|^)[ \t]*ARTICLE\s+([IVXLC]+|\d+)\b[ .:\-–—]*([^\n]{0,100})", re.MULTILINE)
    _REF = re.compile(r"\b(Section|Sections|Article)\s+((?:\d{1,2}\.\d{1,2}(?:\([a-zA-Z0-9]+\))*)|[IVXLC]+)\b")

    def __init__(self, text: str, chunk_chars: int = 1500, fallback_chunk: int = 4000):
        self.text = text
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
                cands.append((_num_key(m.group(1)), m.start(), m.group(1), m.group(2)))
        line_pos = {c[1] for c in cands}
        for rx in (self._SEC_FLAT, self._SEC_FLAT_BARE):
            for m in rx.finditer(text):
                if m.start() not in line_pos and m.group(2)[0].isupper():
                    cands.append((_num_key(m.group(1)), m.start(), m.group(1), m.group(2)))
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
        art_c = [((_roman_to_int(m.group(1)),), m.start(), m.group(1).upper(), _art_title(m))
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

    # ---------------- lookups ----------------
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
            sid = ".".join(str(k) for k in _num_key(m.group(1)))
            return self.by_sid.get(sid), m.group(2)
        return None, ""

    # ---------------- search ----------------
    def _build_search(self, chunk_chars: int) -> None:
        self.chunks: list[tuple[str, int, int]] = []     # (sid, start, end)
        docs: list[list[str]] = []
        for sec in self.sections:
            title_toks = _tokens(sec.title) * 3          # heading words count more
            for s in range(sec.start, sec.end, chunk_chars):
                e = min(sec.end, s + chunk_chars + 200)
                self.chunks.append((sec.sid, s, e))
                docs.append(title_toks + _tokens(self.text[s:e]))
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
        q = _tokens(query)
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
# Small helpers
# ----------------------------------------------------------------------------

_OQ = "\"“”‘’'"          # quote characters seen in EDGAR text

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


def _num_key(num: str) -> tuple:
    """'5.02' -> (5, 2); '10.1' -> (10, 1)."""
    return tuple(int(p) for p in num.split(".") if p.isdigit())


def _norm_term(t: str) -> str:
    t = t.strip(_OQ + " ").lower()
    return re.sub(r"\s+", " ", t)


_STOP = {
    "a", "an", "the", "of", "and", "or",
    "to", "in", "on", "for", "by", "with",
    "as", "at", "be", "is", "are", "was",
    "were", "this", "that", "these", "those",
    "any", "all", "such", "each", "other",
    "its", "it", "from", "into", "under",
    "upon", "shall", "will", "may", "not",
    "no", "if", "than", "then", "which",
    "who", "whom", "whose", "hereof", "herein",
    "hereto", "hereunder", "thereof", "therein",
    "thereto", "party", "parties", "agreement",
    "section", "article"
    }

_TOK = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> list[str]:
    return [t for t in _TOK.findall(s.lower()) if t not in _STOP and len(t) > 1]


def _lis_by_key(cands: list[tuple]) -> list[tuple]:
    """Longest strictly increasing subsequence of candidates (already sorted by
    position) w.r.t. their numeric key. Real headings form a long increasing
    chain; in-text references ('see Section 8.3') break it and get dropped."""
    if not cands:
        return []
    tails, tails_idx, prev = [], [], [-1] * len(cands)
    for i, c in enumerate(cands):
        k = c[0]
        j = bisect.bisect_left(tails, k)
        if j == len(tails):
            tails.append(k)
            tails_idx.append(i)
        else:
            tails[j] = k
            tails_idx[j] = i
        prev[i] = tails_idx[j - 1] if j > 0 else -1
    out, i = [], tails_idx[-1]
    while i != -1:
        out.append(cands[i])
        i = prev[i]
    return out[::-1]



def _clip(s: str, limit: int) -> str:
    return s if len(s) <= limit else s[:limit] + f"\n...[truncated; {len(s) - limit} more chars]"


def _wrap(name: str, fn: Callable, log, max_chars: int) -> Callable:
    """Log every call; never let a tool error crash the REPL."""
    def wrapped(*a, **kw):
        t0 = time.time()
        try:
            out = fn(*a, **kw)
        except Exception as e:
            out = f"[{name} error: {type(e).__name__}: {e}]"
        out = _clip(out, max_chars + 600)
        log.record(name, {"args": [str(x)[:120] for x in a], **{k: str(v)[:120] for k, v in kw.items()}},
                   out, time.time() - t0)
        return out
    wrapped.__name__, wrapped.__doc__ = name, fn.__doc__
    return wrapped


# ----------------------------------------------------------------------------
# Optional legal query expansion (domain vocabulary, NOT answer labels).
# Ablate with expand=False to show its effect separately.
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
# Logging (for the ablation table). Same class as in tool2; either works for both.
# ----------------------------------------------------------------------------

@dataclass
class ToolLog:
    calls: list[dict] = field(default_factory=list)

    def record(self, name: str, args: dict, out: str, secs: float) -> None:
        self.calls.append({"tool": name, "args": args, "out_chars": len(out), "secs": round(secs, 4)})

    def summary(self) -> dict:
        c = Counter(x["tool"] for x in self.calls)
        return {"n_tool_calls": len(self.calls), "by_tool": dict(c),
                "chars_returned": sum(x["out_chars"] for x in self.calls)}

    def to_json(self) -> str:
        return json.dumps({"summary": self.summary(), "calls": self.calls}, ensure_ascii=False)