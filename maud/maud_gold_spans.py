"""
maud_gold_spans.py -- character spans of the gold evidence of every MAUD 152 question.

Every MAUD row comes with the excerpt the annotators read (`text`, with <omitted>
gaps). This locates that excerpt in the full contract with LegalBench-RAG's
procedure (legalbenchrag/generate/generate_maud.py, Pipitone & Houir Alami 2024):
unidecode + drop all whitespace on both sides, split the excerpt on <omitted>,
drop a trailing "(Page(s) N)", find the pieces in order and map them back to the
original offsets. LegalBench-RAG rejects a piece that also occurs later in the
contract (ambiguous); here such rows keep the first match and are marked
"ambiguous", so they can be left out of an analysis.

That exact match fails on about 40% of our rows, mostly long excerpts that cross
a page break: the contract text keeps the "________________" separator and the
page number, the annotators' excerpt does not; other excerpts join passages with
an inner "(Page N)" marker instead of <omitted>. Two fallbacks, both also cutting
the excerpt at inner "(Page N)" markers, each with its own status: "pagebreak"
(exact match once page-break underscore runs are removed from the contract) and
"anchored" (unique 50-character windows of the excerpt, found in
order; the span runs from the first to the last one found and must have about
the excerpt's length). Last, "piecewise": excerpts stitched from many passages
(up to ~40, not always in document order, with repeated short headers) are
located passage by passage, and at least 90% of their characters must be found.

LegalBench-RAG's own MAUD spans only cover 20 deal-point categories with a
generated query each; where one of our rows falls in such a category, its
LegalBench-RAG spans are stored next to ours as a cross-check (lbr_overlap).

    python -m maud.maud_gold_spans      -> maud/dicts/maud_152_gold_spans.json
"""
from __future__ import annotations

import hashlib
import json
import os
import re

from unidecode import unidecode

from maud.maud_data_sampling import get_maud_data_dir, get_sample_rows_for_maud_152

OUT_PATH = "maud/dicts/maud_152_gold_spans.json"
LBR_DIR = "maud_workdir/legalbenchrag"   # benchmarks/maud.json + corpus/maud from the LegalBench-RAG download

# MAUD text_type -> the LegalBench-RAG query suffix (generate_maud.column_to_queries, non-None entries)
LBR_QUERIES = {
    "Type of Consideration": "What is the Type of Consideration",
    "Accuracy of Target R&W Closing Condition": "Information about the Closing Condition: Accuracy of Target's Representations and Warranties",
    "Compliance with Covenant Closing Condition": "Information about the Closing Condition: Compliance with Covenants",
    "Absence of Litigation Closing Condition": "Information about the Closing Condition: No Litigation clause",
    '"No MAE" R&W Made as of a Specified Date': "What is the Target's Representation & Warranty of No Material Adverse Effect, with regards to some specified date",
    "MAE Definition": 'What is the Definition of "Material Adverse Effect"',
    "Knowledge Definition": 'What is the Definition of "Knowledge"',
    "No-Shop": "Where is the No-Shop Clause",
    "Fiduciary exception:  Board determination (no-shop)": "What about the Fiduciary exception to the No-Shop Clause",
    "Superior Offer Definition": 'What is the Definition of "Superior Proposal"',
    "Intervening Event Definition": 'What is the Definition of "Interveining Event"',
    "FTR Triggers": "Information about the Fiduciary Termination Right Triggers for termination",
    "Tail Period & Acquisition Proposal Details": "Is there a Tail provision for acquisition proposals",
    "Breach of No Shop": "What happens during a Breach of No-Shop clause",
    "Breach of Meeting Covenant": "What happens during a Breach of Shareholder Meeting Covenant",
    "Ordinary course covenant": "What are the Ordinary course of business covenants",
    "General Antitrust Efforts Standard": "Where is the Closing Conditions: Regulatory Approvals clause",
    "Limitations on Antitrust Efforts": "I want information about the Limitations on Antitrust Efforts",
    "Specific Performance": "Where is the Specific Performance clause",
}


def normalize(raw: str) -> tuple[str, list[int]]:
    """Whitespace-free unidecoded text and, per character, its offset in `raw`."""
    text, sourcemap = [], []
    for i, c in enumerate(raw):
        for c0 in (unidecode(c) if ord(c) >= 0x80 else c):
            if not c0.isspace():
                text.append(c0)
                sourcemap.append(i)
    return "".join(text), sourcemap


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _pieces(excerpt: str, split_page_refs: bool = False) -> list[str]:
    """split_page_refs: also cut at inner "(Page N)" markers, which some excerpts use
    instead of <omitted> to join passages (LegalBench-RAG only drops a trailing one)."""
    pieces = re.split(r"\s*<omitted>\s*", re.sub(r"\s+", "", unidecode(excerpt)))
    pieces[-1] = re.sub(r"\(Pages?\s*[\d-]+\)\s*$", "", pieces[-1])
    if split_page_refs:
        pieces = [q for p in pieces for q in re.split(r"\(Pages?[\d-]+\)", p)]
    return [p for p in pieces if p]


def _end(sourcemap: list[int], cur: int, raw: str) -> int:
    return sourcemap[cur] if cur < len(sourcemap) else len(raw)


def _exact(pieces: list[str], text: str, sourcemap: list[int], raw: str):
    """LegalBench-RAG's in-order exact match: (ambiguous, spans) or None."""
    ambiguous, spans, cur = False, [], 0
    for piece in pieces:
        idx = text.find(piece, cur)
        if idx == -1:
            return None
        cur = idx + len(piece)
        ambiguous |= text.find(piece, cur) != -1
        spans.append((sourcemap[idx], _end(sourcemap, cur, raw)))
    return ambiguous, spans


def _anchored(pieces: list[str], text: str, sourcemap: list[int], raw: str, w: int = 50):
    """Spans from unique in-order w-character windows of each piece, or None."""
    spans, cur = [], 0
    for piece in pieces:
        hits = []
        for k in range(0, max(1, len(piece) - w + 1), w):
            win = piece[k:k + w]
            idx = text.find(win, cur)
            if idx != -1 and text.find(win, idx + 1) == -1:
                hits.append((idx - k, idx + len(win), k))
                cur = idx + len(win)
        if not hits or len(hits) < 0.5 * max(1, len(piece) // w):
            return None
        start = max(0, hits[0][0])                          # where the piece would start
        stop = hits[-1][1] + (len(piece) - hits[-1][2] - w)  # ... and end
        if not 0.8 <= (stop - start) / len(piece) <= 1.25:
            return None
        spans.append((sourcemap[start], _end(sourcemap, min(stop, len(text)), raw)))
    return spans


def _piecewise(pieces: list[str], text: str, sourcemap: list[int], raw: str, min_share: float = 0.9):
    """Each piece on its own, in any order: unique exact match, else anchored; spans or None
    when less than min_share of the excerpt's characters are found. Excerpts repeat passages,
    also as a longer and a shorter version, so a passage inside another one is counted once."""
    pieces = list(dict.fromkeys(pieces))
    pieces = [p for p in pieces if not any(p != q and p in q for q in pieces)]
    spans, found = [], 0
    for piece in pieces:
        idx = text.find(piece)
        if idx != -1 and text.find(piece, idx + 1) == -1:
            spans.append((sourcemap[idx], _end(sourcemap, idx + len(piece), raw)))
            found += len(piece)
        elif len(piece) >= 100 and (sub := _anchored([piece], text, sourcemap, raw)):
            spans.extend(sub)
            found += len(piece)
    return spans if found >= min_share * sum(map(len, pieces)) else None


def locate(excerpt: str, raw: str, norm: tuple[str, list[int]] | None = None) -> tuple[str, list[tuple[int, int]]]:
    """(status, spans) of `excerpt` in `raw`.

    status: strict | ambiguous (LegalBench-RAG's exact match) | pagebreak | anchored | piecewise | not_found."""
    text, sourcemap = norm or normalize(raw)
    pieces = _pieces(excerpt)
    if not pieces:
        return "not_found", []
    hit = _exact(pieces, text, sourcemap, raw)
    if hit:
        return ("ambiguous" if hit[0] else "strict"), merge_spans(hit[1])
    # the same text without page breaks: the underscore run, then also the page number before it
    # ("58____", "-12-____", "A-8____"); some excerpts keep that number, so both are tried
    flats = []
    for brk in (r"_{3,}", r"(?:-?[A-Z]?-?\d{1,3}-?)?_{3,}"):
        flat, flat_map, cur = [], [], 0
        for m in re.finditer(brk, text):
            flat.append(text[cur:m.start()])
            flat_map.extend(sourcemap[cur:m.start()])
            cur = m.end()
        flat.append(text[cur:])
        flat_map.extend(sourcemap[cur:])
        flats.append(("".join(flat), flat_map))
    pieces = _pieces(excerpt, split_page_refs=True)
    for flat, flat_map in flats:
        hit = _exact(pieces, flat, flat_map, raw)
        if hit and not hit[0]:
            return "pagebreak", merge_spans(hit[1])
    for flat, flat_map in flats:
        spans = _anchored(pieces, flat, flat_map, raw)
        if spans:
            return "anchored", merge_spans(spans)
    for flat, flat_map in flats:
        spans = _piecewise(pieces, flat, flat_map, raw)
        if spans:
            return "piecewise", merge_spans(spans)
    return "not_found", []


def overlap(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> dict:
    """Characters of a, of b and of their intersection."""
    inter = sum(max(0, min(e1, e2) - max(s1, s2)) for s1, e1 in a for s2, e2 in b)
    la, lb = sum(e - s for s, e in a), sum(e - s for s, e in b)
    return {"chars_ours": la, "chars_lbr": lb, "chars_common": inter,
            "iou": inter / (la + lb - inter) if la + lb - inter else 0.0}


def _lbr_index() -> tuple[dict[str, str], dict[tuple[str, str], list[tuple[int, int]]]]:
    """sha256 of a contract text -> LegalBench-RAG file path; (file path, query suffix) -> spans."""
    corpus = f"{LBR_DIR}/corpus/maud"
    by_hash = {}
    for f in os.listdir(corpus):
        with open(f"{corpus}/{f}", encoding="utf-8") as fh:
            by_hash[hashlib.sha256(fh.read().encode()).hexdigest()] = f"maud/{f}"
    with open(f"{LBR_DIR}/benchmarks/maud.json", encoding="utf-8") as fh:
        tests = json.load(fh)["tests"]
    spans = {}
    for t in tests:
        suffix = t["query"].split("; ", 1)[1]
        for sn in t["snippets"]:
            spans.setdefault((sn["file_path"], suffix), []).append(tuple(sn["span"]))
    return by_hash, spans


def build(split: str = "train") -> list[dict]:
    df = get_sample_rows_for_maud_152(split=split)
    contracts_dir = f"{get_maud_data_dir()}/contracts"
    lbr_by_hash, lbr_spans = _lbr_index() if os.path.isdir(LBR_DIR) else ({}, {})
    out = []
    for i, row in enumerate(df.itertuples()):
        # same decoding as benchmarks/maud_bench.py, so offsets index the RLM `context`
        with open(f"{contracts_dir}/{row.contract_name}.txt", encoding="utf-8-sig") as f:
            raw = f.read()
        status, spans = locate(row.text, raw)
        rec = {"index": i, "row_id": f"{row.contract_name}_{i}", "contract_name": row.contract_name,
               "question": row.question, "text_type": row.text_type, "status": status,
               "spans": spans, "gold_chars": sum(e - s for s, e in spans), "contract_chars": len(raw)}
        with open(f"{contracts_dir}/{row.contract_name}.txt", encoding="utf-8") as f:
            lbr_file = lbr_by_hash.get(hashlib.sha256(f.read().encode()).hexdigest())
        query = LBR_QUERIES.get(row.text_type)
        rec["lbr_file"] = lbr_file
        rec["lbr_spans"] = None
        if lbr_file and query and (lbr_file, query) in lbr_spans:
            # a leading BOM shifts our offsets by one against LegalBench-RAG's raw read
            shift = len(open(f"{contracts_dir}/{row.contract_name}.txt", encoding="utf-8").read()) - len(raw)
            rec["lbr_spans"] = merge_spans([(s - shift, e - shift) for s, e in lbr_spans[(lbr_file, query)]])
            rec["lbr_overlap"] = overlap(spans, rec["lbr_spans"]) if spans else None
        out.append(rec)
    return out


if __name__ == "__main__":
    recs = build()
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(recs, f, indent=1)
    from collections import Counter
    print(Counter(r["status"] for r in recs))
    cmp = [r for r in recs if r.get("lbr_overlap")]
    print(f"LegalBench-RAG spans for {sum(r['lbr_spans'] is not None for r in recs)} rows; "
          f"IoU with ours: mean {sum(r['lbr_overlap']['iou'] for r in cmp) / max(1, len(cmp)):.3f}, "
          f">= 0.9 for {sum(r['lbr_overlap']['iou'] >= 0.9 for r in cmp)}/{len(cmp)}")
    print(f"-> {OUT_PATH}")
