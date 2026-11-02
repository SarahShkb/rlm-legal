"""
maud_evidence_coverage.py -- did the RLM ever get shown the gold evidence of a MAUD question?

For every RLM run of an experiment, takes what the root model received (the
non-system messages of every iteration's prompt: REPL outputs as they were
inserted, after any truncation) and, separately, what the sub-LLM received
(llm_query prompts), and measures how much of the question's gold evidence
(maud/dicts/maud_152_gold_spans.json, see maud_gold_spans.py) is in it.

Coverage: both texts are compared without whitespace and page-break underscores,
after unidecode; the gold evidence is cut into 40-character windows (step 20) and
coverage is the share of windows that occur in what the model was shown. Bins:
"seen" >= 0.8, "partial" >= 0.2, "not seen" < 0.2.

Errors then split into "retrieval" (the evidence never reached the model) and
"reading" (it was shown, the answer is still wrong).

    python -m maud.maud_evidence_coverage rerun_v1_morph ds16k_it8 ds32k_it8 ds64k_it8
      -> maud/results/<experiment>/report/evidence_coverage.csv (one row per run)
         maud/results/<experiment>/report/evidence_coverage_summary.csv
"""
from __future__ import annotations

import json
import os
import re
import sys
import tarfile

import pandas as pd
from unidecode import unidecode

from maud.maud_data_sampling import get_maud_data_dir

GOLD_PATH = "maud/dicts/maud_152_gold_spans.json"
RESULTS = "maud/results"
WINDOW, STEP = 40, 20
BINS = [(0.8, "seen"), (0.2, "partial"), (0.0, "not seen")]


def _norm(s: str) -> str:
    return re.sub(r"\s+|_{3,}", "", unidecode(s))


def coverage(gold: str, shown: str) -> float:
    g = _norm(gold)
    if len(g) <= WINDOW:
        return float(g in shown)
    wins = [g[k:k + WINDOW] for k in range(0, len(g) - WINDOW + 1, STEP)]
    return sum(w in shown for w in wins) / len(wins)


def coverage_bin(c: float) -> str:
    return next(name for lo, name in BINS if c >= lo)


def _strings(o) -> list[str]:
    if isinstance(o, str):
        return [o]
    if isinstance(o, dict):
        return [s for k, v in o.items() if k not in ("response", "final_answer") for s in _strings(v)]
    if isinstance(o, list):
        return [s for v in o for s in _strings(v)]
    return []


def shown_texts(lines: list[dict]) -> tuple[str, str]:
    """(normalized text the root was shown, normalized text the sub-LLM was shown)."""
    root, sub = {}, {}
    for o in lines:
        if o.get("type") != "iteration":
            continue
        for m in o.get("prompt") or []:
            if m.get("role") != "system":
                root.setdefault(m["content"], None)
        for cb in o.get("code_blocks") or []:
            for call in (cb.get("result") or {}).get("rlm_calls") or []:
                for s in _strings(call.get("prompt", call)):
                    sub.setdefault(s, None)
    return _norm("\n".join(root)), _norm("\n".join(sub))


MAX_TRAJECTORY_BYTES = 100_000_000   # a few runs wrote multi-GB REPL locals (up to 3.5 GB); skipped


def _shown_by_trajectory(exp_dir: str) -> dict[str, tuple[str, str] | None]:
    """trajectory basename -> shown_texts(), or None when the file is too large to load,
    from trajectories/ or trajectories.tar.zst.
    One file at a time: every iteration line carries the whole contract in its REPL locals."""
    out = {}
    if os.path.isdir(f"{exp_dir}/trajectories"):
        for root, _, files in os.walk(f"{exp_dir}/trajectories"):
            for f in files:
                if f.endswith(".jsonl"):
                    path = os.path.join(root, f)
                    if os.path.getsize(path) > MAX_TRAJECTORY_BYTES:
                        out[f] = None
                        continue
                    with open(path, encoding="utf-8") as fh:
                        out[f] = shown_texts([json.loads(l) for l in fh if l.strip()])
    elif os.path.exists(f"{exp_dir}/trajectories.tar.zst"):
        import zstandard
        with open(f"{exp_dir}/trajectories.tar.zst", "rb") as fh, \
                zstandard.ZstdDecompressor().stream_reader(fh) as rd, tarfile.open(fileobj=rd, mode="r|") as tf:
            for m in tf:
                if m.isfile() and m.name.endswith(".jsonl"):
                    if m.size > MAX_TRAJECTORY_BYTES:
                        out[os.path.basename(m.name)] = None
                        continue
                    text = tf.extractfile(m).read().decode("utf-8")
                    out[os.path.basename(m.name)] = shown_texts([json.loads(l) for l in text.splitlines() if l.strip()])
    return out


def analyze(experiment: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    exp_dir = f"{RESULTS}/{experiment}"
    with open(GOLD_PATH, encoding="utf-8") as f:
        gold = {g["row_id"]: g for g in json.load(f)}
    contracts = {}
    shown = _shown_by_trajectory(exp_dir)
    latest = {}
    with open(f"{exp_dir}/records/maud.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["method"].startswith("rlm"):
                latest[(r["row_id"], r["method"], r.get("repeat", 0))] = r
    rows = []
    for (row_id, method, repeat), r in sorted(latest.items()):
        g = gold.get(row_id)
        path = r.get("rlm_trajectory_path")
        key = os.path.basename(path) if path else None
        texts = shown.get(key)
        row = {"experiment": experiment, "method": method, "repeat": repeat, "row_id": row_id,
               "text_type": g["text_type"] if g else None, "gold_status": g["status"] if g else None,
               "status": r["status"], "correct": bool(r.get("correct")), "has_trajectory": texts is not None,
               "trajectory_too_large": key in shown and texts is None}
        if g and g["spans"] and texts is not None:
            name = g["contract_name"]
            if name not in contracts:
                with open(f"{get_maud_data_dir()}/contracts/{name}.txt", encoding="utf-8-sig") as fh:
                    contracts[name] = fh.read()
            gold_text = "\n".join(contracts[name][s:e] for s, e in g["spans"])
            root, sub = texts
            row["coverage_root"] = coverage(gold_text, root)
            row["coverage_any"] = max(row["coverage_root"], coverage(gold_text, sub)) if sub else row["coverage_root"]
            row["bin_root"] = coverage_bin(row["coverage_root"])
            row["bin_any"] = coverage_bin(row["coverage_any"])
        rows.append(row)
    per_run = pd.DataFrame(rows)
    return per_run, summarize(per_run)


def summarize(per_run: pd.DataFrame) -> pd.DataFrame:
    out = []
    scored = per_run.dropna(subset=["coverage_root"])
    for subset, d0 in (("all located", scored),
                       ("exact match only", scored[scored.gold_status.isin(["strict", "ambiguous"])])):
        for method, d in d0.groupby("method"):
            wrong = d[~d.correct]
            row = {"method": method, "gold_subset": subset, "n": len(d), "accuracy": d.correct.mean(),
                   "mean_coverage_root": d.coverage_root.mean(), "mean_coverage_any": d.coverage_any.mean(),
                   "n_wrong": len(wrong),
                   "wrong_not_seen": int((wrong.bin_any == "not seen").sum()),
                   "wrong_partial": int((wrong.bin_any == "partial").sum()),
                   "wrong_seen": int((wrong.bin_any == "seen").sum())}
            for _, name in BINS:
                b = d[d.bin_any == name]
                row[f"n_{name.replace(' ', '_')}"] = len(b)
                row[f"acc_{name.replace(' ', '_')}"] = b.correct.mean() if len(b) else float("nan")
            out.append(row)
    return pd.DataFrame(out)


if __name__ == "__main__":
    for exp in sys.argv[1:] or ["rerun_v1_morph"]:
        per_run, summary = analyze(exp)
        os.makedirs(f"{RESULTS}/{exp}/report", exist_ok=True)
        per_run.to_csv(f"{RESULTS}/{exp}/report/evidence_coverage.csv", index=False)
        summary.to_csv(f"{RESULTS}/{exp}/report/evidence_coverage_summary.csv", index=False)
        print(f"== {exp}: {len(per_run)} RLM runs, {int(per_run.has_trajectory.sum())} with trajectory")
        print(summary.round(3).to_string(index=False))
