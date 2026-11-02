"""
compare_maud.py -- the RAG baseline (rag_v1) against vanilla and RLM (rerun_v1_morph) on MAUD 152.

Same model and provider (DeepSeek V4.1 Flash, Morph fp8) and the same prompts and
parser; the experiments ran on different days, when Morph's prices differed, so cost
is recomputed from each run's token counts at one price list (PRICES), plus the
one-off embedding cost of the index, amortized over the 152 questions.

Paired accuracy differences: exact McNemar test on the questions both methods ran.
For the RAG methods, accuracy is also split by whether the retrieved chunks hit
the gold evidence (maud/dicts/maud_152_gold_spans.json).

    python -m rag.compare_maud   -> maud/results/rag_v1/report/comparison*.csv
"""
from __future__ import annotations

import json
import os

import pandas as pd
from scipy.stats import binomtest

RESULTS = "maud/results"
BASELINES = ("rerun_v1_morph", ("vanilla", "rlm", "rlm_tool1", "rlm_tool12"))
RAG_EXPERIMENT = "rag_v1"
INDEX_USAGE = "maud_workdir/rag_index/text-embedding-3-large_rcts500/usage.json"
# USD per 1M tokens: uncached input, cached input, output (incl. reasoning)
PRICES = {
    "morph_2026-10-25": (0.10, 0.01, 0.60),     # when rerun_v1_morph ran (fitted from its billed costs)
    "morph_2026-10-30": (0.022, 0.004, 0.192),  # OpenRouter endpoint price when rag_v1 ran
}


def _records(experiment: str, methods=None) -> pd.DataFrame:
    latest = {}
    with open(f"{RESULTS}/{experiment}/records/maud.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if methods is None or r["method"] in methods:
                latest[(r["row_id"], r["method"], r.get("repeat", 0))] = r
    rows = []
    for r in latest.values():
        rag = r.get("rag") or {}
        rows.append({"experiment": experiment, "method": r["method"], "row_id": r["row_id"], "status": r["status"],
                     "correct": bool(r.get("correct")),
                     "prompt_tokens": r.get("prompt_tokens") or 0, "cached": r.get("cached_prompt_tokens") or 0,
                     "completion_tokens": r.get("completion_tokens") or 0, "wall_time_s": r.get("wall_time_s"),
                     "retrieved_chars": rag.get("retrieved_chars"), "gold_recall": rag.get("gold_recall")})
    return pd.DataFrame(rows)


def _cost(d: pd.DataFrame, price: tuple[float, float, float]) -> pd.Series:
    unc, cached, out = price
    return ((d.prompt_tokens - d.cached) * unc + d.cached * cached + d.completion_tokens * out) / 1e6


def compare() -> dict[str, pd.DataFrame]:
    base = _records(*BASELINES)
    rag = _records(RAG_EXPERIMENT)
    df = pd.concat([base, rag], ignore_index=True)
    with open(INDEX_USAGE) as f:
        embed_per_q = json.load(f)["cost_usd"] / 152
    summary = []
    for method, d in df.groupby("method", sort=False):
        row = {"method": method, "experiment": d.experiment.iloc[0], "n": len(d), "accuracy": d.correct.mean(),
               "prompt_tokens_mean": d.prompt_tokens.mean(), "completion_tokens_mean": d.completion_tokens.mean(),
               "wall_time_s_mean": d.wall_time_s.mean()}
        for name, p in PRICES.items():
            llm = _cost(d, p).mean()
            row[f"usd_per_q_{name}"] = llm + (embed_per_q if method.startswith("rag") else 0.0)
        if method.startswith("rag"):
            row["retrieved_chars_mean"] = d.retrieved_chars.mean()
            g = d.dropna(subset=["gold_recall"])
            row["gold_recall_mean"] = g.gold_recall.mean()
            row["acc_gold_hit"] = g[g.gold_recall > 0].correct.mean()
            row["n_gold_missed"] = int((g.gold_recall == 0).sum())
            row["acc_gold_missed"] = g[g.gold_recall == 0].correct.mean() if (g.gold_recall == 0).any() else float("nan")
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary["embedding_usd_per_q_included"] = summary.method.str.startswith("rag") * embed_per_q

    wide = df.pivot_table(index="row_id", columns="method", values="correct", aggfunc="first")
    paired = []
    for m in [m for m in summary.method if m.startswith("rag")]:
        for b in BASELINES[1]:
            w = wide[[m, b]].dropna().astype(bool)
            only_m, only_b = int((w[m] & ~w[b]).sum()), int((~w[m] & w[b]).sum())
            p = binomtest(only_m, only_m + only_b).pvalue if only_m + only_b else 1.0
            paired.append({"method": m, "baseline": b, "n_paired": len(w), "only_method_correct": only_m,
                           "only_baseline_correct": only_b, "acc_diff_method_minus_baseline": w[m].mean() - w[b].mean(),
                           "mcnemar_exact_p": p})
    return {"comparison": summary, "comparison_paired": pd.DataFrame(paired)}


if __name__ == "__main__":
    tables = compare()
    out = f"{RESULTS}/{RAG_EXPERIMENT}/report"
    os.makedirs(out, exist_ok=True)
    pd.set_option("display.width", 250)
    for name, t in tables.items():
        t.to_csv(f"{out}/{name}.csv", index=False)
        print(f"\n== {name}\n{t.round(4).to_string(index=False)}")
    print(f"\n-> {out}/")
