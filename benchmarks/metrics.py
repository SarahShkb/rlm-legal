"""Reports for the Contract-NLI and MAUD re-runs (writes <experiment>/report/*.csv).

Contract-NLI: unit = hypothesis (123 docs x 17). Accuracy and macro-F1 over the
3 labels; a missing/invalid hypothesis answer is wrong. CIs: bootstrap over
DOCUMENTS (a document's 17 hypotheses are not independent).
MAUD: unit = sample (152). Lenient accuracy (as maud_eval: answers recovered
from REPL code count), strict accuracy (only final answers), mean Jaccard.
Both: paired comparisons -- exact McNemar on per-unit correctness and a paired
bootstrap CI of the accuracy difference. Efficiency/cost: lexglue.metrics.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score

from lexglue.metrics import efficiency_table, load_records, overall_table  # noqa: F401

N_BOOTSTRAP = 1000
SEED = 42
NLI_LABELS = ["Entailment", "Contradiction", "NotMentioned"]


def _mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return only_a, only_b, p


# ------------------------------------------------
# Contract-NLI
# ------------------------------------------------

def _nli_long(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (doc, method, repeat, hypothesis)."""
    rows = []
    for r in df.itertuples():
        pred = r.prediction if isinstance(r.prediction, dict) else {}
        for h, g in r.gold.items():
            p = pred.get(h)
            rows.append({"index": r.index, "method": r.method, "repeat": r.repeat, "hypothesis": h,
                         "gold": g, "pred": p if p else "__none__", "correct": p == g})
    return pd.DataFrame(rows)


def contract_nli_quality(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    long = _nli_long(df)
    rng = np.random.default_rng(SEED)
    quality = []
    for (method, repeat), g in long.groupby(["method", "repeat"]):
        docs = g["index"].unique()
        by_doc = {d: x for d, x in g.groupby("index")}

        def stats(sub):
            return (sub["correct"].mean(),
                    f1_score(sub["gold"], sub["pred"], labels=NLI_LABELS, average="macro", zero_division=0))
        acc, macro = stats(g)
        boot = np.array([stats(pd.concat([by_doc[d] for d in rng.choice(docs, len(docs))]))
                         for _ in range(N_BOOTSTRAP)])
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        per_class = f1_score(g["gold"], g["pred"], labels=NLI_LABELS, average=None, zero_division=0)
        recs = df[(df["method"] == method) & (df["repeat"] == repeat)]
        quality.append({
            "method": method, "repeat": repeat, "n_docs": len(docs), "n_hypotheses": len(g),
            "n_unanswered": int((g["pred"] == "__none__").sum()),
            **{f"parse_{k}": int(v) for k, v in recs["parse_status"].value_counts().items()},
            "accuracy": acc, "accuracy_ci_low": lo[0], "accuracy_ci_high": hi[0],
            "macro_f1": macro, "macro_f1_ci_low": lo[1], "macro_f1_ci_high": hi[1],
            **{f"f1_{l}": v for l, v in zip(NLI_LABELS, per_class)},
            "docs_all_17_correct": int(recs["correct"].sum()),
        })

    paired = []
    for repeat, g in long.groupby("repeat"):
        w = g.pivot_table(index=["index", "hypothesis"], columns="method", values="correct", aggfunc="first")
        if not {"vanilla", "rlm"} <= set(w.columns):
            continue
        w = w.dropna().astype(bool)
        v, r = w["vanilla"].to_numpy(), w["rlm"].to_numpy()
        only_v, only_r, p = _mcnemar(v, r)
        docs = w.index.get_level_values(0).to_numpy()
        doc_ids = np.unique(docs)
        diff_by_doc = {d: (r[docs == d].sum() - v[docs == d].sum(), (docs == d).sum()) for d in doc_ids}
        boot = []
        for _ in range(N_BOOTSTRAP):
            pick = rng.choice(doc_ids, len(doc_ids))
            boot.append(sum(diff_by_doc[d][0] for d in pick) / sum(diff_by_doc[d][1] for d in pick))
        paired.append({
            "repeat": repeat, "n_hypotheses": len(w), "only_vanilla_correct": only_v, "only_rlm_correct": only_r,
            "mcnemar_exact_p_hypothesis_level": p,
            "accuracy_diff_rlm_minus_vanilla": r.mean() - v.mean(),
            "diff_ci_low_doc_bootstrap": np.percentile(boot, 2.5),
            "diff_ci_high_doc_bootstrap": np.percentile(boot, 97.5),
        })
    return pd.DataFrame(quality), pd.DataFrame(paired)


# ------------------------------------------------
# MAUD
# ------------------------------------------------

def maud_quality(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    df = df.copy()
    df["multi_gold"] = df["gold"].map(len) > 1
    quality = []
    for (method, repeat), g in df.groupby(["method", "repeat"]):
        for subset, sub in (("all", g), ("gold single-letter", g[~g["multi_gold"]]),
                            ("gold multi-letter", g[g["multi_gold"]])):
            c = sub["correct"].to_numpy(bool)
            boot = c[rng.integers(0, len(c), (N_BOOTSTRAP, len(c)))].mean(1) if len(c) else np.array([np.nan])
            quality.append({
                "method": method, "repeat": repeat, "subset": subset, "n": len(sub),
                "n_no_answer": int(sub["prediction"].isna().sum()),
                "n_final": int((sub["parse_status"] == "final").sum()),
                "n_repl_recovered": int((sub["parse_status"] == "repl_recovered").sum()),
                "n_errors": int((sub["status"] != "ok").sum()),
                "acc_lenient": c.mean() if len(c) else np.nan,
                "acc_lenient_ci_low": np.percentile(boot, 2.5), "acc_lenient_ci_high": np.percentile(boot, 97.5),
                "acc_strict": sub["correct_strict"].fillna(False).astype(bool).mean(),
                "mean_jaccard": pd.to_numeric(sub["jaccard"], errors="coerce").fillna(0).mean(),
            })
    paired = []
    for repeat, g in df.groupby("repeat"):
        w = g.pivot(index="index", columns="method", values="correct")
        for base in ("vanilla", "rlm"):
            for other in w.columns:
                if other == base or base not in w.columns or (base == "rlm" and other == "vanilla"):
                    continue
                both = w[[base, other]].dropna().astype(bool)
                a, b = both[base].to_numpy(), both[other].to_numpy()
                only_a, only_b, p = _mcnemar(a, b)
                idx = rng.integers(0, len(a), (N_BOOTSTRAP, len(a)))
                boot = b[idx].mean(1) - a[idx].mean(1)
                paired.append({
                    "repeat": repeat, "baseline": base, "method": other, "n_paired": len(a),
                    "only_baseline_correct": only_a, "only_method_correct": only_b, "mcnemar_exact_p": p,
                    "acc_diff_method_minus_baseline": b.mean() - a.mean(),
                    "diff_ci_low": np.percentile(boot, 2.5), "diff_ci_high": np.percentile(boot, 97.5),
                })
    return pd.DataFrame(quality), pd.DataFrame(paired)


def tool_usage_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in df[df["method"].str.startswith("rlm_tool")].itertuples():
        summary = getattr(r, "tool_log_summary", None)
        if isinstance(summary, dict):
            rows.append({"method": r.method, "index": r.index, **{f"tool_{k}": v for k, v in summary.items()}})
    if not rows:
        return pd.DataFrame()
    t = pd.DataFrame(rows)
    num = t.select_dtypes("number").columns.drop("index", errors="ignore")
    return t.groupby("method")[list(num)].agg(["mean", "sum"]).reset_index()


def build_report(exp_dir: Path, dataset: str) -> dict[str, pd.DataFrame]:
    df = load_records(exp_dir)
    if dataset == "contract_nli":
        quality, paired = contract_nli_quality(df)
        tables = {"quality": quality, "paired_vanilla_vs_rlm": paired}
    else:
        quality, paired = maud_quality(df)
        tables = {"quality": quality, "paired": paired, "tool_usage": tool_usage_table(df)}
    tables["efficiency_cost"] = efficiency_table(df)
    out = exp_dir / "report"
    out.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t.to_csv(out / f"{name}.csv", index=False)
    flat = df.drop(columns=["error_traceback"], errors="ignore").copy()
    for col in flat.columns:
        if flat[col].map(lambda x: isinstance(x, (list, dict))).any():
            flat[col] = flat[col].map(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x)
    flat.to_csv(out / "all_records.csv", index=False)
    return tables
