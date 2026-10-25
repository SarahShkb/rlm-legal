"""Scores, paired tests, stability and cost/efficiency tables for a LexGLUE experiment.

Scoring rules (kept identical for every task and method):
  * every scheduled (record, method, repeat) counts; a failed run (API error,
    unparseable answer, invalid label) is a wrong answer, never dropped;
  * a label that is not a candidate is ignored (it is not a false positive),
    it only causes a miss -- invalid-label counts are reported separately;
  * micro-F1 over all candidate labels; macro-F1 averaged over the labels that
    occur in the gold labels or the predictions of the evaluated records
    (as sklearn does for multi-class); with 100 records many of the 100
    LEDGAR / EURLEX labels never occur, and averaging zeros for them would
    measure the sample, not the model;
  * ECtHR A/B and UNFAIR-ToS: an empty label set is scored as an extra
    "none" class (LexGLUE convention);
  * accuracy = exact match (exact label set for multi-label tasks);
  * 95% CIs: percentile bootstrap over records (1000 resamples, fixed seed);
    vanilla vs RLM: exact McNemar test on per-record correctness and a paired
    bootstrap CI of the micro-F1 difference.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from lexglue.config import MULTI_CLASS, MULTI_LABEL, MULTIPLE_CHOICE, TASKS
from lexglue.data import label_names, load_split

N_BOOTSTRAP = 1000
SEED = 42
NONE_CLASS = "__none__"


# ------------------------------------------------
# Loading
# ------------------------------------------------

def load_records(exp_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((exp_dir / "records").glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            rows += [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise SystemExit(f"No records in {exp_dir / 'records'}")
    df = pd.DataFrame(rows)
    # a key that errored and was re-run later: keep its last "ok" row, else its last row
    df["_ok"] = df["status"].eq("ok")
    df["_order"] = range(len(df))
    df = (df.sort_values(["_ok", "_order"])
            .groupby(["task", "index", "method", "repeat"], as_index=False).tail(1)
            .sort_values(["task", "repeat", "index", "method"])
            .drop(columns=["_ok", "_order"])
            .reset_index(drop=True))
    df["correct"] = df["correct"].fillna(False).astype(bool)
    return df


def task_columns(task: str, split: str) -> list:
    task_type = TASKS[task]["task_type"]
    if task_type == MULTIPLE_CHOICE:
        return list(range(5))
    cols = label_names(task, load_split(task, split).features)
    if TASKS[task]["none_class"]:
        cols = cols + [NONE_CLASS]
    return cols


# ------------------------------------------------
# F1
# ------------------------------------------------

def binarize(task: str, values: list, columns: list) -> np.ndarray:
    """values: gold or predictions (None = failed); returns n x k bool."""
    pos = {c: j for j, c in enumerate(columns)}
    m = np.zeros((len(values), len(columns)), dtype=bool)
    multi = TASKS[task]["task_type"] == MULTI_LABEL
    for i, v in enumerate(values):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        items = list(v) if multi else [v]
        if multi and not items and NONE_CLASS in pos:
            m[i, pos[NONE_CLASS]] = True
        for item in items:
            if item in pos:
                m[i, pos[item]] = True
    return m


def f1_scores(Y: np.ndarray, P: np.ndarray) -> tuple[float, float]:
    tp = (Y & P).sum(0)
    fp = (~Y & P).sum(0)
    fn = (Y & ~P).sum(0)
    denom = 2 * tp.sum() + fp.sum() + fn.sum()
    micro = 2 * tp.sum() / denom if denom else 0.0
    present = Y.any(0) | P.any(0)
    per_label = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(len(tp)), where=(2 * tp + fp + fn) > 0)
    macro = float(per_label[present].mean()) if present.any() else 0.0
    return float(micro), macro


def _bootstrap(fn, n: int, rng: np.random.Generator) -> np.ndarray:
    return np.array([fn(rng.integers(0, n, n)) for _ in range(N_BOOTSTRAP)])


def score_group(task: str, g: pd.DataFrame, columns: list) -> dict:
    """g: rows of one task x method x repeat."""
    Y = binarize(task, g["gold"].tolist(), columns)
    P = binarize(task, g["prediction"].tolist(), columns)
    micro, macro = f1_scores(Y, P)
    correct = g["correct"].to_numpy()
    rng = np.random.default_rng(SEED)
    boot = _bootstrap(lambda idx: (*f1_scores(Y[idx], P[idx]), correct[idx].mean()), len(g), rng)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    parse = g["parse_status"].value_counts().to_dict()
    return {
        "n": len(g),
        "n_errors": int((g["status"] != "ok").sum()),
        "n_failed_answers": int(g["prediction"].isna().sum()),
        "n_invalid_labels": int(g["invalid_labels"].map(lambda x: len(x) if isinstance(x, list) else 0).sum()),
        **{f"parse_{k}": int(v) for k, v in sorted(parse.items())},
        "accuracy": correct.mean(),
        "accuracy_ci_low": lo[2], "accuracy_ci_high": hi[2],
        "micro_f1": micro, "micro_f1_ci_low": lo[0], "micro_f1_ci_high": hi[0],
        "macro_f1": macro, "macro_f1_ci_low": lo[1], "macro_f1_ci_high": hi[1],
    }


# ------------------------------------------------
# Tables
# ------------------------------------------------

def quality_table(df: pd.DataFrame, split: str) -> pd.DataFrame:
    out = []
    for task, tdf in df.groupby("task", sort=False):
        columns = task_columns(task, split)
        for (method, repeat), g in tdf.groupby(["method", "repeat"]):
            out.append({"task": task, "method": method, "repeat": repeat, **score_group(task, g, columns)})
    return pd.DataFrame(out)


def paired_table(df: pd.DataFrame, split: str) -> pd.DataFrame:
    out = []
    for (task, repeat), g in df.groupby(["task", "repeat"], sort=False):
        wide = g.pivot(index="index", columns="method", values=["correct", "prediction", "gold"])
        if not {"vanilla", "rlm"} <= set(g["method"]):
            continue
        both = wide.dropna(subset=[("correct", "vanilla"), ("correct", "rlm")])
        v = both[("correct", "vanilla")].astype(bool).to_numpy()
        r = both[("correct", "rlm")].astype(bool).to_numpy()
        only_v, only_r = int((v & ~r).sum()), int((~v & r).sum())
        p = binomtest(min(only_v, only_r), only_v + only_r, 0.5).pvalue if only_v + only_r else 1.0

        columns = task_columns(task, split)
        Y = binarize(task, both[("gold", "vanilla")].tolist(), columns)
        Pv = binarize(task, both[("prediction", "vanilla")].tolist(), columns)
        Pr = binarize(task, both[("prediction", "rlm")].tolist(), columns)
        diff = lambda idx: f1_scores(Y[idx], Pr[idx])[0] - f1_scores(Y[idx], Pv[idx])[0]
        boot = _bootstrap(diff, len(both), np.random.default_rng(SEED))
        out.append({
            "task": task, "repeat": repeat, "n_paired": len(both),
            "both_correct": int((v & r).sum()), "only_vanilla_correct": only_v,
            "only_rlm_correct": only_r, "both_wrong": int((~v & ~r).sum()),
            "mcnemar_exact_p": p,
            "micro_f1_diff_rlm_minus_vanilla": diff(np.arange(len(both))),
            "diff_ci_low": np.percentile(boot, 2.5), "diff_ci_high": np.percentile(boot, 97.5),
        })
    return pd.DataFrame(out)


def stability_table(quality: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Spread across repeats, and how often a record gets the same answer in every repeat."""
    out = []
    for (task, method), q in quality.groupby(["task", "method"], sort=False):
        g = df[(df["task"] == task) & (df["method"] == method)]
        preds = g.assign(p=g["prediction"].map(lambda x: json.dumps(x, sort_keys=True)))
        same = preds.groupby("index")["p"].nunique().eq(1).mean()
        out.append({
            "task": task, "method": method, "n_repeats": len(q),
            **{f"{m}_{stat}": getattr(q[m], stat)() for m in ("micro_f1", "macro_f1", "accuracy")
               for stat in ("mean", "std", "min", "max")},
            "same_prediction_all_repeats": same,
        })
    return pd.DataFrame(out)


def efficiency_table(df: pd.DataFrame, project_pricing: dict | None = None) -> pd.DataFrame:
    df = df.copy()
    if project_pricing:
        p = project_pricing
        cache = p["input_cache_read"] if p.get("input_cache_read") is not None else p["prompt"]
        df["projected_cost_usd"] = (
            (df["prompt_tokens"].fillna(0) - df["cached_prompt_tokens"].fillna(0)) * p["prompt"]
            + df["cached_prompt_tokens"].fillna(0) * cache
            + df["completion_tokens"].fillna(0) * p["completion"]
        )
    out = []
    for (task, method), g in df.groupby(["task", "method"], sort=False):
        row = {"task": task, "method": method, "n_runs": len(g), "n_correct": int(g["correct"].sum())}
        for col in ("n_calls", "n_root_calls", "n_sub_calls", "prompt_tokens", "cached_prompt_tokens",
                    "completion_tokens", "reasoning_tokens", "total_tokens", "wall_time_s"):
            s = pd.to_numeric(g[col], errors="coerce")
            row.update({f"{col}_mean": s.mean(), f"{col}_median": s.median(), f"{col}_sum": s.sum()})
        wall = pd.to_numeric(g["wall_time_s"], errors="coerce")
        row["wall_time_s_p95"] = wall.quantile(0.95)
        prompt_sum = row["prompt_tokens_sum"]
        row["cache_hit_ratio"] = row["cached_prompt_tokens_sum"] / prompt_sum if prompt_sum else None
        row["n_truncated_calls"] = int(pd.to_numeric(g["n_truncated_calls"], errors="coerce").fillna(0).sum())
        if "n_empty_responses" in g:
            row["n_empty_responses"] = int(pd.to_numeric(g["n_empty_responses"], errors="coerce").fillna(0).sum())
        if "rlm_iterations" in g and method == "rlm":
            row["rlm_iterations_mean"] = pd.to_numeric(g["rlm_iterations"], errors="coerce").mean()
            row["rlm_hit_max_iterations_rate"] = g["rlm_hit_max_iterations"].fillna(False).astype(bool).mean()
            row["rlm_repl_errors_mean"] = pd.to_numeric(g["rlm_repl_errors"], errors="coerce").mean()
        for col in ("reported_cost_usd", "estimated_cost_usd", "projected_cost_usd"):
            if col in g and g[col].notna().any():
                s = pd.to_numeric(g[col], errors="coerce")
                row[f"{col}_sum"] = s.sum()
                row[f"{col}_mean"] = s.mean()
                row[f"{col}_per_correct"] = s.sum() / row["n_correct"] if row["n_correct"] else None
        row["providers"] = sorted({p for ps in g["providers"] if isinstance(ps, list) for p in ps})
        row["response_models"] = sorted({m for ms in g["response_models"] if isinstance(ms, list) for m in ms})
        out.append(row)
    return pd.DataFrame(out)


def overall_table(quality: pd.DataFrame, efficiency: pd.DataFrame) -> pd.DataFrame:
    """LexGLUE-style average over tasks (mean of per-task scores), plus totals."""
    q = quality.groupby(["task", "method"])[["micro_f1", "macro_f1", "accuracy"]].mean().reset_index()
    out = q.groupby("method")[["micro_f1", "macro_f1", "accuracy"]].mean().add_prefix("avg_over_tasks_")
    out["n_tasks"] = q.groupby("method")["task"].nunique()
    sums = [c for c in efficiency.columns if c.endswith("_sum")]
    out = out.join(efficiency.groupby("method")[sums + ["n_runs", "n_correct"]].sum())
    return out.reset_index()


def build_report(exp_dir: Path, split: str, project_pricing: dict | None = None) -> dict[str, pd.DataFrame]:
    df = load_records(exp_dir)
    quality = quality_table(df, split)
    tables = {
        "quality": quality,
        "paired_vanilla_vs_rlm": paired_table(df, split),
        "stability_across_repeats": stability_table(quality, df),
        "efficiency_cost": efficiency_table(df, project_pricing),
    }
    tables["overall"] = overall_table(quality, tables["efficiency_cost"])
    report_dir = exp_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(report_dir / f"{name}.csv", index=False)
    flat = df.copy()
    for col in flat.columns:
        if flat[col].map(lambda x: isinstance(x, (list, dict))).any():
            flat[col] = flat[col].map(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x)
    flat.drop(columns=["error_traceback"], errors="ignore").to_csv(report_dir / "all_records.csv", index=False)
    return tables
