import json
from collections import Counter
from itertools import zip_longest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from llm.rlm import run_rlm_customized
from llm.vanila import run_vanilla
from utils.maud_utils import get_main_def, get_maud_data_dir

RANDOM_SEED = 42



def bin_contracts_by_length(maud_contracts_token_count_json_path: str, n_bins: int = 4) -> pd.DataFrame:

    # Load JSON already produced by maud_contract_token_counts.py 
    # (list of dicts, each with "file" and "n_tokens")
    with open(maud_contracts_token_count_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    contracts_list = data["contracts"]
    contract_token_counts = pd.Series(
        {item["file"]: item["n_tokens"] for item in contracts_list}
    )
    labels = [f"len_bin_{i + 1}" for i in range(n_bins)]

    bins = pd.qcut(
        contract_token_counts,
        q=n_bins,
        labels=labels
    )
    return pd.DataFrame({
        "contract_name": contract_token_counts.index.str.removesuffix(".txt"),
        "n_tokens_contract": contract_token_counts.values,
        "length_bin": bins.values,
    })


def add_complexity_features(df: pd.DataFrame, excerpt_tokenizer=None) -> pd.DataFrame:
    # adds complexity_tier (binary / 3-way / multi-way)
    # and excerpt_n_tokens columns, plus cleans up category

    df = df.copy()
    df["category"] = df["category"].astype(str).str.strip()

    n_options = df.groupby("question")["answer"].transform("nunique")
    df["n_answer_options"] = n_options
    df["complexity_tier"] = pd.cut(
        df["n_answer_options"], bins=[0, 2, 3, 100],
        labels=["binary", "3-way", "multi-way(4+)"]
    )

    if excerpt_tokenizer is None:
        df["excerpt_n_tokens"] = df["text"].astype(str).str.split().str.len()
    else:
        df["excerpt_n_tokens"] = df["text"].astype(str).map(excerpt_tokenizer)

    return df


def build_master_table(df: pd.DataFrame, contract_bins: pd.DataFrame) -> pd.DataFrame:
    filtered_df = df[df["contract_name"] != "<RARE_ANSWERS>"]
    merged = filtered_df.merge(contract_bins, on="contract_name", how="left")
    n_unmatched = merged["length_bin"].isna().sum()
    if n_unmatched:
        print(f"WARNING: {n_unmatched} rows didn't match a contract length bin -- "
              f"check contract_name formatting/spelling consistency.")
    return merged


def check_axis_confounding(master: pd.DataFrame) -> None:
    # check whether full-contract length correlates
    # with category and with excerpt length
    # and also check the correlation between excerpt length and category

    print("Mean excerpt length by category:")
    print(master.groupby("category")["excerpt_n_tokens"].mean().sort_values(ascending=False))
    print("\nContract length vs. category (row counts):")
    ct = pd.crosstab(master["length_bin"], master["category"])
    print(ct)
    chi2, _, _, _ = chi2_contingency(ct)
    n = ct.to_numpy().sum()
    k = min(ct.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * k)) if k > 0 else float("nan")
    print(f"Cramer's V(length_bin, category) = {cramers_v:.3f}  "
          f"(rule of thumb: <0.1 negligible, 0.1-0.3 weak, 0.3-0.5 moderate, >0.5 strong)")
    corr = master[["n_tokens_contract", "excerpt_n_tokens"]].corr().iloc[0, 1]
    print(f"\nCorrelation(full contract length, excerpt length) = {corr:.3f}")


# tells us how many distinct contracts are represented 
# in each length_bin x question_type cell
def question_support_table(df: pd.DataFrame) -> pd.DataFrame:
    support = (
        df.groupby(["question", "length_bin"])["contract_name"]
        .nunique()
        .unstack(fill_value=0)
    )

    return support.sort_values(
        by=list(support.columns),
        ascending=False
    )


# ---------------------------------------------------------------------------
# 5. SAMPLING
# ---------------------------------------------------------------------------

# this function is for sampling based of 4 features:
# 1. length_bin (contract-level full text length)
# 2. category (question type)
# 3. complexity_tier (binary / 3-way / multi-way)   
# 4.  length_text
# but for some reason, it doesn't work as I want. 
# One of the disadvantages is that it doesn't guarantee that the longest contracts 
# are included in the sample.
def stratified_sample(master: pd.DataFrame,
                       n_contracts_per_bin=None,
                       max_rows_per_contract: int = 6,
                       force_include_longest: int = 2,
                       target_total=(100, 250),
                       max_attempts: int = 6,
                       growth_factor: float = 1.6,
                       seed: int = RANDOM_SEED) -> pd.DataFrame:
    
    bin_order = (master.groupby("length_bin", observed=True)["n_tokens_contract"]
                        .mean().sort_values().index.tolist())
    n_bins = len(bin_order)
    print(f"Bin order resolved shortest -> longest (by mean contract length): {bin_order}")
 
    if n_contracts_per_bin is None:
        n_contracts_per_bin = np.linspace(4, 10, n_bins).round().astype(int).tolist()
    elif isinstance(n_contracts_per_bin, int):
        n_contracts_per_bin = [n_contracts_per_bin] * n_bins
    n_contracts_per_bin = list(n_contracts_per_bin)
    assert len(n_contracts_per_bin) == n_bins, (
        f"n_contracts_per_bin must be an int or a list with one value per bin "
        f"({n_bins} bins found: {bin_order})"
    )
 
    contract_lengths_master = (master.drop_duplicates("contract_name")
                                      .set_index("contract_name")["n_tokens_contract"])
    overall_longest = contract_lengths_master.nlargest(force_include_longest).index.tolist()
    print(f"Force-including as longest ({force_include_longest}): {overall_longest}")
    if contract_lengths_master.isna().any():
        print(f"NOTE: {contract_lengths_master.isna().sum()} contracts in master have no "
              f"n_tokens_contract (merge didn't match) -- they CANNOT be force-included "
              f"even if they're actually your longest. Run diagnose_contract_matching().")
 
    # Precompute each bin's full contract pool once, so retries don't refilter master repeatedly.
    bin_pools = {}
    bin_dfs = {}
    for bin_name in bin_order:
        bin_df = master[master["length_bin"] == bin_name]
        bin_dfs[bin_name] = bin_df
        lengths = bin_df.drop_duplicates("contract_name").set_index("contract_name")["n_tokens_contract"]
        bin_pools[bin_name] = lengths.sort_values()
 
    def build_once(n_per_bin):
        selected = []
        for bin_name, n_pick_target in zip(bin_order, n_per_bin):
            lengths = bin_pools[bin_name]
            n_pick = min(n_pick_target, len(lengths))
            if n_pick == 0:
                continue
            idx = np.linspace(0, len(lengths) - 1, n_pick).round().astype(int)
            picked_contracts = set(lengths.iloc[idx].index.tolist())
            picked_contracts |= (set(overall_longest) & set(lengths.index))
 
            bin_df = bin_dfs[bin_name]
            for contract in picked_contracts:
                c_df = bin_df[bin_df["contract_name"] == contract]
                picks = []
                for _, cell_df in c_df.groupby("category", observed=True):
                    if len(cell_df) > 0:
                        picks.append(cell_df.sample(1, random_state=seed))
                if picks:
                    contract_pick = pd.concat(picks)
                    if len(contract_pick) > max_rows_per_contract:
                        contract_pick = contract_pick.sample(max_rows_per_contract, random_state=seed)
                    selected.append(contract_pick)
        if not selected:
            return master.iloc[0:0].copy()
        return pd.concat(selected).drop_duplicates(subset="id").reset_index(drop=True)
 
    lo, hi = target_total
    current_n = n_contracts_per_bin[:]
    sample = build_once(current_n)
 
    attempt = 1
    while len(sample) < lo and attempt <= max_attempts:
        pool_sizes = {b: len(bin_pools[b]) for b in bin_order}
        maxed_out = all(n >= pool_sizes[b] for n, b in zip(current_n, bin_order))
        if maxed_out:
            print(f"NOTE: every bin is already drawing its FULL contract pool {pool_sizes} "
                  f"at max_rows_per_contract={max_rows_per_contract}, and that only yields "
                  f"{len(sample)} rows -- this is a structural ceiling in your data (probably "
                  f"fewer categories represented per contract than {max_rows_per_contract}). "
                  f"Raise max_rows_per_contract, lower target_total's floor, or add more "
                  f"length bins' worth of contracts to go further.")
            break
        current_n = [min(pool_sizes[b], max(n + 1, round(n * growth_factor)))
                     for n, b in zip(current_n, bin_order)]
        print(f"Attempt {attempt}: {len(sample)} rows < target floor {lo}. "
              f"Growing n_contracts_per_bin to {current_n} and retrying.")
        sample = build_once(current_n)
        attempt += 1
 
    if len(sample) > hi:
        is_protected = sample["contract_name"].isin(overall_longest)
        protected, rest = sample[is_protected], sample[~is_protected]
        n_trim = max(0, hi - len(protected))
        rest_trimmed = rest.sample(min(n_trim, len(rest)), random_state=seed) if len(rest) else rest
        sample = (pd.concat([protected, rest_trimmed])
                    .drop_duplicates(subset="id").reset_index(drop=True))
        print(f"Trimmed down to {len(sample)} rows to respect the target ceiling of {hi} "
              f"(force-included longest contracts were protected from trimming).")
 
    if not (lo <= len(sample) <= hi):
        print(f"NOTE: final sample size = {len(sample)}, still outside target range {target_total}.")
 
    missing = [c for c in overall_longest if c not in sample["contract_name"].values]
    if missing:
        print(f"WARNING: these force-included contracts are STILL not in the final sample: "
              f"{missing}. Reason for each:")
        for c in missing:
            length = contract_lengths_master.get(c, np.nan)
            if pd.isna(length):
                print(f"  '{c}': no valid n_tokens_contract in master (merge never matched it) "
                      f"-- run diagnose_contract_matching() / check_contract_present().")
                continue
            c_rows = master[master["contract_name"] == c]
            c_bins = c_rows["length_bin"].dropna().unique().tolist()
            print(f"  '{c}': n_tokens_contract={length}, length_bin={c_bins}, "
                  f"{len(c_rows)} rows in master, categories present: "
                  f"{c_rows['category'].unique().tolist()}. It matched fine but produced zero "
                  f"picks -- check whether c_rows above is unexpectedly empty for this contract.")
    else:
        print(f"Confirmed: all {len(overall_longest)} force-included longest contract(s) "
              f"are present in the final sample.")
 
    return sample


def sample_one_row_per_contract(df, seed=42, w_cell=1.0, w_bin_q=1.0, w_global_q=1.0):
    """
    Greedy, usage-balanced selection.
 
    For each contract, look at all its candidate rows and pick the one whose
    question / category / tier has been used the least so far:
 
        score = w_cell     * uses of (bin, category, tier)   -> variety inside each bin
              + w_bin_q    * uses of (bin, question)         -> avoid repeating a question in a bin
              + w_global_q * uses of question (all bins)     -> every question gets covered
 
    Lowest score wins; ties are broken randomly. Contracts are processed
    round-robin across the length bins so no single bin uses up the rare
    questions first.
    """
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)
 
    # each contract must belong to exactly one length bin
    assert df.groupby("contract_name")["length_bin"].nunique().max() == 1, \
        "a contract has rows in more than one bin"
 
    groups = {c: g for c, g in df.groupby("contract_name")}
 
    # shuffle contracts inside each bin, then interleave the bins
    per_bin = [rng.permutation(g["contract_name"].unique()).tolist()
               for _, g in df.groupby("length_bin")]
    order = [c for tup in zip_longest(*per_bin) for c in tup if c is not None]
 
    q_global, q_bin, cell_bin = Counter(), Counter(), Counter()
    picked = []
 
    for contract in order:
        cand = groups[contract]
        b = cand["length_bin"].iloc[0]
 
        scores = np.array([
            w_cell * cell_bin[(b, cat, tier)]
            + w_bin_q * q_bin[(b, q)]
            + w_global_q * q_global[q]
            for q, cat, tier in zip(cand["question"], cand["category"], cand["complexity_tier"])
        ], dtype=float)
        scores += rng.uniform(0, 1e-3, size=len(scores))  # random tie-break
 
        row = cand.iloc[int(scores.argmin())]
        picked.append(row)
 
        q_global[row["question"]] += 1
        q_bin[(b, row["question"])] += 1
        cell_bin[(b, row["category"], row["complexity_tier"])] += 1
 
    return pd.DataFrame(picked).reset_index(drop=True)


def coverage_report(sample: pd.DataFrame) -> None:
    print(f"Total rows: {len(sample)} | Distinct contracts: {sample['contract_name'].nunique()}")
    print("\nCells (length_bin x category):")
    print(pd.crosstab(sample["length_bin"], sample["category"]))
    print("\nCategory totals in sample:")
    print(sample["category"].value_counts())
    print("\ncomplexity_tier distribution (recorded, not sampling-forced -- check it's not "
          "badly skewed):")
    print(sample["complexity_tier"].value_counts())
    print("\nRows per length bin:")
    print(sample["length_bin"].value_counts().sort_index())


def collapse_rare_categories(sample: pd.DataFrame, min_count: int = 12,
                             other_label: str = "Other") -> pd.DataFrame:
    """
    For statistical modeling only (keep the original `category` column for
    descriptive reporting). Categories with fewer than `min_count` rows IN
    THIS SAMPLE get folded into "Other" -- with 100-200 rows, General
    Information/Knowledge/Remedies will typically land here. This avoids
    near-empty or empty model x category cells in the GEE interaction model,
    which otherwise risk unstable estimates or quasi-separation.
    """
    sample = sample.copy()
    counts = sample["category"].value_counts()
    rare = counts[counts < min_count].index
    sample["category_grp"] = sample["category"].where(~sample["category"].isin(rare), other_label)
    if len(rare):
        print(f"Collapsed into '{other_label}' for modeling: {list(rare)} "
              f"(counts: {counts[rare].to_dict()})")
    return sample



def build_query(question_text: str, options: list) -> str:
    letters = [chr(ord("A") + i) for i in range(len(options))]
    options_block = "\n".join(f"{l}. {opt}" for l, opt in zip(letters, options))
    prompt = f"""You are analyzing a merger agreement.

            Question:
            {question_text}

            Answer choices:
            {options_block}

            Answer the question using the merger agreement provided to you.

            Select the single answer choice that is best supported by the agreement.

            Return only the answer choice."""

    return prompt
    


def run_maud_152_sample_pipeline():
    maud_data_dir = get_maud_data_dir()
    CONTRACTS_DIR = f"""{maud_data_dir}/contracts"""
    with open("maud/question_full_text_map.json", "r", encoding="utf-8") as f:
        question_full_text_map = json.load(f)
    with open("maud/question_answer_map_unique.json", "r", encoding="utf-8") as f:
        question_answer_map_unique = json.load(f)

    df = get_main_def(split="train")
    contract_token_counts_path = "maud_contract_lengths.json"
    contract_bins = bin_contracts_by_length(contract_token_counts_path, n_bins=4)
    master = build_master_table(add_complexity_features(df), contract_bins)
    sample = sample_one_row_per_contract(master, seed=RANDOM_SEED)


    for row in sample.itertuples(index=True):
        full_question = question_full_text_map[row.question]
        answer_options = question_answer_map_unique[row.question]
        with open(f"{CONTRACTS_DIR}/{row.contract_name}.txt", "r", encoding="utf-8-sig") as f:
            contract_text = f.read()
        
        task_prompt = build_query(question_text=full_question, options=answer_options)
        print("=== 1. VANILLA ===")
        print(run_vanilla("maud_152_samples", f"{row.Index}", contract_text, TASK_SPEC=task_prompt, experiment="maud_152_samples_per_contract_batch_run"))
        
        print("\n=== 3. RLM CUSTOM SYSTEM PROMPT ===")
        print(run_rlm_customized("maud_152_samples", f"{row.Index}", contract_text, TASK_SPEC=task_prompt, experiment="maud_152_samples_per_contract_batch_run"))



# ------------------------------------------------
# generate json files for question mappings
# -------------------------------------------------
def extract_question_answer_options(df, normalize=True):
    """
    Returns:
      summary   -- one row per question: n_rows, n_unique_answers, answer_options (sorted list)
      options   -- dict {question: [answer options...]}, ready to json.dump
    """
    QUESTION_COL = "question"
    ANSWER_COL = "answer"
    work = df[[QUESTION_COL, ANSWER_COL]].copy()
 
    if normalize:
        # strip whitespace so "Yes" and "Yes " don't count as two different answers;
        # keep original case though, since e.g. "CoR"/"FTR" abbreviations are case-sensitive
        work[ANSWER_COL] = work[ANSWER_COL].astype(str).str.strip()
 
    grouped = work.groupby(QUESTION_COL)[ANSWER_COL]
 
    summary = grouped.agg(
        n_rows="count",
        n_unique_answers="nunique",
        answer_options=lambda s: s.unique(),
    ).reset_index()
    summary = summary.sort_values(QUESTION_COL).reset_index(drop=True)
 
    options = dict(zip(summary[QUESTION_COL], summary["answer_options"]))
    return summary, options


def fix_answer_options(data: dict, majority_threshold: float = 0.5):
    """
    data: dict of {question: [answer strings]}
    Returns (fixed, report):
      fixed  -- same shape as `data`, with multi-select questions' answer
                lists replaced by their atomic tag vocabulary
      report -- {question: {"status", "n_before", "n_after", "frac_multi_token"}}
                status is one of:
                  "fixed_multiselect" -- auto-replaced with atomic tags
                  "check_manually"    -- some but not most answers have commas;
                                         left unchanged, worth a manual look
                  "left_unchanged"    -- looks like a genuine single-choice question
    """
    fixed, report = {}, {}
 
    for question, answers in data.items():
        answers = [a.strip() for a in answers if isinstance(a, str) and a.strip()]
        tokenized = [[t.strip() for t in a.split(",")] for a in answers]
        frac_multi = sum(1 for toks in tokenized if len(toks) > 1) / len(answers) if answers else 0.0
 
        if frac_multi >= majority_threshold:
            vocab = sorted({t for toks in tokenized for t in toks})
            status = "fixed_multiselect"
            fixed[question] = vocab
        elif frac_multi > 0:
            status = "check_manually"
            fixed[question] = answers
        else:
            status = "left_unchanged"
            fixed[question] = answers
 
        report[question] = {
            "status": status,
            "n_before": len(answers),
            "n_after": len(fixed[question]),
            "frac_multi_token": round(frac_multi, 2),
        }
 
    return fixed, report
 
 
def fix_answer_options_file(in_path: str, out_path: str, majority_threshold: float = 0.5):
    """Load JSON from in_path, fix it, save to out_path, print a short summary."""
    data = json.load(open(in_path))
    fixed, report = fix_answer_options(data, majority_threshold)
 
    json.dump(fixed, open(out_path, "w"), indent=2)
 
    for status in ("fixed_multiselect", "check_manually"):
        rows = [(q, r) for q, r in report.items() if r["status"] == status]
        if rows:
            print(f"\n{status} ({len(rows)}):")
            for q, r in rows:
                print(f"  {q}: {r['n_before']} -> {r['n_after']} options "
                      f"(frac_multi_token={r['frac_multi_token']})")
 
    return fixed, report

#---------------------------------------------------



if __name__ == "__main__":
    df = get_main_def(split="train")

    contract_token_counts_path = "maud_contract_lengths.json"
    contract_bins = bin_contracts_by_length(contract_token_counts_path, n_bins=4)

    master = build_master_table(add_complexity_features(df), contract_bins)
    support_table = question_support_table(master)
    # with pd.option_context("display.max_rows", None):
    #     print(support_table)

    # EDA before locking the design
    # check_axis_confounding(master)

    # Sample (stratifies by length_bin x category; complexity_tier is
    #    recorded on every row for analysis, not forced as a sampling cell)
    sample = sample_one_row_per_contract(master, seed=RANDOM_SEED)
    with pd.option_context("display.max_rows", None):
        print(sample.sort_values(by="n_tokens_contract").reset_index(drop=True)["complexity_tier"])


    # print(options)
    # with open("maud/question_answer_map.json", "w", encoding="utf-8") as f:
    #     json.dump(options, f, indent=4, default=lambda o: o.to_pylist() if hasattr(o, "to_pylist") else (o.as_py() if hasattr(o, "as_py") else str(o)))


    # coverage_report(sample)

    # 6. Answer key from the FULL table
    # answer_key = build_answer_key(df)

    # 7-8. Run eval (plug in your vanilla_call / rlm_call / contract_texts)
    # res_df = run_eval(sample, answer_key, vanilla_call, rlm_call, contract_texts)

    # 10. Analyze
    # mcnemar_overall(res_df)
    # mcnemar_by_group(res_df, "length_bin")
    # pooled_interaction_model(res_df)
    # bootstrap_gap_ci(res_df, group_col="length_bin")
    