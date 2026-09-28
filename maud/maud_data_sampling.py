"""
MAUD RLM-vs-Vanilla evaluation pipeline
========================================

Goal: decide, with 100-200 examples, whether Recursive Language Models (RLM;
Zhang, Kraska & Khattab, 2025, arXiv:2512.24601) outperform a vanilla LLM call
as (a) full-contract context length and (b) task complexity grow, on MAUD
(Wang et al., 2023, EMNLP).

Sections:
  1. Load data + sanity-check schema
  2. Contract-level length bins (you already built this on the 152 contracts)
  3. Complexity feature engineering (answer-option count, excerpt length)
  4. EDA: check whether your stratification axes are confounded before locking design
  5. Stratified sampler (length_bin x category), with complexity_tier recorded
     as a covariate rather than a forced sampling stratum (see note in section 5)
  6. Per-question answer-option key (built from the FULL ~39k table, not the sample)
  7. Prompt templates (identical query for both arms -- only the context-handling
     harness should differ between vanilla and RLM)
  8. Evaluation loop skeleton (plug in your own model / RLM callers)
  9. Answer parsing + scoring
 10. Statistical analysis: McNemar (per cell), pooled GEE logistic interaction
     model (primary inferential tool -- more powered than per-cell tests at n=100-200),
     bootstrap CIs on the accuracy gap

Schema assumed (matches theatticusproject/maud / nguha-legalbench derived tables):
  contract_name, text, text_type, category, question, subquestion, answer, label, id
If your columns differ, the loader below will warn you rather than silently failing.
"""

import json
import re
from collections import Counter
from itertools import zip_longest

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency
from statsmodels.stats.contingency_tables import mcnemar

from utils.maud_utils import get_main_def

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
 
    with pd.option_context("display.max_rows", None):
        print(sample)
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


# ---------------------------------------------------------------------------
# 6. ANSWER-OPTION KEY -- build from the FULL table so rare options aren't lost
# ---------------------------------------------------------------------------
def build_answer_key(full_df: pd.DataFrame) -> dict:
    return {q: sorted(grp["answer"].dropna().unique().tolist())
            for q, grp in full_df.groupby("question")}


# ---------------------------------------------------------------------------
# 7. PROMPTS -- identical query text for both arms; only the harness differs
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an experienced M&A attorney reviewing a merger agreement. "
    "Answer strictly and only from the contract text you are given. "
    "The clause relevant to the question is not always near the start of the "
    "document, and some questions require connecting language from more than "
    "one section (for example, a defined term in one article and its use "
    "elsewhere). Do not rely on outside knowledge of market-standard terms; "
    "if this contract's language differs from typical practice, follow the "
    "contract as written."
)


def build_query(question_text: str, options: list) -> str:
    letters = [chr(ord("A") + i) for i in range(len(options))]
    options_block = "\n".join(f"{l}. {opt}" for l, opt in zip(letters, options))
    return (
        f"Deal point question: {question_text}\n\n"
        f"Answer options:\n{options_block}\n\n"
        "Instructions: choose exactly one option based only on the contract "
        "text. You may briefly explain your reasoning, but the LAST line of "
        "your response must be exactly in the form:\nAnswer: <LETTER>"
    )


def build_vanilla_prompt(context_text: str, question_text: str, options: list) -> list:
    query = build_query(question_text, options)
    user_msg = f"[CONTRACT TEXT]\n{context_text}\n[END CONTRACT TEXT]\n\n{query}"
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}]


def build_rlm_inputs(context_text: str, question_text: str, options: list) -> dict:
    """
    For the RLM harness: `context_text` is loaded as the `context` variable in
    the REPL environment; `query` is the top-level prompt given to the root LM,
    which then recursively decomposes/queries `context` (calling itself over
    programmatic snippets). Keep `query` byte-for-byte identical to the
    vanilla arm's query -- the experiment is only valid if the sole difference
    between arms is the context-handling harness, not the wording of the ask.
    """
    return {
        "context": context_text,
        "query": SYSTEM_PROMPT + "\n\n" + build_query(question_text, options),
    }


# ---------------------------------------------------------------------------
# 8. EVALUATION LOOP -- plug in your own model / RLM call functions
# ---------------------------------------------------------------------------
def run_eval(sample: pd.DataFrame, answer_key: dict,
             vanilla_call, rlm_call, contract_texts: dict) -> pd.DataFrame:
    """
    vanilla_call(messages: list[dict]) -> str
    rlm_call(query: str, context: str) -> str
    contract_texts: dict mapping contract_name -> FULL raw contract string
                    (use the full document, not the MAUD excerpt, if your goal
                    is to test long-context handling -- see the note above the
                    prompt section in the chat reply).
    """
    results = []
    for _, row in sample.iterrows():
        options = answer_key[row["question"]]
        context_text = contract_texts[row["contract_name"]]

        vanilla_msgs = build_vanilla_prompt(context_text, row["question"], options)
        rlm_inputs = build_rlm_inputs(context_text, row["question"], options)

        vanilla_out = vanilla_call(vanilla_msgs)
        rlm_out = rlm_call(rlm_inputs["query"], rlm_inputs["context"])

        record = row.to_dict()
        record["_answer_options"] = options
        record["vanilla_output"] = vanilla_out
        record["rlm_output"] = rlm_out
        results.append(record)

    res_df = pd.DataFrame(results)
    res_df["vanilla_correct"] = res_df.apply(lambda r: score_row(r, "vanilla_output"), axis=1)
    res_df["rlm_correct"] = res_df.apply(lambda r: score_row(r, "rlm_output"), axis=1)
    return res_df


# ---------------------------------------------------------------------------
# 9. PARSING + SCORING
# ---------------------------------------------------------------------------
def parse_answer_letter(model_output: str):
    matches = re.findall(r"Answer:\s*([A-Z])\b", str(model_output).strip())
    return matches[-1] if matches else None


def score_row(row, output_col: str):
    letter = parse_answer_letter(row[output_col])
    if letter is None:
        return np.nan
    idx = ord(letter) - ord("A")
    options = row["_answer_options"]
    if not (0 <= idx < len(options)):
        return np.nan
    return int(options[idx] == row["answer"])


# ---------------------------------------------------------------------------
# 10. STATISTICAL ANALYSIS
# ---------------------------------------------------------------------------
def mcnemar_overall(res_df: pd.DataFrame):
    both = ((res_df.vanilla_correct == 1) & (res_df.rlm_correct == 1)).sum()
    v_only = ((res_df.vanilla_correct == 1) & (res_df.rlm_correct == 0)).sum()
    r_only = ((res_df.vanilla_correct == 0) & (res_df.rlm_correct == 1)).sum()
    neither = ((res_df.vanilla_correct == 0) & (res_df.rlm_correct == 0)).sum()
    table = [[both, v_only], [r_only, neither]]
    result = mcnemar(table, exact=True)
    print(f"n={len(res_df)}  Vanilla acc={res_df.vanilla_correct.mean():.1%}  "
          f"RLM acc={res_df.rlm_correct.mean():.1%}  "
          f"(RLM-only-correct={r_only}, Vanilla-only-correct={v_only})  "
          f"McNemar exact p={result.pvalue:.4f}")
    return result


def mcnemar_by_group(res_df: pd.DataFrame, group_col: str):
    """Treat these as DESCRIPTIVE, not the primary test -- at n~15-30/cell,
    McNemar alone is only well-powered for large (>~25pt) gaps. See the pooled
    model below for the properly-powered interaction test."""
    for g, gdf in res_df.groupby(group_col, observed=True):
        print(f"\n--- {group_col} = {g} (n={len(gdf)}) ---")
        mcnemar_overall(gdf)


def pooled_interaction_model(res_df: pd.DataFrame, min_category_count: int = 12):
    """
    Primary inferential tool: reshape to long format (one row per
    model x question-instance) and fit a GEE logistic model clustered by
    contract+question, testing whether the vanilla-vs-RLM gap depends on
    length_bin / category / complexity_tier. This pools information across all
    cells rather than testing each cell in isolation, which is what makes it
    usable at n=100-200 (see the power simulation for why per-cell tests alone
    need much larger n).

    Uses category_grp (rare categories collapsed to "Other") rather than the
    raw 7-level category, since at n=100-200 the rare categories (General
    Information, Knowledge, Remedies) will typically have too few rows for a
    stable model x category interaction term on their own.
    """
    res_df = collapse_rare_categories(res_df, min_count=min_category_count)
    long_df = pd.concat([
        res_df.assign(model="vanilla", correct=res_df.vanilla_correct),
        res_df.assign(model="rlm", correct=res_df.rlm_correct),
    ], ignore_index=True)
    long_df["model"] = pd.Categorical(long_df["model"], categories=["vanilla", "rlm"])
    long_df["cluster_id"] = (long_df["contract_name"].astype(str) + "_" +
                              long_df["question"].astype(str))

    model = smf.gee(
        "correct ~ model * length_bin + model * category_grp + model * complexity_tier",
        groups="cluster_id", data=long_df, family=sm.families.Binomial()
    )
    fit = model.fit()
    print(fit.summary())
    return fit


def bootstrap_gap_ci(res_df: pd.DataFrame, group_col=None, n_boot=5000, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    def one_group_ci(gdf):
        idx_all = gdf.index.to_numpy()
        gaps = np.empty(n_boot)
        for i in range(n_boot):
            samp = rng.choice(idx_all, size=len(idx_all), replace=True)
            s = gdf.loc[samp]
            gaps[i] = s.rlm_correct.mean() - s.vanilla_correct.mean()
        return gaps.mean(), np.percentile(gaps, [2.5, 97.5])

    if group_col is None:
        mean_gap, ci = one_group_ci(res_df)
        print(f"Overall RLM-Vanilla accuracy gap: {mean_gap:+.1%}  95% CI [{ci[0]:+.1%}, {ci[1]:+.1%}]")
    else:
        for g, gdf in res_df.groupby(group_col, observed=True):
            mean_gap, ci = one_group_ci(gdf)
            print(f"{group_col}={g} (n={len(gdf)}): gap={mean_gap:+.1%}  95% CI [{ci[0]:+.1%}, {ci[1]:+.1%}]")


# ---------------------------------------------------------------------------
# EXAMPLE USAGE
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Load
    df = get_main_def(split="train")

    # 2. Your existing contract-length quantiles (reuse your own Series here)
    contract_token_counts_path = "maud_contract_lengths.json"
    contract_bins = bin_contracts_by_length(contract_token_counts_path, n_bins=4)

    # 3. Features
    master = build_master_table(add_complexity_features(df), contract_bins)
    support_table = question_support_table(master)
    with pd.option_context("display.max_rows", None):
        print(support_table)

    # 4. EDA before locking the design
    # check_axis_confounding(master)

    # 5. Sample (stratifies by length_bin x category; complexity_tier is
    #    recorded on every row for analysis, not forced as a sampling cell)
    sample = sample_one_row_per_contract(master, seed=RANDOM_SEED)
    with pd.option_context("display.max_rows", None):
        print(sample.sort_values(by="n_tokens_contract").reset_index(drop=True))

    with pd.option_context("display.max_colwidth", 20):
        for (length_bin, complexity_tier, category), group in sample.groupby(
            ["length_bin", "complexity_tier", "category"], observed=True
            ):
            print(f"\n--- Length Bin: {length_bin}, Complexity Tier: {complexity_tier}, Category: {category} ---")
            print(group[["contract_name", "complexity_tier", "category"]])

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
    