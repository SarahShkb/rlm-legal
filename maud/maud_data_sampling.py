import difflib
import json
import numbers
import re
from collections import Counter
from itertools import zip_longest
from shutil import which
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from legal_tools.run_tools import build_tools
from legal_tools.tools_utils import ToolLog
from llm.rlm import run_rlm_customized
from llm.vanila import run_vanilla
from utils.maud_utils import get_main_def, get_maud_data_dir

RANDOM_SEED = 42


# ------------------------------------------------
# Helper functions
# ------------------------------------------------

def _options_resolver(options_by_question: dict[str, list]):
    """
    Returns get_options(question) -> list of options.
    Matches the question text exactly first, then ignoring case/whitespace,
    then ignoring punctuation as well (a loose key shared by two different
    questions is not used, to avoid guessing).
    """
    norm_map = {_norm(q): opts for q, opts in options_by_question.items()}
    loose_map: dict[str, list] = {}
    clashes = set()
    for q, opts in options_by_question.items():
        k = _loose(q)
        if k in loose_map and loose_map[k] is not opts:
            clashes.add(k)
        loose_map[k] = opts
    for k in clashes:
        loose_map.pop(k)
 
    def get_options(question: str) -> list:
        if question in options_by_question:
            return options_by_question[question]
        opts = norm_map.get(_norm(question)) or loose_map.get(_loose(question))
        if opts is None:
            close = difflib.get_close_matches(str(question), list(options_by_question), n=1, cutoff=0.6)
            hint = f" Closest key in the options JSON: {close[0]!r}." if close else ""
            raise KeyError(f"No options found for question {question!r}.{hint}")
        return opts
 
    return get_options


def _okey(k: any) -> any:
    """Normalise a question key (string or tuple of strings) for override lookup."""
    if isinstance(k, tuple):
        return tuple(_okey(x) for x in k)
    return _norm(k) if isinstance(k, str) else k


def _multi_select_info(
    df: pd.DataFrame,
    options_by_question: dict[str, list],
    question_col: str,
    gold_col: str,
    overrides: Optional[dict],
    reference_df: Optional[pd.DataFrame],
    key_cols: Optional[list],
) -> pd.DataFrame:
    """
    Shared engine (private). Returns a DataFrame aligned POSITIONALLY with the rows
    of `df` (fresh RangeIndex) with the columns: n_options, n_gold, is_multi, source.
    Nothing is added to `df` itself.
    """
    get_options = _options_resolver(options_by_question)
    key_cols = key_cols or [question_col]
 
    def keys_of(frame: pd.DataFrame) -> list:
        return list(zip(*[frame[c] for c in key_cols]))
 
    def n_gold_of(frame: pd.DataFrame) -> list:
        return [
            len(gold_to_letters(g, get_options(q)))
            for g, q in zip(frame[gold_col], frame[question_col])
        ]
 
    n_options = [len(get_options(q)) for q in df[question_col]]
    n_gold = n_gold_of(df)
 
    # Most gold answers ever seen per question (this df + optional reference_df)
    max_gold: dict = {}
    evidence = [(keys_of(df), n_gold)]
    if reference_df is not None:
        evidence.append((keys_of(reference_df), n_gold_of(reference_df)))
    for keys, counts in evidence:
        for key, n in zip(keys, counts):
            max_gold[key] = max(max_gold.get(key, 0), n)
 
    if overrides and key_cols == [question_col]:
        for k in overrides:      # a typo in an override key fails loudly, with a hint
            get_options(k)
    manual = {_okey(k): bool(v) for k, v in (overrides or {}).items()}
 
    flags, sources = [], []
    for key in keys_of(df):
        ov = manual.get(_okey(key[0] if len(key) == 1 else key))
        if ov is not None:
            if not ov and max_gold[key] > 1:
                raise ValueError(
                    f"Override says single-answer but the data has {max_gold[key]} gold "
                    f"answers for: {key[0] if len(key) == 1 else key!r}"
                )
            flags.append(ov)
            sources.append("manual")
        elif max_gold[key] > 1:
            flags.append(True)
            sources.append("observed_multiple_gold")
        else:
            flags.append(False)
            sources.append("no_multi_observed")
 
    return pd.DataFrame(
        {"n_options": n_options, "n_gold": n_gold, "is_multi_select": flags, "source": sources}
    )


def letters_for(n: int) -> list[str]:
    return [chr(ord("A") + i) for i in range(n)] 


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().casefold()



def _loose(s) -> str:
    """
    Loose key: also ignores all punctuation and whitespace, so
    '(e.g., hybrid standard)', '(e.g. hybrid standard)' and '(e.g.,hybrid standard)'
    all compare equal.
    """
    return re.sub(r"[\W_]+", "", _norm(s))
 
_SEP_RE = re.compile(r"([,;\n])")

def _build_lookups(options: list) -> tuple[dict[str, str], dict[str, str]]:
    """(exact_lookup, loose_lookup): key -> letter. Ambiguous loose keys are dropped."""
    letters = letters_for(len(options))
    exact = {_norm(o): l for o, l in zip(options, letters)}
    loose: dict[str, str] = {}
    clashes = set()
    for o, l in zip(options, letters):
        k = _loose(o)
        if k in loose and loose[k] != l:
            clashes.add(k)
        loose[k] = l
    for k in clashes:
        loose.pop(k)
    return exact, loose


def _match(cand: str, exact: dict, loose: dict) -> Optional[str]:
    """Exact match first, then punctuation/whitespace-insensitive match."""
    return exact.get(_norm(cand)) or loose.get(_loose(cand))


def _text_to_letters(s: str, exact: dict, loose: dict) -> list[str]:
    """
    Map a gold string to option letters. The string may be one option, or
    several options joined by ',' / ';' / newline.
 
    Option texts can themselves contain commas (e.g. "... (e.g., hybrid
    standard)"), so instead of a blind split we search for a segmentation
    where EVERY segment is a known option, trying longer segments first
    (so the whole string is tried as a single option before anything is split).
    """
    s = s.strip()
    whole = _match(s, exact, loose)
    if whole:
        return [whole]
 
    pieces = _SEP_RE.split(s)            # text, sep, text, sep, text, ...
    texts, seps = pieces[0::2], pieces[1::2]
    n = len(texts)
 
    def segment(i: int) -> Optional[list[str]]:
        if i == n:
            return []
        if not texts[i].strip():         # empty piece, e.g. trailing comma
            return segment(i + 1)
        for j in range(n, i, -1):        # longest span first
            cand = "".join(texts[k] + seps[k] for k in range(i, j - 1)) + texts[j - 1]
            letter = _match(cand, exact, loose)
            if letter:
                rest = segment(j)
                if rest is not None:
                    return [letter] + rest
        return None
 
    result = segment(0)
    if not result:
        # Diagnostic: strip every recognisable option out of the (normalised)
        # gold string and show what is left over.
        leftover = _loose(s)
        for k in sorted(loose, key=len, reverse=True):
            leftover = leftover.replace(k, "")
        hint = f" Unmatched text (normalised): {leftover!r}." if leftover else ""
        raise ValueError(f"Gold answer {s!r} could not be matched to the options.{hint}")
    return result

# ------------------------------------------------


# ------------------------------------------------
# Preparing the main `sampple` DataFrame
# ------------------------------------------------

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


def add_is_multi_select(
    df: pd.DataFrame,
    options_by_question: dict[str, list],
    question_col: str = "question",
    gold_col: str = "gold",
    overrides: Optional[dict] = None,
    reference_df: Optional[pd.DataFrame] = None,
    key_cols: Optional[list] = None,
    out_col: str = "is_multi_select",
) -> pd.DataFrame:
    
    # adds the column `is_multi_select` to the DataFrame,
    # based on the gold answers and the options_by_question mapping.
    info = _multi_select_info(
        df, options_by_question, question_col, gold_col, overrides, reference_df, key_cols
    )
    df = df.copy()
    df[out_col] = info["is_multi_select"].to_numpy()
    return df

 
def get_tier(options: list, multi_select: bool = False) -> str:
    if multi_select:
        return "multi"
    return "binary" if len(options) == 2 else "single"


def load_options(path: str) -> dict[str, list[str]]:
    # load the dictionary of question as key and its array of options as value
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Options JSON must be an object: {question: [options]}")
    return {q: list(opts) for q, opts in data.items()}
 

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

# Method 1 (I am not currently using it because of complexity)
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


# Method 2 (simpler, greedy)
def sample_one_row_per_contract(df, seed=42, w_cell=1.0, w_bin_q=1.0, w_global_q=1.0):
    
    # Greedy, usage-balanced selection:
    # For each contract, look at all its candidate rows and pick the one whose
    # question / category / tier has been used the least so far:
 
    #    score = w_cell     * uses of (bin, category, tier)   -> variety inside each bin
    #          + w_bin_q    * uses of (bin, question)         -> avoid repeating a question in a bin
    #         + w_global_q * uses of question (all bins)     -> every question gets covered
 
    # Lowest score wins; ties are broken randomly. Contracts are processed
    # round-robin across the length bins so no single bin uses up the rare
    # questions first.
    
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


#----------------------------------------------------------------------------



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

# ---------------------------------------------------



#------------------------------------------------
# RUNNING
# ------------------------------------------------
 

def build_query(
    question_text: str,
    options: list,
    multi_select: bool = False,
    allow_none: bool = False,
) -> str:
    ROLE = "You are an expert M&A attorney analyzing a merger agreement."
    letters = letters_for(len(options))
    options_block = "\n".join(f"{l}. {o}" for l, o in zip(letters, options))
    tier = get_tier(options, multi_select)
 
    if tier == "binary":
        task = (
            "This is a binary question: exactly ONE of the two answer choices is correct.\n"
            "Select the choice that is best supported by the merger agreement. "
        )
        fmt = (
            "Respond with exactly one line in this format, and nothing else:\n"
            f"Answer: <letter>   (where <letter> is {letters[0]} or {letters[1]})"
        )
 
    elif tier == "single":
        task = (
            f"This is a single-answer question: exactly ONE of the {len(options)} "
            "answer choices is correct.\n"
            "Select the choice that is best supported by the merger agreement. "
            "When choices differ by degree or scope, pick the one that matches the "
            "agreement's precise language rather than a broader or narrower reading."
        )
        fmt = (
            "Respond with exactly one line in this format, and nothing else:\n"
            f"Answer: <letter>   (where <letter> is one of {', '.join(letters)})"
        )
 
    else:  # multi
        none_rule = (
            "\nIf none of the choices apply, respond with: Answer: NONE"
            if allow_none
            else "\nAt least one choice must be selected."
        )
        task = (
            "This is a select-all-that-apply question: ONE OR MORE answer choices "
            "may be correct.\n"
            "Evaluate each choice independently against the merger agreement. "
            "Select every choice that the agreement supports, and no others. "
            "Do not select a choice merely because it is plausible."
        )
        fmt = (
            "Respond with exactly one line in this format, and nothing else:\n"
            "Answer: <letters>   (comma-separated, alphabetical order, "
            "e.g. \"Answer: A, C\"; valid letters: " + ", ".join(letters) + ")"
            + none_rule
        )
 
    # No indentation inside the string: leading spaces would leak into the prompt.
    return f"""{ROLE}
 
    Question:
    {question_text}
    
    Answer choices:
    {options_block}
    
    Instructions:
    {task}
    Base your answer only on the merger agreement provided to you, not on outside knowledge.
    
    {fmt}"""


def gold_to_letters(gold, options: list) -> list[str]:
    # converts the answer to the corresponding letter(s) for the given options
    
    # gold may be: an option text, several option texts joined by commas
    #             ("X, Y"), a list of option texts (each item may itself be
    #             comma-joined), an int index, or a list of int indices.
    # Returns sorted letters, e.g. ["A"] or ["A", "C"].
    
    letters = letters_for(len(options))
    exact, loose = _build_lookups(options)
    items = [gold] if isinstance(gold, (str, numbers.Integral, float)) else list(gold)
    if pd.isna(gold):
        items = ['None']

    out: list[str] = []
    for g in items:
        if isinstance(g, numbers.Integral):
            if not 0 <= int(g) < len(options):
                raise ValueError(f"Gold index {g} out of range for {len(options)} options")
            out.append(letters[int(g)])
        else:
            try:
                out.extend(_text_to_letters(str(g), exact, loose))
            except ValueError as e:
                raise ValueError(f"{e} Options: {options}") from None
    return sorted(set(out))


def get_sample_rows_for_maud_152(split: str = "train") -> list:
    with open("maud/dicts/question_answer_map_unique.json", "r", encoding="utf-8") as f:
        question_answer_map_unique = json.load(f)
    df = get_main_def(split=split)
    contract_token_counts_path = "maud/dicts/maud_contract_lengths.json"
    contract_bins = bin_contracts_by_length(contract_token_counts_path, n_bins=4)
    master = build_master_table(add_complexity_features(df), contract_bins)
    options_by_question = load_options("maud/dicts/question_answer_map_unique.json")
    master = add_is_multi_select(
        master, question_col="question", options_by_question=options_by_question, gold_col="answer"
    )
    sample = sample_one_row_per_contract(master, seed=RANDOM_SEED)
    # with pd.option_context("display.max_rows", None):
    #     print(sample[["question","answer", "is_multi_select"]])
    sample["gold_letters"] = [
        gold_to_letters(
            g,
            question_answer_map_unique[row.question]
        )
        for sample_idx, (g, (_, row)) in enumerate(
            zip(sample["answer"], sample.iterrows()),
            start=1
        )
    ]
    samples_sorted = sample.sort_values(by="n_tokens_contract").reset_index(drop=True)
    # with pd.option_context("display.max_rows", None, "display.max_colwidth", 20):
    #     print(samples_sorted[["answer", "complexity_tier", "is_multi_select", "gold_letters"]])
    # with pd.option_context("display.max_rows", None, "display.max_colwidth", 30):
    #     print(samples_sorted[["question", "answer", "gold_letters"]])
    return samples_sorted


def run_maud_152_sample_pipeline(tools: set[str] | None=None) -> None:
    maud_data_dir = get_maud_data_dir()
    CONTRACTS_DIR = f"""{maud_data_dir}/contracts"""
    with open("maud/dicts/question_full_text_map.json", "r", encoding="utf-8") as f:
        question_full_text_map = json.load(f)
    with open("maud/dicts/question_answer_map_unique.json", "r", encoding="utf-8") as f:
        question_answer_map_unique = json.load(f)


    # change this if the pipeline crashed and 
    # you want to resume from a specific row index
    limit_index = -1 

    samples_sorted = get_sample_rows_for_maud_152(split="train")
    print(samples_sorted[["contract_name", "question", "answer", "gold_letters", "n_tokens_contract"]])

    for row in samples_sorted.itertuples(index=True):
        if row.Index < limit_index:
            continue
        full_question = question_full_text_map[row.question]
        answer_options = question_answer_map_unique[row.question]
        is_multi_select = row.is_multi_select
        with open(f"{CONTRACTS_DIR}/{row.contract_name}.txt", "r", encoding="utf-8-sig") as f:
            contract_text = f.read()

        print(f"=========ROW: {row.Index}============")
        
        task_prompt = build_query(question_text=full_question, multi_select=is_multi_select, options=answer_options)
        if tools is None:
            print("=== VANILLA ===")
            vanilla_res = run_vanilla(
                dataset="maud_152_samples",
                row_id=f"{row.contract_name}_{row.Index}",
                contract_text=contract_text,
                TASK_SPEC=task_prompt,
                experiment="maud_152_samples_per_contract_batch_run",
            )
            print(vanilla_res)


        print("\n=== RLM CUSTOM SYSTEM PROMPT ===")
        if tools is not None:
            log = ToolLog()
            tools = build_tools(contract_text, tools, log) if which else None
        rlm_res = run_rlm_customized(
            dataset="maud_152_samples",
            row_id=f"{row.contract_name}_{row.Index}",
            contract_text=contract_text,
            TASK_SPEC=task_prompt,
            experiment="maud_152_samples_per_contract_batch_run",
            max_iterations=8,
            tools=tools
        )
        print(rlm_res)
        

    