def score_row(row, output_col: str):
    letter = parse_answer_letter(row[output_col])
    if letter is None:
        return np.nan
    idx = ord(letter) - ord("A")
    options = row["_answer_options"]
    if not (0 <= idx < len(options)):
        return np.nan
    return int(options[idx] == row["answer"])


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
