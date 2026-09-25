import os

import pandas as pd
import tiktoken

from maud.maud_utils import get_maud_data_dir

DEEKSEEK_MODEL = os.environ.get('DEEPSEEK_MODEL_NAME')

def get_nominated_maud_raw_data(main_df: pd.DataFrame) -> pd.DataFrame:
    enc = tiktoken.encoding_for_model(DEEKSEEK_MODEL)
    main_df['n_tokens'] = main_df['text'].apply(lambda t: len(enc.encode(str(t))))

    q25, q50, q75 = main_df['n_tokens'].quantile([.25, .5, .75])
    bins = [0, q25, q50, q75, float("inf")]
    main_df['length_bucket'] = pd.cut(
        main_df['n_tokens'], bins=bins,
        labels=["short", "medium", "long", "very_long"]
    )
    print(main_df['length_bucket'].value_counts())

    primary_sample = main_df.groupby('length_bucket', group_keys=False).apply(
        lambda g: g.sample(n=min(25, len(g)), random_state=42)
    )
    primary_sample.to_csv("batch100_length_stratified.csv", index=False)

    # --- Step 2: secondary, separate robustness set (rare answers) ---
    rare_df = df[df['text_type'] == RARE_LABEL].copy()
    rare_sample = rare_df.sample(n=min(25, len(rare_df)), random_state=42)
    rare_sample.to_csv("bonus25_rare_answers.csv", index=False)

    # --- Step 3: leakage check between the two sets ---
    overlap_keys = set(zip(primary_sample['contract_name'], primary_sample['question'])) & \
                set(zip(rare_df['contract_name'], rare_df['question']))
    print(f"Overlapping contract+question pairs between primary and rare sets: {len(overlap_keys)}")
