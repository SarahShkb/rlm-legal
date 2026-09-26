from __future__ import annotations

import os
import re
import subprocess
import zipfile

import pandas as pd

# -----------------------------------------------
# Data configuration
# -----------------------------------------------
REPO_URL = "https://github.com/The-Atticus-Project/maud.git"

# --------------------------------------------------------------------------- #
# Get the data
# --------------------------------------------------------------------------- #
def get_maud_data_dir(work_dir: str = "maud_workdir", force: bool = False) -> str:

    os.makedirs(work_dir, exist_ok=True)
    repo_dir = os.path.join(work_dir, "maud_repo")
    data_dir = os.path.join(work_dir, "data")

    # check if the data already exists
    if os.path.exists(data_dir) and not force:
        return data_dir

    if not os.path.exists(repo_dir):
        print("Cloning MAUD repo...")
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, repo_dir], check=True
        )

    # unzip the data.zip file from the repo
    zip_path = os.path.join(repo_dir, "data.zip")
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(work_dir)  # zip already contains a top-level "data/" folder

    return data_dir


def get_main_def(split: str = "train") -> pd.DataFrame:
    data_dir = get_maud_data_dir()
    df = pd.read_csv(os.path.join(data_dir, f"MAUD_{split}.csv"))
    main_df = df[df['data_type'] == "main"].copy()
    return main_df


def get_text_token_count(text: str, tokenizer) -> int:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_length=True,
    )
    return encoded["length"][0]