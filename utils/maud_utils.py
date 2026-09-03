from __future__ import annotations

import argparse
import os
import re
import subprocess
import zipfile
from dataclasses import dataclass

import pandas as pd

from llm.rlm import run_rlm_customized

# -----------------------------------------------
# Data configuration
# -----------------------------------------------
REPO_URL = "https://github.com/The-Atticus-Project/maud.git"
TEXT_TYPE = "Type of Consideration"  # = legalbench "maud_type_of_consideration"

QUESTION = "What type of consideration is specified in this agreement?"
OPTIONS = {
    "A": "All Cash",
    "B": "All Stock",
    "C": "Mixed Cash/Stock",
    "D": "Mixed Cash/Stock: Election",
}
LABEL_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


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


# --------------------------------------------------------------------------- 
# Build the eval set: one row per contract, full contract text as context
# ---------------------------------------------------------------------------
@dataclass
class MAUD_Dataset_Sample:
    contract_name: str
    contract_path: str
    gold_answer: str
    gold_letter: str


def load_maud_dataset_samples(data_dir: str, split: str = "test") -> list[MAUD_Dataset_Sample]:
    # split: train | dev | test | all (concatenates all three)
    if split == "all":
        frames = [pd.read_csv(os.path.join(data_dir, f"MAUD_{s}.csv")) for s in ("train", "dev", "test")]
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(os.path.join(data_dir, f"MAUD_{split}.csv"))

    sub = df[df["text_type"] == TEXT_TYPE].copy()

    # Multiple rows per contract are just different excerpt variants of the
    # same underlying question -> collapse to one gold label per contract.
    sub = sub.drop_duplicates(subset="contract_name", keep="first")

    samples = []
    contracts_dir = os.path.join(data_dir, "contracts")
    for _, row in sub.iterrows():
        contract_path = os.path.join(contracts_dir, f"{row['contract_name']}.txt")
        if not os.path.exists(contract_path):
            print(f"  [skip] missing contract file for {row['contract_name']}")
            continue
        samples.append(
            MAUD_Dataset_Sample(
                contract_name=row["contract_name"],
                contract_path=contract_path,
                gold_answer=row["answer"],
                gold_letter=LABEL_TO_LETTER[int(row["label"])],
            )
        )
    return samples


# ---------------------------------------------------------------------------
# Build the prompt (question + options; the full contract is passed
#    separately as the RLM's long-context input, not inlined into the prompt)
# ---------------------------------------------------------------------------
def build_maud_type_of_consideration_prompt() -> str:
    lines = [
        ("Instruction: Read the merger agreement provided as context and answer "
        "the multiple-choice question by choosing the option that best "
        "characterizes the agreement."),
        "",
        f"Question: {QUESTION}",
    ]
    for letter, text in OPTIONS.items():
        lines.append(f"Option {letter}: {text}")
    lines.append("")
    lines.append("Respond with a single letter (A, B, C, or D) and nothing else.")
    return "\n".join(lines)


def parse_letter(response: str) -> str | None:
    _LETTER_RE = re.compile(r"\b([ABCD])\b")
    match = _LETTER_RE.search(response.strip().upper())
    return match.group(1) if match else None


def calculate_accuracy(results: pd.DataFrame) -> float:
    acc = results["is_correct"].mean() if len(results) else float("nan")
    n_correct = int(results["is_correct"].sum())
    print(f"\nAccuracy: {acc:.3f} ({n_correct}/{len(results)})")
    return acc


def run_maud_pipeline(limit: int | None = None):
    split = "dev"
    data_dir = get_maud_data_dir()
    data = load_maud_dataset_samples(data_dir, split=split)
    prompt = build_maud_type_of_consideration_prompt()
    rows = data[:limit] if limit else data
    save_path = "logs/runs_maud_toc.csv"

    results = []
    for i, ex in enumerate(rows, 1):

        # read contract text from its corresponding file
        with open(ex.contract_path, encoding="utf-8") as f:
            context = f.read()

            # RLM
            raw_rlm_response = run_rlm_customized(
                dataset="maud_type_of_consideration",
                row_id=f"maud_type_of_consideration_{split}_{i}",
                contract_text=context, 
                TASK_SPEC=prompt
            )

            prediction = parse_letter(raw_rlm_response)


        is_correct = prediction == ex.gold_letter
        results.append(
            {
                "split": split,
                "index": i,
                "contract_name": ex.contract_name,
                "gold_letter": ex.gold_letter,
                "gold_answer": ex.gold_answer,
                "pred_letter": prediction,
                "raw_response": raw_rlm_response,
                "is_correct": is_correct,
            }
        )
        print(f"[{i}/{len(rows)}] {ex.contract_name}: gold={ex.gold_letter} pred={prediction} {'OK' if is_correct else ''}")

    res_df = pd.DataFrame(results)
    res_df.to_csv(save_path, index=False)
    calculate_accuracy(res_df)

