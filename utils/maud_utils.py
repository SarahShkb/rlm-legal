"""
Batch runner: MAUD (Merger Agreement Understanding Dataset), via LegalBench,
wired up for a Recursive Language Model (RLM) pipeline.

Data:  https://huggingface.co/datasets/nguha/legalbench   (config name = task name, e.g. "maud_type_of_consideration")
Task docs / prompt templates: https://github.com/HazyResearch/legalbench/tree/main/tasks/
Paper: Wang et al. 2023, "MAUD: An Expert-Annotated Legal NLP Dataset for
       Merger Agreement Understanding" (arXiv:2301.00876)

IMPORTANT: the LegalBench GitHub repo only ships a tiny `train.tsv` per task
(a handful of few-shot examples). The real evaluation split lives on
Hugging Face only -- load it with `datasets.load_dataset`, not by cloning
the repo.

Each MAUD subtask is an independent N-way multiple-choice classification
problem built around one "deal point" question from the ABA's 2021 Public
Target Deal Points Study. Every row has:
    index  -> row id
    answer -> gold answer key, a single letter (e.g. "A")
    text   -> the merger-agreement excerpt ("deal point" text) to read

Install once:
    pip install datasets
    # On Debian/Ubuntu with an externally-managed Python you may need:
    #   pip install datasets --break-system-packages
"""

import csv
import json
import re
import time
from pathlib import Path
from typing import Optional

from datasets import load_dataset

from llm.rlm import run_rlm_customized

# ---------------------------------------------------------------------------
# 1. CONFIG -- pick which MAUD subtask(s) to batch-run.
#    Full list of maud_* task names lives at:
#    https://github.com/HazyResearch/legalbench/blob/main/tasks.py
# ---------------------------------------------------------------------------
TASKS = [
    "maud_type_of_consideration",
    # add more maud_* task names here to widen the batch, e.g.:
    # "maud_financial_point_of_view_is_the_sole_consideration",
]

# The multiple-choice options for each task are NOT in the data files --
# they live in that task's README.md / base_prompt.txt on GitHub. You need
# to add an entry here for every maud_* task you include in TASKS above.
TASK_OPTIONS = {
    "maud_type_of_consideration": {
        "A": "All Cash",
        "B": "All Stock",
        "C": "Mixed Cash/Stock",
        "D": "Mixed Cash/Stock: Election",
    },
}

TASK_QUESTION = {
    "maud_type_of_consideration": "What type of consideration is specified in this agreement?",
}

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 2. PROMPT BUILDER -- mirrors LegalBench's own base_prompt.txt structure
#    (so results stay comparable to published baselines), but hands the
#    excerpt over whole rather than truncating it to 512 tokens the way the
#    original baselines had to. Let your RLM decide how to read it.
# ---------------------------------------------------------------------------
def build_maud_prompt(task: str, excerpt: str) -> str:
    options = TASK_OPTIONS[task]
    question = TASK_QUESTION[task]
    options_block = "\n".join(f"Option {k}: {v}" for k, v in options.items())
    return (
        "Instruction: Read the segment of a merger agreement and answer the "
        "multiple-choice question by choosing the option that best characterizes "
        "the agreement.\n"
        f"Question: {question}\n"
        f"{options_block}\n\n"
        f"Merger Agreement: {excerpt}\n\n"
        "Reply with only the option letter (e.g. \"A\").\nAnswer:"
    )


# ---------------------------------------------------------------------------
# 3. YOUR RLM CALL -- replace the body of this function with your actual
#    recursive-language-model pipeline. It should take the prompt (which may
#    be very long) and return the model's raw text answer.
# ---------------------------------------------------------------------------
def run_rlm(prompt: str) -> str:
    raise NotImplementedError(
        "Plug in your RLM pipeline here: the function that treats `prompt` as "
        "the external environment, recursively decomposes/queries it, and "
        "returns the final answer text."
    )


# ---------------------------------------------------------------------------
# 4. ANSWER PARSING -- model replies are rarely a bare letter, so extract one.
# ---------------------------------------------------------------------------
def extract_answer(raw_answer: str, valid_letters: set) -> Optional[str]:
    raw_answer = raw_answer.strip()
    match = re.search(r"\b([A-Z])\b", raw_answer.upper())
    if match and match.group(1) in valid_letters:
        return match.group(1)
    for letter in valid_letters:
        if raw_answer.upper().startswith(letter):
            return letter
    return None


# ---------------------------------------------------------------------------
# 5. BATCH LOOP
# ---------------------------------------------------------------------------
def run_batch_maud(task: str, split: str = "test", limit: int | None = None):
    ds = load_dataset("nguha/legalbench", task)[split]
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    valid_letters = set(TASK_OPTIONS[task].keys())
    results = []
    correct = 0

    for i, row in enumerate(ds):
        prompt = build_maud_prompt(task, row["text"])
        start = time.time()
        try:
            raw = run_rlm_customized(prompt)
        except NotImplementedError:
            raise
        except Exception as e:
            raw = f"[ERROR: {e}]"
        elapsed = time.time() - start

        predicted = extract_answer(raw, valid_letters)
        gold = row["answer"].strip().upper()
        is_correct = predicted == gold
        correct += int(is_correct)

        results.append({
            "task": task,
            "index": row["index"],
            "gold": gold,
            "predicted": predicted,
            "raw_response": raw,
            "correct": is_correct,
            "seconds": round(elapsed, 2),
        })
        print(f"[{task}] {i + 1}/{len(ds)}  gold={gold}  pred={predicted}  "
              f"{'correct' if is_correct else 'WRONG'}  ({elapsed:.1f}s)")

    accuracy = correct / len(ds) if len(ds) else 0.0
    print(f"\n{task}: {correct}/{len(ds)} correct = {accuracy:.1%}")
    return results, accuracy


# ---------------------------------------------------------------------------
# 6. RUN + SAVE
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    all_results = []
    summary = {}

    for task in TASKS:
        # Start small (limit=20-30) to sanity-check cost/latency before
        # scaling up to the full split (each maud_* task runs ~90-175 rows).
        results, accuracy = run_batch_maud(task, split="test", limit=30)
        all_results.extend(results)
        summary[task] = accuracy

    with open(OUTPUT_DIR / "maud_batch_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    with open(OUTPUT_DIR / "maud_batch_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "accuracy"])
        for task, acc in summary.items():
            writer.writerow([task, f"{acc:.4f}"])

    print("\nSaved detailed results to results/maud_batch_results.json")
    print("Saved summary to results/maud_batch_summary.csv")