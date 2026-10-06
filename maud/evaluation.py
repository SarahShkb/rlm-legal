import csv
import json
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd

from maud.maud_data_sampling import get_sample_rows_for_maud_152

# ============================================================
# 1. CONFIG
# ============================================================

LOG_PATH = Path("maud/maud_152_samples_log.csv")
OPTIONS_JSON_PATH = Path("maud/question_answer_map_unique.json")


# ============================================================
# 2. READ YOUR LOG
# ============================================================

def read_log(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    header = rows[1]
    records = []
    for row in rows[2:]:
        # if len(row) < 16:
        #     continue

        record = dict(zip(header, row))
        record["answer"] = row[len(row) - 1]  
        records.append(record)

    df = pd.DataFrame(records)

    numeric_cols = [
        "num_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cost_usd",
        "execution_time_s",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


    return df


# ============================================================
# 3. YOUR ORIGINAL LETTER ASSIGNMENT
# ============================================================

def assign_option_letters(options: list[str]) -> dict[str, str]:
    letters = [
        chr(ord("A") + i)
        for i in range(len(options))
    ]

    return dict(zip(letters, options))


# ============================================================
# 4. NORMALIZATION HELPERS
# ============================================================

def normalize_text(text) -> str:
    text = str(text).strip()
    text = text.strip("`\"'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def candidate_to_letter(
    candidate,
    option_map: dict[str, str] | None = None,
) -> str | None:

    if candidate is None:
        return None

    candidate = normalize_text(candidate)

    valid_letters = (
        set(option_map.keys())
        if option_map
        else set(string.ascii_uppercase)
    )

    # --------------------------------------------------------
    # Nested JSON string:
    # '{"answer": "B"}'
    # --------------------------------------------------------

    if candidate.startswith("{") and candidate.endswith("}"):
        try:
            obj = json.loads(candidate)

            if isinstance(obj, dict):
                for key in ("answer", "content", "choice"):
                    if key in obj:
                        result = candidate_to_letter(
                            obj[key],
                            option_map,
                        )

                        if result:
                            return result

        except json.JSONDecodeError:
            pass

    # --------------------------------------------------------
    # Exact single letter: A / B / C / ... / J
    # --------------------------------------------------------

    if re.fullmatch(r"[A-Z]", candidate, re.IGNORECASE):
        letter = candidate.upper()

        if letter in valid_letters:
            return letter

    # --------------------------------------------------------
    # Letter + option text:
    #
    # A. Yes
    # B. All Stock
    # C: Something
    # --------------------------------------------------------

    match = re.fullmatch(
        r"([A-Z])\s*[\.\):\-]\s*.+",
        candidate,
        re.IGNORECASE | re.DOTALL,
    )

    if match:
        letter = match.group(1).upper()

        if letter in valid_letters:
            return letter

    # --------------------------------------------------------
    # Exact option text
    #
    # Example:
    # model says "Yes"
    # JSON option is:
    # ["No", "Yes"]
    #
    # -> returns B
    # --------------------------------------------------------

    if option_map:

        candidate_norm = re.sub(
            r"^[A-Z]\s*[\.\):\-]\s*",
            "",
            candidate,
            flags=re.IGNORECASE,
        ).casefold()

        for letter, option in option_map.items():

            option_norm = normalize_text(option)

            option_norm = re.sub(
                r"^[A-Z]\s*[\.\):\-]\s*",
                "",
                option_norm,
                flags=re.IGNORECASE,
            ).casefold()

            if candidate_norm == option_norm:
                return letter

    return None


# ============================================================
# 5. EXTRACT JSON ANSWERS
# ============================================================

def extract_structured_value(text: str):
    """
    Looks for structures such as:

        {"answer": "C"}

        {"content": "B"}

        {"choice": "D"}

    """

    text = text.strip()

    # First try real JSON
    candidates = [
        text,
        re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ),
    ]

    for candidate in candidates:

        try:
            obj = json.loads(candidate)

            if isinstance(obj, dict):

                for key in ("answer", "content", "choice"):

                    if key in obj:
                        return obj[key]

        except json.JSONDecodeError:
            pass

    # JSON-like structures embedded in text
    patterns = [
        r'"answer"\s*:\s*"([^"]+)"',
        r"'answer'\s*:\s*'([^']+)'",

        r'"content"\s*:\s*"([^"]+)"',
        r"'content'\s*:\s*'([^']+)'",

        r'"choice"\s*:\s*"([^"]+)"',
        r"'choice'\s*:\s*'([^']+)'",
    ]

    for pattern in patterns:

        matches = list(
            re.finditer(
                pattern,
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
        )

        if matches:
            return matches[-1].group(1).strip()

    return None


# ============================================================
# 6. RECOVER ANSWER FROM REPL CODE
# ============================================================

def extract_code_assignment(text: str):
    """
    Only recovers explicit assignments such as:

        answer["content"] = "D"

        answer["content"] = json.dumps("B. announcement")

        answer["content"] = json.dumps({"answer": "C"})
    """

    # --------------------------------------------------------
    # answer["content"] = "B"
    # --------------------------------------------------------

    pattern = (
        r'answer\s*\[\s*["\']content["\']\s*\]'
        r'\s*=\s*(["\'])(.*?)\1'
    )

    matches = list(
        re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )

    if matches:
        return matches[-1].group(2)

    # --------------------------------------------------------
    # answer["content"] = json.dumps("B")
    # --------------------------------------------------------

    pattern = (
        r'answer\s*\[\s*["\']content["\']\s*\]'
        r'\s*=\s*json\.dumps\(\s*'
        r'(["\'])(.*?)\1\s*\)'
    )

    matches = list(
        re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )

    if matches:
        return matches[-1].group(2)

    # --------------------------------------------------------
    # answer["content"] = json.dumps({"answer": "C"})
    # --------------------------------------------------------

    pattern = (
        r'answer\s*\[\s*["\']content["\']\s*\]'
        r'\s*=\s*json\.dumps\(\s*'
        r'(\{.*?\})\s*\)'
    )

    matches = list(
        re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )

    if matches:
        return matches[-1].group(1)

    return None


# ============================================================
# 7. MAIN ANSWER EXTRACTOR
# ============================================================

def extract_answer(
    raw_output,
    option_map: dict[str, str] | None = None,
    recover_code: bool = False,
):
    """
    Returns:

        (letter, source)

    Example:

        ("B", "strict_direct")
        ("C", "strict_structured")
        ("D", "code_recovery_assignment")
        (None, "unparsed")
    """

    raw_output = "" if raw_output is None else str(raw_output)
    text = raw_output.strip()

    # Only executable code is treated as code here.
    # ```json ...``` is still a legitimate final response.
    code_pattern = r"```(?:repl|python|py)\s*.*?\s*```"

    code_blocks = list(
        re.finditer(
            code_pattern,
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )

    # Remove executable code for strict evaluation.
    noncode = re.sub(
        code_pattern,
        "\n",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    # ========================================================
    # STRICT EVALUATION
    # ========================================================

    if noncode:

        # JSON answer
        value = extract_structured_value(noncode)

        letter = candidate_to_letter(
            value,
            option_map,
        )

        if letter:
            return letter, "strict_structured"

        # Explicit answer markers outside code
        marker_patterns = [
            # 1. Matches "choice: A", "final answer = 'B'", etc.
            r'\b(?:final\s+answer|answer|choice)\s*[:=]\s*["\']?([A-Z])\b',

            # 2. Matches "the answer is C", "choice would be 'D'", etc.
            r'\b(?:the\s+)?(?:answer|choice)\s+(?:is|should be|would be)\s*["\']?([A-Z])\b',

            # 3. Matches "matches choice E", "match 'F'", etc.
            r'\bmatches?\s+(?:choice\s+)?["\']?([A-Z])\b',
        ]

        valid_letters = (
            set(option_map.keys())
            if option_map
            else set(string.ascii_uppercase)
        )

        for pattern in marker_patterns:

            matches = list(
                re.finditer(
                    pattern,
                    noncode,
                    flags=re.IGNORECASE,
                )
            )

            for match in reversed(matches):

                letter = match.group(1).upper()

                if letter in valid_letters:
                    return letter, "strict_marker"

        # Direct answer:
        # A
        # A. Something
        # Yes
        # No
        letter = candidate_to_letter(
            noncode,
            option_map,
        )

        if letter:
            return letter, "strict_direct"

    # ========================================================
    # RECOVER FROM REPL
    # ========================================================

    if recover_code and code_blocks:

        code_text = "\n".join(
            block.group(0)
            for block in code_blocks
        )

        # Explicit assignment
        value = extract_code_assignment(code_text)

        if value is not None:

            letter = candidate_to_letter(
                value,
                option_map,
            )

            if letter:
                return letter, "code_recovery_assignment"

        # JSON embedded in code
        value = extract_structured_value(code_text)

        letter = candidate_to_letter(
            value,
            option_map,
        )

        if letter:
            return letter, "code_recovery_structured"

        # CHOICE: B inside REPL
        patterns = [
            r'\bCHOICE\s*:\s*([A-Z])\b',
            r'\b(?:final\s+answer|answer|choice)\s*[:=]\s*["\']?([A-Z])\b',
        ]

        valid_letters = (
            set(option_map.keys())
            if option_map
            else set(string.ascii_uppercase)
        )

        for pattern in patterns:

            matches = list(
                re.finditer(
                    pattern,
                    code_text,
                    flags=re.IGNORECASE,
                )
            )

            for match in reversed(matches):

                letter = match.group(1).upper()

                if letter in valid_letters:
                    return letter, "code_recovery_marker"

    return None, "unparsed"



# ============================================================
# 8. LOAD EVERYTHING
# ============================================================


# ---- Load options JSON ----

with OPTIONS_JSON_PATH.open("r", encoding="utf-8") as f:
    options_by_question = json.load(f)


# ---- Load your original 152-row MAUD sample ----

samples_df = get_sample_rows_for_maud_152(split="train")

# IMPORTANT:
# Your sample was already sorted by n_tokens_contract
# before the experiment, so preserve that exact order.
samples_df = samples_df.reset_index(drop=True)

print(f"Number of sampled MAUD rows: {len(samples_df)}")


# ---- Load experiment log ----

log_df = read_log(LOG_PATH)

print(f"Number of log rows: {len(log_df)}")


# ============================================================
# 9. SPLIT VANILLA / RLM AND ALIGN BY POSITION
# ============================================================

vanilla = (
    log_df[log_df["method"] == "vanilla"]
    .reset_index(drop=True)
)

rlm = (
    log_df[log_df["method"] == "RLM-customized"]
    .reset_index(drop=True)
)


# Safety checks ------------------------------------------------

if len(vanilla) != len(samples_df):
    raise ValueError(
        f"Vanilla has {len(vanilla)} rows, "
        f"but sample has {len(samples_df)} rows."
    )

if len(rlm) != len(samples_df):
    raise ValueError(
        f"RLM has {len(rlm)} rows, "
        f"but sample has {len(samples_df)} rows."
    )


# Both models must have exactly the same number of records.
if len(vanilla) != len(rlm):
    raise ValueError(
        f"Vanilla has {len(vanilla)} rows, "
        f"RLM has {len(rlm)} rows."
    )


# ============================================================
# 10. BUILD ONE ROW PER MAUD QUESTION
# ============================================================

paired = pd.DataFrame({
    # Information from the original MAUD sample
    "question": samples_df["question"].values,
    "answer": samples_df["answer"].values,

    # Keep your contract-length information
    "n_tokens_contract": samples_df["n_tokens_contract"].values,

    # Useful additional columns if they exist
    **{
        col: samples_df[col].values
        for col in [
            "contract_name",
            "length_bin",
            "category",
            "text_type",
            "question_complexity",
            "material_adverse_effect",
        ]
        if col in samples_df.columns
    },

    # Vanilla
    "vanilla_raw": vanilla["raw_result"].values,

    "vanilla_num_calls": vanilla["num_calls"].values,
    "vanilla_input_tokens": vanilla["input_tokens"].values,
    "vanilla_output_tokens": vanilla["output_tokens"].values,
    "vanilla_total_tokens": vanilla["total_tokens"].values,
    "vanilla_cost_usd": vanilla["cost_usd"].values,
    "vanilla_execution_time_s": vanilla["execution_time_s"].values,

    # RLM
    "rlm_raw": rlm["raw_result"].values,

    "rlm_num_calls": rlm["num_calls"].values,
    "rlm_input_tokens": rlm["input_tokens"].values,
    "rlm_output_tokens": rlm["output_tokens"].values,
    "rlm_total_tokens": rlm["total_tokens"].values,
    "rlm_cost_usd": rlm["cost_usd"].values,
    "rlm_execution_time_s": rlm["execution_time_s"].values,

    # Keep the log's generated identifier only for debugging.
    # It does NOT identify the original MAUD row.
    "log_row_id": vanilla["row_id"].values,
})


# ============================================================
# 11. PARSE THE MODEL OUTPUTS
# ============================================================

def parse_model_row(row):

    question = row["question"]

    if question not in options_by_question:
        return pd.Series({
            "option_map": None,

            "vanilla_letter": None,
            "vanilla_source": "missing_options",

            "rlm_strict_letter": None,
            "rlm_strict_source": "missing_options",

            "rlm_recovered_letter": None,
            "rlm_recovered_source": "missing_options",
        })

    options = options_by_question[question]

    option_map = assign_option_letters(options)

    # Vanilla
    vanilla_letter, vanilla_source = extract_answer(
        row["vanilla_raw"],
        option_map,
        recover_code=False,
    )

    # RLM: strict
    rlm_strict_letter, rlm_strict_source = extract_answer(
        row["rlm_raw"],
        option_map,
        recover_code=False,
    )

    # RLM: recovered from REPL code
    rlm_recovered_letter, rlm_recovered_source = extract_answer(
        row["rlm_raw"],
        option_map,
        recover_code=True,
    )

    return pd.Series({
        "option_map": option_map,

        "vanilla_letter": vanilla_letter,
        "vanilla_source": vanilla_source,

        "rlm_strict_letter": rlm_strict_letter,
        "rlm_strict_source": rlm_strict_source,

        "rlm_recovered_letter": rlm_recovered_letter,
        "rlm_recovered_source": rlm_recovered_source,
    })


parsed = paired.apply(
    parse_model_row,
    axis=1,
)

paired = pd.concat(
    [paired, parsed],
    axis=1,
)





# ============================================================
# 13. DETAILED STATISTICS
# ============================================================
def get_gold_letter(row):
    question = row["question"]
    gold_answer = row["answer"]

    if question not in options_by_question:
        return None

    options = options_by_question[question]

    # A -> first option
    # B -> second option
    # C -> third option
    # ...
    option_map = assign_option_letters(options)

    return candidate_to_letter(
        gold_answer,
        option_map,
    )



paired["gold_letter"] = paired.apply(
    get_gold_letter,
    axis=1,
)

N = len(paired)
with pd.option_context("display.max_rows", None):
    print(paired[["vanilla_letter", "rlm_strict_letter", "rlm_recovered_letter", "answer", "gold_letter"]])

# ------------------------------------------------------------
# Basic counts
# ------------------------------------------------------------

vanilla_correct_n = int(paired["vanilla_correct"].sum())
rlm_strict_correct_n = int(paired["rlm_strict_correct"].sum())
rlm_recovered_correct_n = int(
    paired["rlm_recovered_correct"].sum()
)

vanilla_wrong_n = N - vanilla_correct_n
rlm_strict_wrong_n = N - rlm_strict_correct_n
rlm_recovered_wrong_n = N - rlm_recovered_correct_n


# ------------------------------------------------------------
# Accuracy
# ------------------------------------------------------------

vanilla_accuracy = vanilla_correct_n / N
rlm_strict_accuracy = rlm_strict_correct_n / N
rlm_recovered_accuracy = rlm_recovered_correct_n / N


# ------------------------------------------------------------
# Extraction / parse rate
# ------------------------------------------------------------

vanilla_parsed_n = int(
    paired["vanilla_letter"].notna().sum()
)

rlm_strict_parsed_n = int(
    paired["rlm_strict_letter"].notna().sum()
)

rlm_recovered_parsed_n = int(
    paired["rlm_recovered_letter"].notna().sum()
)

vanilla_parse_rate = vanilla_parsed_n / N
rlm_strict_parse_rate = rlm_strict_parsed_n / N
rlm_recovered_parse_rate = rlm_recovered_parsed_n / N


# ------------------------------------------------------------
# Direct comparison
# ------------------------------------------------------------

both_correct_n = int(
    (
        paired["vanilla_correct"]
        & paired["rlm_strict_correct"]
    ).sum()
)

vanilla_only_n = int(
    (
        paired["vanilla_correct"]
        & ~paired["rlm_strict_correct"]
    ).sum()
)

rlm_only_n = int(
    (
        ~paired["vanilla_correct"]
        & paired["rlm_strict_correct"]
    ).sum()
)

both_wrong_n = int(
    (
        ~paired["vanilla_correct"]
        & ~paired["rlm_strict_correct"]
    ).sum()
)


# ------------------------------------------------------------
# Accuracy difference
# ------------------------------------------------------------

accuracy_difference = (
    rlm_strict_accuracy
    - vanilla_accuracy
)

accuracy_difference_pp = accuracy_difference * 100


# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("OVERALL STATISTICS")
print("=" * 60)

print(f"\nTotal questions: {N}")

print("\n--- VANILLA ---")
print(f"Correct:    {vanilla_correct_n}/{N}")
print(f"Wrong:      {vanilla_wrong_n}/{N}")
print(f"Accuracy:   {vanilla_accuracy:.4f} "
      f"({vanilla_accuracy * 100:.2f}%)")
print(f"Parsed:     {vanilla_parsed_n}/{N}")
print(f"Parse rate: {vanilla_parse_rate * 100:.2f}%")

print("\n--- RLM (STRICT) ---")
print(f"Correct:    {rlm_strict_correct_n}/{N}")
print(f"Wrong:      {rlm_strict_wrong_n}/{N}")
print(f"Accuracy:   {rlm_strict_accuracy:.4f} "
      f"({rlm_strict_accuracy * 100:.2f}%)")
print(f"Parsed:     {rlm_strict_parsed_n}/{N}")
print(f"Parse rate: {rlm_strict_parse_rate * 100:.2f}%")

print("\n--- RLM (RECOVERED) ---")
print(f"Correct:    {rlm_recovered_correct_n}/{N}")
print(f"Wrong:      {rlm_recovered_wrong_n}/{N}")
print(f"Accuracy:   {rlm_recovered_accuracy:.4f} "
      f"({rlm_recovered_accuracy * 100:.2f}%)")
print(f"Parsed:     {rlm_recovered_parsed_n}/{N}")
print(f"Parse rate: {rlm_recovered_parse_rate * 100:.2f}%")

print("\n--- VANILLA vs RLM (STRICT) ---")
print(f"Both correct:              {both_correct_n}")
print(f"Vanilla correct / RLM wrong: {vanilla_only_n}")
print(f"RLM correct / Vanilla wrong: {rlm_only_n}")
print(f"Both wrong:                {both_wrong_n}")

print(
    f"\nAccuracy difference (RLM - Vanilla): "
    f"{accuracy_difference_pp:+.2f} percentage points"
)

if rlm_strict_accuracy > vanilla_accuracy:
    print(
        f"RLM has higher accuracy by "
        f"{abs(accuracy_difference_pp):.2f} percentage points."
    )
elif rlm_strict_accuracy < vanilla_accuracy:
    print(
        f"Vanilla has higher accuracy by "
        f"{abs(accuracy_difference_pp):.2f} percentage points."
    )
else:
    print("Vanilla and RLM have identical accuracy.")


# ============================================================
# 14. PAIRED OUTCOME TABLE
# ============================================================

comparison_table = pd.DataFrame({
    "outcome": [
        "Both correct",
        "Vanilla correct, RLM wrong",
        "Vanilla wrong, RLM correct",
        "Both wrong",
    ],
    "count": [
        both_correct_n,
        vanilla_only_n,
        rlm_only_n,
        both_wrong_n,
    ],
})

comparison_table["percentage"] = (
    comparison_table["count"] / N * 100
)

print("\n=== PAIRED OUTCOMES ===")
print(comparison_table.to_string(index=False))


# ============================================================
# 15. STATISTICS BY CONTRACT LENGTH BIN
# ============================================================

if "length_bin" in paired.columns:

    length_stats = (
        paired
        .groupby("length_bin", observed=False)
        .agg(
            n=("question", "size"),

            vanilla_correct=("vanilla_correct", "sum"),
            rlm_strict_correct=("rlm_strict_correct", "sum"),

            vanilla_accuracy=("vanilla_correct", "mean"),
            rlm_strict_accuracy=("rlm_strict_correct", "mean"),
        )
        .reset_index()
    )

    length_stats["difference_pp"] = (
        length_stats["rlm_strict_accuracy"]
        - length_stats["vanilla_accuracy"]
    ) * 100

    print("\n=== ACCURACY BY CONTRACT LENGTH BIN ===")
    print(length_stats.to_string(index=False))

    length_stats.to_csv(
        "maud_accuracy_by_length_bin.csv",
        index=False,
    )


# ============================================================
# 16. STATISTICS BY CATEGORY
# ============================================================

if "category" in paired.columns:

    category_stats = (
        paired
        .groupby("category", dropna=False)
        .agg(
            n=("question", "size"),

            vanilla_correct=("vanilla_correct", "sum"),
            rlm_strict_correct=("rlm_strict_correct", "sum"),

            vanilla_accuracy=("vanilla_correct", "mean"),
            rlm_strict_accuracy=("rlm_strict_correct", "mean"),
        )
        .reset_index()
    )

    category_stats["difference_pp"] = (
        category_stats["rlm_strict_accuracy"]
        - category_stats["vanilla_accuracy"]
    ) * 100

    print("\n=== ACCURACY BY CATEGORY ===")
    print(category_stats.to_string(index=False))

    category_stats.to_csv(
        "maud_accuracy_by_category.csv",
        index=False,
    )


# ============================================================
# 17. RLM RECOVERY ANALYSIS
# ============================================================

recovered_from_code = paired[
    paired["rlm_recovered_source"].str.startswith(
        "code_recovery",
        na=False,
    )
]

print("\n=== RLM CODE-RECOVERY ANALYSIS ===")

print(
    f"Answers recovered from REPL code: "
    f"{len(recovered_from_code)}"
)

if len(recovered_from_code) > 0:

    recovered_correct = int(
        recovered_from_code["rlm_recovered_correct"].sum()
    )

    print(
        f"Correct after recovery: "
        f"{recovered_correct}/{len(recovered_from_code)}"
    )

    print(
        f"Recovery accuracy: "
        f"{recovered_correct / len(recovered_from_code) * 100:.2f}%"
    )