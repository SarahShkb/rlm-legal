import csv
import json
import re
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

CSV_LOG = Path("contract-nli/results/contract-nli-first-3-dev-contracts.csv")
CONTRACT_NLI_JSON = Path("contract-nli/main_repo/test.json")

# Record range in the CSV.
# Python-style indexing: i is inclusive, j is exclusive.
# Example: 0, 10 => first 10 records.
i = 0
j = 5

# Optional: filter to a specific method/model.
# Set to None if you want all records.
METHOD_FILTER = "RLM-customized"
# METHOD_FILTER = None


# ============================================================
# 1. LOAD GOLD LABELS FROM CONTRACT-NLI
# ============================================================

with open(CONTRACT_NLI_JSON, "r", encoding="utf-8") as f:
    contract_nli = json.load(f)


# Build:
# gold_choices[document_id]["nda-11"] = "Entailment"
gold_choices = {}

for document in contract_nli["documents"]:
    doc_id = document["id"]

    # Contract-NLI normally contains annotation_sets.
    # We use the first annotation set here.
    annotations = document["annotation_sets"][0]["annotations"]

    gold_choices[doc_id] = {
        hypothesis_id: annotation["choice"]
        for hypothesis_id, annotation in annotations.items()
    }


print(f"Loaded gold labels for {len(gold_choices)} documents.")


# ============================================================
# 2. LOAD CSV LOG
# ============================================================

with open(CSV_LOG, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)


print(f"Loaded {len(rows)} log records.")


# ============================================================
# 3. SELECT RECORDS i -> j
# ============================================================

selected_rows = rows[i:j]

print(f"Evaluating records [{i}:{j}] -> {len(selected_rows)} records.")


# ============================================================
# 4. COMPARE PREDICTIONS WITH GOLD LABELS
# ============================================================

total_predictions = 0
correct_predictions = 0

per_document = []


for row in selected_rows:

    row_id = row["row_id"]

    # Example:
    # "dev-15" -> split="dev", document_id=15
    match = re.fullmatch(r"([^-]+)-(\d+)", row_id)

    if not match:
        print(f"WARNING: Could not parse row_id: {row_id}")
        continue

    split = match.group(1)
    document_id = int(match.group(2))

    # Since this script is loading dev.json, make sure the split matches.
    if split != "dev":
        print(
            f"WARNING: {row_id} is not from dev split, "
            f"but gold file is dev.json."
        )
        continue

    # --------------------------------------------------------
    # Find prediction JSON.
    # Your example puts it in the final CSV field.
    # --------------------------------------------------------
    prediction_column = list(row.keys())[-1]
    prediction_raw = row[prediction_column]

    try:
        # --------------------------------------------------------
        # Case 1: CSV value is a list containing one JSON string
        # Example:
        # ['{"nda-1": {"choice": "Entailment"}, ...}']
        # --------------------------------------------------------
        if isinstance(prediction_raw, list):
            if len(prediction_raw) != 1:
                print(
                    f"WARNING: Prediction list for {row_id} has "
                    f"{len(prediction_raw)} elements"
                )
                continue

            prediction_raw = prediction_raw[0]

        # --------------------------------------------------------
        # Now prediction_raw should be the JSON string
        # --------------------------------------------------------
        if isinstance(prediction_raw, str):
            predictions = json.loads(prediction_raw)
        else:
            predictions = prediction_raw

    except (json.JSONDecodeError, TypeError) as e:
        print(f"WARNING: Could not parse predictions for {row_id}: {e}")
        continue


    # ------------------------------------------------------------
    # Make sure we ended up with a dictionary
    # ------------------------------------------------------------

    if not isinstance(predictions, dict):
        print(
            f"WARNING: Predictions for {row_id} are not a dictionary. "
            f"Got {type(predictions).__name__}: {predictions}"
        )
        continue


    # ------------------------------------------------------------
    # Compare predictions with gold labels
    # ------------------------------------------------------------

    gold = gold_choices[document_id]
    doc_correct = 0
    doc_total = 0

    for hypothesis_id, predicted_annotation in predictions.items():

        predicted_choice = predicted_annotation.get("choice")
        gold_choice = gold.get(hypothesis_id)

        if gold_choice is None:
            print(
                f"WARNING: {row_id}: "
                f"{hypothesis_id} not found in gold labels"
            )
            continue

        doc_total += 1
        total_predictions += 1

        if predicted_choice == gold_choice:
            doc_correct += 1
            correct_predictions += 1

    doc_accuracy = (
        doc_correct / doc_total
        if doc_total > 0
        else 0.0
    )

    per_document.append({
        "row_id": row_id,
        "document_id": document_id,
        "method": row.get("method"),
        "model": row.get("model"),
        "correct": doc_correct,
        "total": doc_total,
        "accuracy": doc_accuracy,
    })


# ============================================================
# 5. PRINT RESULTS
# ============================================================

overall_accuracy = (
    correct_predictions / total_predictions
    if total_predictions > 0
    else 0.0
)

print("\n" + "=" * 60)
print("RESULT: last 5 records with max_iterations=20")
print("=" * 60)

print(f"Records evaluated:     {len(selected_rows)}")
print(f"Predictions evaluated: {total_predictions}")
print(f"Correct predictions:   {correct_predictions}")
print(f"Incorrect predictions: {total_predictions - correct_predictions}")
print(f"Accuracy:              {overall_accuracy:.4%}")


# ============================================================
# 6. PRINT PER-DOCUMENT ACCURACY
# ============================================================

print("\nPer-document accuracy:")
print("-" * 60)

for result in per_document:
    print(
        f"{result['row_id']:10s} "
        f"{result['correct']:2d}/{result['total']:2d} "
        f"({result['accuracy']:.2%})"
    )