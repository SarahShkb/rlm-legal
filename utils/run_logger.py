import csv
from datetime import datetime
from pathlib import Path
from typing import Any


CSV_COLUMNS = [
    "timestamp",
    "experiment",
    "dataset",
    "row_id",
    "model",
    "method",
    "backend",
    "environment",
    "num_calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cost_usd",
    "execution_time_s",
    "response",
    "actual",
]


def log_run(
    result: Any,
    actual: Any,
    method: Any, #RLM or vanilla
    model: Any, #LLM model name
    rlm: Any, #RLM object, if applicable,
    total_tokens: int | str = "",
    output_tokens: int | str = "",
    input_tokens: int | str = "",
    *,
    experiment: str = "",
    dataset: str = "",
    row_id: int | str = "",
    csv_path: str | Path = "logs/runs.csv",
) -> None:
    """Append one RLM completion as one CSV row."""

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if method != "vanilla":
        usage = result.usage_summary
        model_summaries = usage.model_usage_summaries

        models = list(model_summaries.keys())

        num_calls = sum(
            summary.total_calls
            for summary in model_summaries.values()
        )

        rlm_input_tokens = usage.total_input_tokens
        rlm_output_tokens = usage.total_output_tokens
        rlm_total_tokens = input_tokens + output_tokens

        cost = usage.total_cost

    else:
        models = model
        num_calls = 1
        cost = None

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment": experiment,
        "dataset": dataset,
        "row_id": row_id,
        "model": models if method == "vanilla" else ", ".join(models),
        "method": method,
        "backend": getattr(rlm, "backend", "") if method == "RLM" else model,
        "environment": getattr(rlm, "environment_type", "") if method == "RLM" else "",
        "num_calls": num_calls,
        "input_tokens": input_tokens if method == "vanilla" else rlm_input_tokens,
        "output_tokens": output_tokens if method == "vanilla" else rlm_output_tokens,
        "total_tokens": total_tokens if method == "vanilla" else rlm_total_tokens,
        "cost_usd": "" if cost is None else round(cost, 8),
        "execution_time_s": None if method == "vanilla" else round(result.execution_time, 3),
        "response": result.response if method != "vanilla" else result.choices[0].message.content,
        "actual": actual
    }

    file_exists = csv_path.exists()

    with csv_path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CSV_COLUMNS,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)