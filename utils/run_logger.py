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
    "backend",
    "environment",
    "num_calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cost_usd",
    "execution_time_s",
]


def log_run(
    result: Any,
    rlm: Any,
    *,
    experiment: str = "",
    dataset: str = "",
    row_id: int | str = "",
    csv_path: str | Path = "logs/runs.csv",
) -> None:
    """Append one RLM completion as one CSV row."""

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    usage = result.usage_summary
    model_summaries = usage.model_usage_summaries

    models = list(model_summaries.keys())

    num_calls = sum(
        summary.total_calls
        for summary in model_summaries.values()
    )

    input_tokens = usage.total_input_tokens
    output_tokens = usage.total_output_tokens
    total_tokens = input_tokens + output_tokens

    cost = usage.total_cost

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment": experiment,
        "dataset": dataset,
        "row_id": row_id,
        "model": ", ".join(models),
        "backend": getattr(rlm, "backend", ""),
        "environment": getattr(rlm, "environment_type", ""),
        "num_calls": num_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": "" if cost is None else round(cost, 8),
        "execution_time_s": round(result.execution_time, 3),
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