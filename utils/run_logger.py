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
    "root_calls",
    "sub_calls",
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "total_tokens",
    "cost_usd",
    "execution_time_s",
    "response",
]


def log_run(
    result: Any,
    method: Any, #RLM or vanilla
    model: Any, #LLM model name
    rlm: Any, #RLM object, if applicable,
    total_tokens: int | str = "",
    output_tokens: int | str = "",
    input_tokens: int | str = "",
    cached_input_tokens: int | str = "",
    *,
    experiment: str = "",
    dataset: str = "",
    row_id: int | str = "",
    csv_path: str | Path = "logs/runs.csv",
) -> None:
    """Append one completion (vanilla or RLM) as one CSV row."""

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if method != "vanilla":
        # Requires llm.rlm_usage_patch: without it, root-model usage is
        # overwritten by the sub-call client's usage.
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
        cached_input_tokens = getattr(usage, "cached_input_tokens", "")
        root_calls = getattr(usage, "root_calls", "")
        sub_calls = getattr(usage, "sub_calls", "")

        cost = usage.total_cost

    else:
        models = model
        num_calls = 1
        root_calls = 1
        sub_calls = 0
        cost = None

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment": experiment,
        "dataset": dataset,
        "row_id": row_id,
        "model": models if method == "vanilla" else ", ".join(models),
        "method": method,
        "backend": model if method == "vanilla" else getattr(rlm, "backend", ""),
        "environment": "" if method == "vanilla" else getattr(rlm, "environment_type", ""),
        "num_calls": num_calls,
        "root_calls": root_calls,
        "sub_calls": sub_calls,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": "" if cost is None else round(cost, 8),
        "execution_time_s": None if method == "vanilla" else round(result.execution_time, 3),
        "response": result.response if method != "vanilla" else result.choices[0].message.content,
    }

    file_exists = csv_path.exists()

    # Don't append rows to a file written with a different column layout:
    # move it aside and start a fresh file with the current header.
    if file_exists:
        with csv_path.open(newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), None)
        if header != CSV_COLUMNS:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            legacy = csv_path.with_name(f"{csv_path.stem}_legacy_{stamp}{csv_path.suffix}")
            csv_path.rename(legacy)
            print(f"[run_logger] {csv_path} had an old header; moved it to {legacy}")
            file_exists = False

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
