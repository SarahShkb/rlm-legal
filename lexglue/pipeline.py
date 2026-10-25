"""LexGLUE main loop: vanilla LLM vs RLM (custom system prompt), every call logged.

Output layout, lexglue/results/<experiment>/:
  manifest.json        model/provider/sampling/prompt fingerprint fixed at the first run;
                       later runs refuse to append if it changed (no silently mixed results)
  sessions.jsonl       one line per invocation: code state, environment, pricing,
                       OpenRouter credit before/after, totals, how it ended
  records/<task>.jsonl one line per (record, method, repeat): answer, parse, gold,
                       correctness, tokens, cost, latency, RLM trajectory stats
  calls/<task>.jsonl   one line per API request (root and sub-calls), incl. raw usage,
                       provider, finish reason, answer and reasoning text
  trajectories/<task>/ RLM trajectory of every RLM run (RLMLogger), linked from the record
  runs_csv/            the llm/ functions' log_run summary rows (kept out of logs/runs.csv)
Appends are serialised with a file lock, so several processes (lexglue.parallel)
can work on the same experiment, even on the same task.
A (record, method, repeat) already logged with status "ok" is skipped (resume).
"""

import fcntl
import inspect
import json
import os
import time
import traceback
import uuid
from pathlib import Path

import openai
from rlm.utils.prompts import RLM_SYSTEM_PROMPT

import lexglue.prompts as lexglue_prompts
import prompts.prompts as rlm_prompt_parts
from lexglue.config import (
    DEFAULT_MAX_ITERATIONS,
    MULTIPLE_CHOICE,
    RESULTS_DIR,
    TASKS,
)
from lexglue.data import dataset_fingerprint, label_names, load_records, load_split
from lexglue.parsing import is_correct, parse_prediction
from lexglue.prompts import CONTEXT_LABEL, build_task_spec
from lexglue.tracking import (
    CallLedger,
    code_state,
    environment_info,
    estimate_cost,
    openrouter_endpoint,
    openrouter_key_status,
    sha256_text,
    utc_now,
)
from llm import config as llm_config
from llm.rlm import run_rlm_customized
from llm.vanila import run_vanilla

# Request errors that will not go away by moving to the next record:
# bad key, no credit, forbidden, unknown model / pinned provider unavailable
FATAL_STATUS_CODES = {401, 402, 403, 404}
MAX_CONSECUTIVE_ERRORS = 3

LEDGER = CallLedger()


class BudgetExceeded(RuntimeError):
    pass


# ------------------------------------------------
# Helpers
# ------------------------------------------------

def experiment_dir(experiment: str) -> Path:
    return RESULTS_DIR / experiment


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        try:
            return [json.loads(line) for line in f if line.strip()]
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def completed_keys(experiment: str, task: str) -> set[tuple[int, str, int]]:
    rows = read_jsonl(experiment_dir(experiment) / "records" / f"{task}.jsonl")
    return {(r["index"], r["method"], r["repeat"]) for r in rows if r["status"] == "ok"}


def prompt_fingerprint() -> dict:
    """Hashes of everything that shapes the prompts and sampling of both conditions."""
    parts = {
        "RLM_SYSTEM_PROMPT": RLM_SYSTEM_PROMPT,
        **{name: getattr(rlm_prompt_parts, name) for name in (
            "RLM_FINISH_GUIDE", "CRITICAL_OUTPUT_RULE", "CRASH_PROOF_FINAL_SUBMISSION",
            "REPL_STDOUT_VISIBILITY", "PRESERVE_VALID_ANSWERS", "SUB_LLM_VERIFICATION")},
        "run_rlm_customized_source": inspect.getsource(run_rlm_customized),
        "run_vanilla_source": inspect.getsource(run_vanilla),
        "lexglue_prompts_source": inspect.getsource(lexglue_prompts),
    }
    return {k: sha256_text(v) for k, v in parts.items()}


def llm_settings(max_iterations: int, split: str) -> dict:
    return {
        "llm_provider": llm_config.LLM_PROVIDER,
        "model": llm_config.MODEL,
        "base_url": llm_config.BASE_URL,
        "extra_body": llm_config.EXTRA_BODY,
        "temperature": llm_config.TEMPERATURE,
        "rlm_max_iterations": max_iterations,
        "split": split,
    }


def check_manifest(experiment: str, settings: dict, fingerprint: dict, allow_change: bool) -> None:
    path = experiment_dir(experiment) / "manifest.json"
    current = {"settings": settings, "prompt_fingerprint": fingerprint}
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**current, "created_at": utc_now()}, indent=2))
        return
    saved = json.loads(path.read_text())
    diffs = [
        f"{section}.{k}: {saved[section].get(k)!r} -> {v!r}"
        for section in ("settings", "prompt_fingerprint")
        for k, v in current[section].items()
        if saved[section].get(k) != v
    ]
    if diffs and not allow_change:
        raise SystemExit(
            f"Experiment '{experiment}' was started with a different configuration:\n  "
            + "\n  ".join(diffs)
            + "\nUse a new --experiment name, or --allow-config-change if this is intended "
              "(the change is recorded in sessions.jsonl)."
        )
    if diffs:
        print("[lexglue] WARNING: configuration changed:\n  " + "\n  ".join(diffs))


def session_pricing(price_per_mtok: tuple[float, float, float] | None) -> dict | None:
    """Per-token prices used for estimated_cost_usd (cross-check of the billed cost)."""
    if price_per_mtok:
        p_in, p_out, p_cache = price_per_mtok
        return {"source": "manual", "pricing_per_token": {
            "prompt": p_in / 1e6, "completion": p_out / 1e6, "input_cache_read": p_cache / 1e6}}
    if llm_config.LLM_PROVIDER == "openrouter":
        endpoint = openrouter_endpoint(llm_config.MODEL, llm_config.OPENROUTER_PROVIDER)
        if endpoint is None:
            raise SystemExit(
                f"OpenRouter has no endpoint tagged {llm_config.OPENROUTER_PROVIDER!r} for "
                f"{llm_config.MODEL!r}; check OPENROUTER_MODEL / OPENROUTER_PROVIDER."
            )
        return {"source": "openrouter_endpoint", **endpoint}
    return None


# ------------------------------------------------
# One record x one method
# ------------------------------------------------

def _rlm_stats(result, rlm) -> dict:
    trajectory = getattr(result, "metadata", None) or {}
    iterations = trajectory.get("iterations", [])
    blocks = [b for it in iterations for b in (it.get("code_blocks") or [])]
    last = iterations[-1] if iterations else {}
    last_prompt = last.get("prompt")
    # RLM._default_answer: out of iterations, one extra call asks for a final answer
    fallback = (
        isinstance(last_prompt, list) and bool(last_prompt)
        and str(last_prompt[-1].get("content", "")).startswith("Please provide a final answer")
    )
    usage = getattr(result, "usage_summary", None)
    return {
        "rlm_iterations": len(iterations) - (1 if fallback else 0),
        "rlm_hit_max_iterations": fallback,
        "rlm_code_blocks": len(blocks),
        "rlm_repl_errors": sum(1 for b in blocks if (b.get("result") or {}).get("stderr")),
        "rlm_execution_time_s": round(getattr(result, "execution_time", 0) or 0, 3),
        "rlm_trajectory_path": getattr(getattr(rlm, "logger", None), "log_file_path", None),
        "rlm_usage_summary": usage.to_dict() if usage is not None else None,
        "rlm_usage_root_calls": getattr(usage, "root_calls", None),
        "rlm_usage_sub_calls": getattr(usage, "sub_calls", None),
    }


def _aggregate_calls(calls: list[dict], pricing: dict | None) -> dict:
    def total(key):
        values = [c.get(key) for c in calls if c.get(key) is not None]
        return sum(values) if values else None

    prices = (pricing or {}).get("pricing_per_token")
    estimated = [estimate_cost(c, prices) for c in calls if c["error"] is None]
    finish = {}
    for c in calls:
        if c["error"] is None:
            finish[str(c.get("finish_reason"))] = finish.get(str(c.get("finish_reason")), 0) + 1
    return {
        "n_calls": len(calls),
        "n_root_calls": sum(c["role"] == "root" for c in calls),
        "n_sub_calls": sum(c["role"] == "sub" for c in calls),
        "n_failed_calls": sum(c["error"] is not None for c in calls),
        # malformed provider answers, re-sent by llm/empty_response_retry.py
        "n_empty_responses": sum(c["error"] is None and not c.get("content_chars") for c in calls),
        "prompt_tokens": total("prompt_tokens"),
        "cached_prompt_tokens": total("cached_prompt_tokens"),
        "completion_tokens": total("completion_tokens"),
        "reasoning_tokens": total("reasoning_tokens"),
        "total_tokens": total("total_tokens"),
        "reported_cost_usd": total("reported_cost_usd"),
        "upstream_cost_usd": total("upstream_cost_usd"),
        "estimated_cost_usd": sum(e for e in estimated if e is not None) if any(e is not None for e in estimated) else None,
        "api_latency_s_sum": round(sum(c["latency_s"] for c in calls), 3),
        "api_latency_s_max": max((c["latency_s"] for c in calls), default=None),
        "providers": sorted({str(c.get("provider")) for c in calls if c.get("provider")}),
        "response_models": sorted({str(c.get("response_model")) for c in calls if c.get("response_model")}),
        "finish_reasons": finish,
        "n_truncated_calls": finish.get("length", 0),
    }


def run_one(record, labels, method: str, repeat: int, experiment: str, session_id: str,
            max_iterations: int, pricing: dict | None) -> dict:
    task_spec = build_task_spec(record, labels)
    dataset = f"lexglue-{record.task}"
    exp_dir = experiment_dir(experiment)
    # one summary CSV per process: log_run has no file locking
    runs_csv = str(exp_dir / "runs_csv" / f"{record.task}__{session_id}.csv")
    rlm_info: dict = {}

    def keep_rlm_result(result, rlm):
        rlm_info.update(_rlm_stats(result, rlm))

    row = {
        "experiment": experiment,
        "session_id": session_id,
        "task": record.task,
        "split": record.split,
        "index": record.index,
        "row_id": record.row_id,
        "method": method,
        "repeat": repeat,
        "llm_provider": llm_config.LLM_PROVIDER,
        "model": llm_config.MODEL,
        "document_chars": len(record.document),
        "document_sha256": record.document_sha256,
        "task_spec_chars": len(task_spec),
        "task_spec_sha256": sha256_text(task_spec),
        "started_at": utc_now(),
    }
    LEDGER.begin()
    t0 = time.perf_counter()
    error = None
    response = None
    try:
        if method == "vanilla":
            response = run_vanilla(
                dataset=dataset,
                row_id=record.row_id,
                contract_text=record.document,
                TASK_SPEC=task_spec,
                experiment=experiment,
                temperature=llm_config.TEMPERATURE,
                context_label=CONTEXT_LABEL,
                runs_csv=runs_csv,
            )
        elif method == "rlm":
            response = run_rlm_customized(
                dataset=dataset,
                row_id=record.row_id,
                contract_text=record.document,
                TASK_SPEC=task_spec,
                experiment=experiment,
                max_iterations=max_iterations,
                escape_task_spec_braces=True,
                on_result=keep_rlm_result,
                runs_csv=runs_csv,
                log_dir=str(exp_dir / "trajectories" / record.task),
            )
        else:
            raise ValueError(f"unknown method {method!r}")
    except Exception as e:  # noqa: BLE001 -- logged, then decided by the caller
        error = e
    wall = time.perf_counter() - t0
    calls = LEDGER.end()

    parsed = parse_prediction(record.task, labels, len(record.options), response)
    row.update({
        "finished_at": utc_now(),
        "wall_time_s": round(wall, 3),
        "status": "ok" if error is None else "error",
        "error": None if error is None else f"{type(error).__name__}: {error}",
        "error_status_code": getattr(error, "status_code", None),
        "error_traceback": None if error is None else "".join(traceback.format_exception(error)),
        "response": response,
        "prediction": parsed.prediction,
        "parse_status": parsed.status if error is None else "error",
        "invalid_labels": parsed.invalid,
        "gold": record.gold,
        "correct": is_correct(record.task, parsed.prediction, record.gold) if error is None else None,
        **_aggregate_calls(calls, pricing),
        **rlm_info,
    })

    for i, call in enumerate(calls):
        append_jsonl(experiment_dir(experiment) / "calls" / f"{record.task}.jsonl", {
            "experiment": experiment,
            "session_id": session_id,
            "task": record.task,
            "index": record.index,
            "row_id": record.row_id,
            "method": method,
            "repeat": repeat,
            "call_seq": i,
            "estimated_cost_usd": estimate_cost(call, (pricing or {}).get("pricing_per_token")),
            **call,
        })
    append_jsonl(experiment_dir(experiment) / "records" / f"{record.task}.jsonl", row)
    if error is not None:
        row["_exception"] = error
    return row


# ------------------------------------------------
# Experiment loop
# ------------------------------------------------

def run_experiment(
        experiment: str,
        tasks: list[str],
        methods: list[str],
        split: str,
        indices: list[int],
        repeats: int = 1,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        max_cost_usd: float | None = None,
        price_per_mtok: tuple[float, float, float] | None = None,
        allow_config_change: bool = False,
        store_text: bool = True,
) -> None:
    LEDGER.store_text = store_text
    LEDGER.install()
    settings = llm_settings(max_iterations, split)
    fingerprint = prompt_fingerprint()
    check_manifest(experiment, settings, fingerprint, allow_config_change)
    pricing = session_pricing(price_per_mtok)

    session_id = uuid.uuid4().hex[:12]
    is_openrouter = llm_config.LLM_PROVIDER == "openrouter"
    session = {
        "type": "session_start",
        "session_id": session_id,
        "experiment": experiment,
        "started_at": utc_now(),
        "settings": settings,
        "prompt_fingerprint": fingerprint,
        "tasks": tasks,
        "methods": methods,
        "indices": indices,
        "repeats": repeats,
        "max_cost_usd": max_cost_usd,
        "pricing": pricing,
        "dataset_fingerprints": {t: dataset_fingerprint(t, split) for t in tasks},
        "openrouter_key_before": openrouter_key_status(llm_config.API_KEY) if is_openrouter else None,
        "code": code_state(),
        "environment": environment_info(),
    }
    sessions_path = experiment_dir(experiment) / "sessions.jsonl"
    append_jsonl(sessions_path, session)
    print(f"[lexglue] session {session_id} | {settings['llm_provider']} {settings['model']} | "
          f"tasks={tasks} methods={methods} n={len(indices)} repeats={repeats}")

    spent = 0.0          # billed (OpenRouter) or estimated cost of this session
    totals = {"records_ok": 0, "records_error": 0, "records_skipped": 0}
    consecutive_errors = 0
    stop_reason = "completed"
    try:
        for task in tasks:
            ds = load_split(task, split)
            labels = None if TASKS[task]["task_type"] == MULTIPLE_CHOICE else label_names(task, ds.features)
            records = load_records(task, split, indices)
            done = completed_keys(experiment, task)
            for repeat in range(repeats):
                for record in records:
                    for method in methods:
                        if (record.index, method, repeat) in done:
                            totals["records_skipped"] += 1
                            continue
                        row = run_one(record, labels, method, repeat, experiment, session_id,
                                      max_iterations, pricing)
                        cost = row["reported_cost_usd"] if row["reported_cost_usd"] is not None else row["estimated_cost_usd"]
                        spent += cost or 0.0
                        mark = {True: "correct", False: "wrong", None: "-"}[row["correct"]]
                        print(f"[lexglue] {record.row_id} {method:7s} r{repeat} {row['status']:5s} "
                              f"{mark:7s} parse={row['parse_status']:15s} calls={row['n_calls']:3d} "
                              f"tok={row['total_tokens']} cost=${(cost or 0):.5f} "
                              f"t={row['wall_time_s']:.1f}s | session ${spent:.4f}")
                        if row["status"] == "ok":
                            totals["records_ok"] += 1
                            consecutive_errors = 0
                        else:
                            totals["records_error"] += 1
                            consecutive_errors += 1
                            print(f"[lexglue]   ERROR: {row['error']}")
                            exc = row.get("_exception")
                            if isinstance(exc, openai.APIStatusError) and exc.status_code in FATAL_STATUS_CODES:
                                stop_reason = f"fatal API error {exc.status_code}"
                                raise exc
                            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                                stop_reason = f"{consecutive_errors} consecutive errors"
                                raise RuntimeError(stop_reason)
                        if max_cost_usd is not None and spent >= max_cost_usd:
                            stop_reason = f"budget reached (${spent:.4f} >= ${max_cost_usd})"
                            raise BudgetExceeded(stop_reason)
    except KeyboardInterrupt:
        stop_reason = "interrupted by user"
        raise
    except BudgetExceeded:
        print(f"[lexglue] STOP: {stop_reason}")
    finally:
        append_jsonl(sessions_path, {
            "type": "session_end",
            "session_id": session_id,
            "experiment": experiment,
            "finished_at": utc_now(),
            "stop_reason": stop_reason,
            "session_cost_usd": round(spent, 8),
            **totals,
            "openrouter_key_after": openrouter_key_status(llm_config.API_KEY) if is_openrouter else None,
        })
        print(f"[lexglue] session {session_id} ended: {stop_reason} | ok={totals['records_ok']} "
              f"errors={totals['records_error']} skipped={totals['records_skipped']} | cost ${spent:.4f}")
