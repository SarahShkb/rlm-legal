"""Generic experiment runner for the Contract-NLI and MAUD re-runs.

Same guarantees and logging as the LexGLUE pipeline (lexglue/pipeline.py), whose
call ledger and helpers it reuses. Layout of <dataset results>/<experiment>/:
  manifest.json            model/provider/sampling/prompt fingerprint fixed at the first run
  sessions.jsonl           one line per invocation (code state, pricing, OpenRouter credit)
  records/<dataset>.jsonl  one line per (item, method, repeat)
  calls/<dataset>.jsonl    one line per API request (root and sub-calls)
  trajectories/<method>/   RLM trajectories
  runs_csv/                the llm/ functions' log_run rows (kept out of logs/runs.csv)

Record status: "ok"; "failed" = the METHOD failed (RLM token budget exceeded, a
request too large for the provider, run timeout) -- final, scored as a wrong
answer and never retried, so failures are not re-rolled until they succeed;
"error" = transient (network, 5xx, ...) -- re-run when the command is repeated.
A watchdog interrupts a run after RUN_TIMEOUT_S (e.g. REPL code stuck in a
catastrophic regex): it is recorded as "failed" (RunTimeout); if the interrupt
does not land, the worker writes the record and exits with TIMEOUT_EXIT_CODE,
and lexglue-style launchers restart the rest of its slice.
"""

import hashlib
import inspect
import json
import os
import signal
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import openai
from rlm.utils.prompts import RLM_SYSTEM_PROMPT

import prompts.prompts as rlm_prompt_parts
from llm import config as llm_config
from llm.rlm import run_rlm_customized
from llm.vanila import run_vanilla
from lexglue.pipeline import (
    FATAL_STATUS_CODES,
    LEDGER,
    MAX_CONSECUTIVE_ERRORS,
    BudgetExceeded,
    _aggregate_calls,
    _rlm_stats,
    append_jsonl,
    read_jsonl,
    session_pricing,
)
from lexglue.tracking import (
    REPO_ROOT,
    code_state,
    environment_info,
    estimate_cost,
    openrouter_key_status,
    sha256_text,
    utc_now,
)


RUN_TIMEOUT_S = 30 * 60
HARD_EXIT_GRACE_S = 120
TIMEOUT_EXIT_CODE = 3


class RunTimeout(Exception):
    pass


def is_method_failure(e: BaseException) -> bool:
    """Failures caused by what the method did, not by the infrastructure."""
    if isinstance(e, RunTimeout) or type(e).__name__ == "TokenLimitExceededError":
        return True
    # the RLM built a request the provider refuses ("total text input size exceeds 8 MB")
    return isinstance(e, openai.BadRequestError) and "exceeds" in str(e)


class Watchdog:
    """Interrupts the current run after RUN_TIMEOUT_S; hard-exits if that does not work."""

    def __init__(self, timeout_s: float = RUN_TIMEOUT_S):
        self.timeout_s = timeout_s
        self._deadline = None
        self.fired = False
        self.on_hard_exit = None
        threading.Thread(target=self._loop, daemon=True).start()

    def start_run(self, on_hard_exit) -> None:
        self.fired, self.on_hard_exit = False, on_hard_exit
        self._deadline = time.monotonic() + self.timeout_s

    def end_run(self) -> None:
        self._deadline = None

    def _loop(self) -> None:
        while True:
            time.sleep(5)
            deadline = self._deadline
            if deadline is None:
                continue
            now = time.monotonic()
            if now > deadline and not self.fired:
                self.fired = True
                print(f"[benchmarks] WATCHDOG: run exceeded {self.timeout_s:.0f}s -> interrupting", flush=True)
                os.kill(os.getpid(), signal.SIGINT)
            elif self.fired and now > deadline + HARD_EXIT_GRACE_S:
                print("[benchmarks] WATCHDOG: interrupt did not land -> recording failure, hard exit", flush=True)
                if self.on_hard_exit:
                    self.on_hard_exit()
                os._exit(TIMEOUT_EXIT_CODE)


WATCHDOG: Watchdog | None = None


@dataclass
class Item:
    index: int            # position in the dataset's item list
    row_id: str
    document: str         # the contract: vanilla user message / RLM `context`
    task_spec: str
    gold: object
    meta: dict = field(default_factory=dict)


class Dataset:
    """Interface implemented by benchmarks.contract_nli_bench / benchmarks.maud_bench."""
    name: str
    results_root: Path
    methods: tuple[str, ...]
    source_files: list[str]          # files whose content shapes the prompts / tools

    def all_indices(self) -> list[int]: ...
    def items(self, indices: list[int]) -> list[Item]: ...
    def call(self, item: Item, method: str, *, experiment: str, max_iterations: int,
             runs_csv: str, log_dir: str, on_result) -> tuple[str, dict]: ...
    def score(self, item: Item, response: str | None) -> dict: ...


def experiment_dir(dataset: Dataset, experiment: str) -> Path:
    return dataset.results_root / experiment


def completed_keys(dataset: Dataset, experiment: str) -> set[tuple[int, str, int]]:
    rows = read_jsonl(experiment_dir(dataset, experiment) / "records" / f"{dataset.name}.jsonl")
    return {(r["index"], r["method"], r["repeat"]) for r in rows if r["status"] in ("ok", "failed")}


def _file_sha(rel: str) -> str | None:
    path = REPO_ROOT / rel
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def prompt_fingerprint(dataset: Dataset) -> dict:
    parts = {
        "RLM_SYSTEM_PROMPT": RLM_SYSTEM_PROMPT,
        **{name: getattr(rlm_prompt_parts, name) for name in (
            "RLM_FINISH_GUIDE", "CRITICAL_OUTPUT_RULE", "CRASH_PROOF_FINAL_SUBMISSION",
            "REPL_STDOUT_VISIBILITY", "PRESERVE_VALID_ANSWERS", "SUB_LLM_VERIFICATION")},
        "run_rlm_customized_source": inspect.getsource(run_rlm_customized),
        "run_vanilla_source": inspect.getsource(run_vanilla),
    }
    fp = {k: sha256_text(v) for k, v in parts.items()}
    fp.update({f"file:{rel}": _file_sha(rel) for rel in dataset.source_files})
    return fp


def llm_settings(max_iterations: int) -> dict:
    return {
        "llm_provider": llm_config.LLM_PROVIDER,
        "model": llm_config.MODEL,
        "base_url": llm_config.BASE_URL,
        "extra_body": llm_config.EXTRA_BODY,
        "temperature": llm_config.TEMPERATURE,
        "rlm_max_iterations": max_iterations,
    }


def check_manifest(exp_dir: Path, settings: dict, fingerprint: dict, allow_change: bool) -> None:
    path = exp_dir / "manifest.json"
    current = {"settings": settings, "prompt_fingerprint": fingerprint}
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**current, "created_at": utc_now()}, indent=2))
        return
    saved = json.loads(path.read_text())
    diffs = [f"{s}.{k}: {saved[s].get(k)!r} -> {v!r}"
             for s in ("settings", "prompt_fingerprint")
             for k, v in current[s].items() if saved[s].get(k) != v]
    if diffs and not allow_change:
        raise SystemExit(f"{exp_dir} was started with a different configuration:\n  " + "\n  ".join(diffs)
                         + "\nUse a new --experiment name, or --allow-config-change.")
    if diffs:
        print("[benchmarks] WARNING: configuration changed:\n  " + "\n  ".join(diffs))


def run_one(dataset: Dataset, item: Item, method: str, repeat: int, experiment: str,
            session_id: str, max_iterations: int, pricing: dict | None) -> dict:
    exp_dir = experiment_dir(dataset, experiment)
    rlm_info: dict = {}
    row = {
        "experiment": experiment,
        "session_id": session_id,
        "task": dataset.name,
        "index": item.index,
        "row_id": item.row_id,
        "method": method,
        "repeat": repeat,
        "llm_provider": llm_config.LLM_PROVIDER,
        "model": llm_config.MODEL,
        "meta": item.meta,
        "document_chars": len(item.document),
        "document_sha256": sha256_text(item.document),
        "task_spec_chars": len(item.task_spec),
        "task_spec_sha256": sha256_text(item.task_spec),
        "started_at": utc_now(),
    }
    LEDGER.begin()
    t0 = time.perf_counter()
    error, response, extra = None, None, {}
    records_path = exp_dir / "records" / f"{dataset.name}.jsonl"

    def hard_exit_record():
        append_jsonl(records_path, {**row, "finished_at": utc_now(), "wall_time_s": round(time.perf_counter() - t0, 3),
                                    "status": "failed", "error": f"RunTimeout: no result after {RUN_TIMEOUT_S}s (hard exit)",
                                    "parse_status": "error", "correct": False, "gold": item.gold})

    if WATCHDOG:
        WATCHDOG.start_run(hard_exit_record)
    try:
        response, extra = dataset.call(
            item, method,
            experiment=experiment,
            max_iterations=max_iterations,
            runs_csv=str(exp_dir / "runs_csv" / f"{dataset.name}__{session_id}.csv"),
            log_dir=str(exp_dir / "trajectories" / method),
            on_result=lambda result, rlm: rlm_info.update(_rlm_stats(result, rlm)),
        )
    except BaseException as e:  # noqa: BLE001 -- logged, then decided by the caller
        if WATCHDOG and WATCHDOG.fired:
            error = RunTimeout(f"no result after {RUN_TIMEOUT_S}s, interrupted ({type(e).__name__}: {e})")
        elif isinstance(e, Exception):
            error = e
        else:
            raise   # a real Ctrl+C
    finally:
        if WATCHDOG:
            WATCHDOG.end_run()
    wall = time.perf_counter() - t0
    calls = LEDGER.end()

    status = "ok" if error is None else ("failed" if is_method_failure(error) else "error")
    scored = dataset.score(item, response) if error is None else {"parse_status": "error", "correct": False if status == "failed" else None}
    row.update({
        "finished_at": utc_now(),
        "wall_time_s": round(wall, 3),
        "status": status,
        "error": None if error is None else f"{type(error).__name__}: {error}",
        "error_status_code": getattr(error, "status_code", None),
        "error_traceback": None if error is None else "".join(traceback.format_exception(error)),
        "response": response,
        "gold": item.gold,
        **scored,
        **extra,
        **_aggregate_calls(calls, pricing),
        "n_sdk_retries": len(LEDGER.last_retries),
        "sdk_retries": LEDGER.last_retries,
        **rlm_info,
    })
    for i, call in enumerate(calls):
        append_jsonl(exp_dir / "calls" / f"{dataset.name}.jsonl", {
            "experiment": experiment, "session_id": session_id, "task": dataset.name,
            "index": item.index, "row_id": item.row_id, "method": method, "repeat": repeat,
            "call_seq": i, "estimated_cost_usd": estimate_cost(call, (pricing or {}).get("pricing_per_token")),
            **call,
        })
    append_jsonl(records_path, row)
    if error is not None:
        row["_exception"] = error
    return row


def run_experiment(dataset: Dataset, experiment: str, indices: list[int], methods: list[str],
                   repeats: int = 1, max_iterations: int = 8, max_cost_usd: float | None = None,
                   allow_config_change: bool = False, store_text: bool = True) -> None:
    global WATCHDOG
    LEDGER.store_text = store_text
    LEDGER.install()
    if WATCHDOG is None:
        WATCHDOG = Watchdog()
    exp_dir = experiment_dir(dataset, experiment)
    settings = llm_settings(max_iterations)
    fingerprint = prompt_fingerprint(dataset)
    check_manifest(exp_dir, settings, fingerprint, allow_config_change)
    pricing = session_pricing(None)
    session_id = uuid.uuid4().hex[:12]
    is_openrouter = llm_config.LLM_PROVIDER == "openrouter"
    append_jsonl(exp_dir / "sessions.jsonl", {
        "type": "session_start", "session_id": session_id, "experiment": experiment,
        "dataset": dataset.name, "started_at": utc_now(), "settings": settings,
        "prompt_fingerprint": fingerprint, "methods": methods, "indices": indices,
        "repeats": repeats, "max_cost_usd": max_cost_usd, "pricing": pricing,
        "openrouter_key_before": openrouter_key_status(llm_config.API_KEY) if is_openrouter else None,
        "code": {**code_state(), "dataset_sources": {r: _file_sha(r) for r in dataset.source_files}},
        "environment": environment_info(),
    })
    print(f"[benchmarks] session {session_id} | {dataset.name} | {settings['llm_provider']} {settings['model']} "
          f"| methods={methods} n={len(indices)} repeats={repeats}")

    spent, consecutive_errors, stop_reason = 0.0, 0, "completed"
    totals = {"records_ok": 0, "records_error": 0, "records_skipped": 0}
    try:
        done = completed_keys(dataset, experiment)
        items = dataset.items(indices)
        for repeat in range(repeats):
            for item in items:
                for method in methods:
                    if (item.index, method, repeat) in done:
                        totals["records_skipped"] += 1
                        continue
                    row = run_one(dataset, item, method, repeat, experiment, session_id, max_iterations, pricing)
                    cost = row["reported_cost_usd"] if row["reported_cost_usd"] is not None else row["estimated_cost_usd"]
                    spent += cost or 0.0
                    print(f"[benchmarks] {item.row_id} {method:10s} r{repeat} {row['status']:5s} "
                          f"score={row.get('score')} parse={row.get('parse_status')} calls={row['n_calls']:3d} "
                          f"tok={row['total_tokens']} cost=${(cost or 0):.5f} t={row['wall_time_s']:.1f}s | session ${spent:.4f}")
                    if row["status"] == "ok":
                        totals["records_ok"] += 1
                        consecutive_errors = 0
                    elif row["status"] == "failed":
                        totals["records_failed"] = totals.get("records_failed", 0) + 1
                        consecutive_errors = 0
                        print(f"[benchmarks]   METHOD FAILURE (final, scored wrong): {row['error'][:200]}")
                    else:
                        totals["records_error"] += 1
                        consecutive_errors += 1
                        print(f"[benchmarks]   ERROR: {row['error']}")
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
        print(f"[benchmarks] STOP: {stop_reason}")
    finally:
        append_jsonl(exp_dir / "sessions.jsonl", {
            "type": "session_end", "session_id": session_id, "experiment": experiment,
            "dataset": dataset.name, "finished_at": utc_now(), "stop_reason": stop_reason,
            "session_cost_usd": round(spent, 8), **totals,
            "openrouter_key_after": openrouter_key_status(llm_config.API_KEY) if is_openrouter else None,
        })
        print(f"[benchmarks] session {session_id} ended: {stop_reason} | {totals} | cost ${spent:.4f}")
