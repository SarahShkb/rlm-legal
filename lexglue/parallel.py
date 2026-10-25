"""Run one LexGLUE experiment as parallel worker processes under ONE global budget.

Each worker is a normal `run_lexglue run` on one task and a slice of its records
(resumable, fully logged by lexglue.pipeline). The launcher:
  * skips slices that are already complete (re-running the command resumes);
  * every POLL_S seconds sums the cost of the records logged since launch and
    asks OpenRouter how much the key has spent since launch (free endpoint);
    if either reaches --max-cost it stops launching and sends SIGINT to every
    worker (each one closes its session record cleanly);
  * stops launching new workers after MAX_FAILED_WORKERS workers failed
    (e.g. no credit, provider unavailable);
  * writes worker stdout to results/<experiment>/workers/ and a summary line to
    sessions.jsonl.
"""

import os
import signal
import subprocess
import sys
import time
import uuid

from llm import config as llm_config
from lexglue.config import TASKS
from lexglue.pipeline import (
    append_jsonl,
    check_manifest,
    completed_keys,
    experiment_dir,
    llm_settings,
    prompt_fingerprint,
    read_jsonl,
)
from lexglue.tracking import REPO_ROOT, openrouter_key_status, utc_now

POLL_S = 20
MAX_FAILED_WORKERS = 3
# longest tasks first, so the slowest workers start early
TASK_ORDER = ["ecthr_b", "eurlex", "ecthr_a", "scotus", "case_hold", "unfair_tos", "ledgar"]


def plan_shards(experiment, tasks, indices, methods, repeats, shard_size) -> list[tuple[str, list[int]]]:
    shards = []
    for task in sorted(tasks, key=TASK_ORDER.index):
        done = completed_keys(experiment, task)
        todo = [i for i in indices
                if any((i, m, r) not in done for m in methods for r in range(repeats))]
        shards += [(task, todo[k:k + shard_size]) for k in range(0, len(todo), shard_size)]
    return shards


def spent_since(experiment: str, tasks: list[str], since: str) -> tuple[float, int]:
    """(cost of records started since `since`, number of such records)."""
    total, n = 0.0, 0
    for task in tasks:
        for r in read_jsonl(experiment_dir(experiment) / "records" / f"{task}.jsonl"):
            if r["started_at"] >= since:
                cost = r["reported_cost_usd"] if r["reported_cost_usd"] is not None else r["estimated_cost_usd"]
                total += cost or 0.0
                n += 1
    return total, n


def run_parallel(experiment, tasks, methods, split, indices, repeats, max_iterations,
                 max_cost_usd, workers, shard_size, store_text=True) -> int:
    check_manifest(experiment, llm_settings(max_iterations, split), prompt_fingerprint(), allow_change=False)
    is_openrouter = llm_config.LLM_PROVIDER == "openrouter"
    shards = plan_shards(experiment, tasks, indices, methods, repeats, shard_size)
    total_runs = sum(len(idx) for _, idx in shards) * len(methods) * repeats
    launch_id = uuid.uuid4().hex[:12]
    since = utc_now()
    key_before = openrouter_key_status(llm_config.API_KEY) if is_openrouter else None
    log_dir = experiment_dir(experiment) / "workers"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[parallel] {launch_id}: {len(shards)} shards (<= {shard_size} records), {workers} workers, "
          f"<= {total_runs} runs, budget ${max_cost_usd}")

    queue = list(shards)
    running: dict[str, tuple[subprocess.Popen, object]] = {}
    exit_codes: dict[str, int] = {}
    stop_reason = None
    last_poll = 0.0
    spent, n_logged, key_spent = 0.0, 0, None

    def stop_all(reason):
        nonlocal stop_reason
        if stop_reason is None:
            stop_reason = reason
            print(f"[parallel] STOP: {reason} -> interrupting {len(running)} workers")
            for proc, _ in running.values():
                if proc.poll() is None:
                    proc.send_signal(signal.SIGINT)

    try:
        while running or (queue and stop_reason is None):
            while queue and len(running) < workers and stop_reason is None:
                task, idx = queue.pop(0)
                name = f"{task}_{idx[0]:04d}-{idx[-1]:04d}"
                cmd = [sys.executable, "-m", "lexglue.run_lexglue", "run", "--experiment", experiment,
                       "--tasks", task, "--split", split, "--indices", *map(str, idx),
                       "--methods", *methods, "--repeats", str(repeats),
                       "--max-iterations", str(max_iterations), "--max-cost", str(max_cost_usd), "--yes"]
                if not store_text:
                    cmd.append("--no-store-text")
                out = open(log_dir / f"{name}.log", "a")
                running[name] = (subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=out, stderr=subprocess.STDOUT,
                                                  env=os.environ.copy()), out)

            for name, (proc, out) in list(running.items()):
                if proc.poll() is not None:
                    out.close()
                    exit_codes[name] = proc.returncode
                    del running[name]
                    if proc.returncode != 0:
                        print(f"[parallel] worker {name} exited with {proc.returncode} (see workers/{name}.log)")
            failed = sum(code != 0 for code in exit_codes.values())
            if failed >= MAX_FAILED_WORKERS and stop_reason is None:
                stop_reason = f"{failed} workers failed; not launching more"
                print(f"[parallel] {stop_reason}")

            if time.monotonic() - last_poll >= POLL_S or not (queue or running):
                last_poll = time.monotonic()
                spent, n_logged = spent_since(experiment, tasks, since)
                if is_openrouter:
                    key_now = openrouter_key_status(llm_config.API_KEY)
                    if key_before and key_now and "usage" in key_before and "usage" in key_now:
                        key_spent = key_now["usage"] - key_before["usage"]
                print(f"[parallel] {utc_now()[11:19]} runs logged {n_logged}/{total_runs} | logged cost ${spent:.4f}"
                      f" | key spend ${key_spent if key_spent is not None else float('nan'):.4f}"
                      f" | running {len(running)} queued {len(queue)} done {len(exit_codes)}")
                if max(spent, key_spent or 0.0) >= max_cost_usd:
                    stop_all(f"budget reached (logged ${spent:.4f}, key ${key_spent or 0:.4f} >= ${max_cost_usd})")
            time.sleep(2)
    except KeyboardInterrupt:
        stop_all("interrupted by user")
        for proc, _ in running.values():
            proc.wait()
        raise
    finally:
        key_after = openrouter_key_status(llm_config.API_KEY) if is_openrouter else None
        append_jsonl(experiment_dir(experiment) / "sessions.jsonl", {
            "type": "parallel_launch",
            "launch_id": launch_id,
            "experiment": experiment,
            "started_at": since,
            "finished_at": utc_now(),
            "stop_reason": stop_reason or "completed",
            "workers": workers,
            "shard_size": shard_size,
            "max_cost_usd": max_cost_usd,
            "runs_planned": total_runs,
            "runs_logged": n_logged,
            "logged_cost_usd": spent,
            "openrouter_key_before": key_before,
            "openrouter_key_after": key_after,
            "worker_exit_codes": exit_codes,
        })
    print(f"[parallel] finished: {stop_reason or 'completed'} | logged {n_logged}/{total_runs} runs, "
          f"${spent:.4f} | failed workers: {[n for n, c in exit_codes.items() if c != 0]}")
    return 0 if stop_reason is None and all(c == 0 for c in exit_codes.values()) else 1
