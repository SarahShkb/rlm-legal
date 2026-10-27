"""Parallel worker processes for one or more datasets under ONE global budget.

Same behaviour as lexglue/parallel.py: unfinished slices only (resume), the
budget is checked every POLL_S seconds against the cost of the records logged
since launch and the OpenRouter key's spend since launch (SIGINT to every
worker when reached), and no new workers after MAX_FAILED_WORKERS failures.
"""

import os
import signal
import subprocess
import sys
import time
import uuid

from llm import config as llm_config
from lexglue.pipeline import append_jsonl, read_jsonl
from lexglue.tracking import REPO_ROOT, openrouter_key_status, utc_now

from benchmarks.runner import (
    TIMEOUT_EXIT_CODE,
    Dataset,
    check_manifest,
    completed_keys,
    experiment_dir,
    llm_settings,
    prompt_fingerprint,
)

POLL_S = 20
MAX_FAILED_WORKERS = 3


def spent_since(datasets: list[Dataset], experiment: str, since: str) -> tuple[float, int]:
    total, n = 0.0, 0
    for ds in datasets:
        for r in read_jsonl(experiment_dir(ds, experiment) / "records" / f"{ds.name}.jsonl"):
            if r["started_at"] >= since:
                cost = r["reported_cost_usd"] if r["reported_cost_usd"] is not None else r["estimated_cost_usd"]
                total += cost or 0.0
                n += 1
    return total, n


def run_parallel(datasets: list[Dataset], experiment: str, methods_by_dataset: dict[str, list[str]],
                 indices_by_dataset: dict[str, list[int]], repeats: int, max_iterations: int,
                 max_cost_usd: float, workers: int, shard_size: int) -> int:
    shards = []
    for ds in datasets:
        check_manifest(experiment_dir(ds, experiment), llm_settings(max_iterations), prompt_fingerprint(ds), False)
        done = completed_keys(ds, experiment)
        methods = methods_by_dataset[ds.name]
        todo = [i for i in indices_by_dataset[ds.name]
                if any((i, m, r) not in done for m in methods for r in range(repeats))]
        shards += [(ds, todo[k:k + shard_size]) for k in range(0, len(todo), shard_size)]
    total_runs = sum(len(idx) * len(methods_by_dataset[ds.name]) for ds, idx in shards) * repeats

    is_openrouter = llm_config.LLM_PROVIDER == "openrouter"
    launch_id, since = uuid.uuid4().hex[:12], utc_now()
    key_before = openrouter_key_status(llm_config.API_KEY) if is_openrouter else None
    print(f"[parallel] {launch_id}: {len(shards)} shards, {workers} workers, <= {total_runs} runs, "
          f"budget ${max_cost_usd}", flush=True)

    queue, running, exit_codes = list(shards), {}, {}
    stop_reason, last_poll, spent, n_logged, key_spent = None, 0.0, 0.0, 0, None

    def stop_all(reason):
        nonlocal stop_reason
        if stop_reason is None:
            stop_reason = reason
            print(f"[parallel] STOP: {reason} -> interrupting {len(running)} workers", flush=True)
            for proc, *_ in running.values():
                if proc.poll() is None:
                    proc.send_signal(signal.SIGINT)

    try:
        while running or (queue and stop_reason is None):
            while queue and len(running) < workers and stop_reason is None:
                ds, idx = queue.pop(0)
                name = f"{ds.name}_{idx[0]:04d}-{idx[-1]:04d}"
                cmd = [sys.executable, "-m", "benchmarks.cli", "run", "--dataset", ds.name,
                       "--experiment", experiment, "--indices", *map(str, idx),
                       "--methods", *methods_by_dataset[ds.name], "--repeats", str(repeats),
                       "--max-iterations", str(max_iterations), "--max-cost", str(max_cost_usd), "--yes"]
                log_dir = experiment_dir(ds, experiment) / "workers"
                log_dir.mkdir(parents=True, exist_ok=True)
                out = open(log_dir / f"{name}.log", "a")
                running[name] = (subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=out, stderr=subprocess.STDOUT,
                                                  env=os.environ.copy()), out, (ds, idx))
            for name, (proc, out, shard) in list(running.items()):
                if proc.poll() is not None:
                    out.close()
                    del running[name]
                    if proc.returncode == TIMEOUT_EXIT_CODE and stop_reason is None:
                        # the timed-out run is recorded as failed; resume the rest of the slice
                        print(f"[parallel] worker {name} hard-exited on a run timeout; restarting its slice", flush=True)
                        queue.insert(0, shard)
                        continue
                    exit_codes[name] = proc.returncode
                    if proc.returncode != 0:
                        print(f"[parallel] worker {name} exited with {proc.returncode}", flush=True)
            failed = sum(c != 0 for c in exit_codes.values())
            if failed >= MAX_FAILED_WORKERS and stop_reason is None:
                stop_reason = f"{failed} workers failed; not launching more"
                print(f"[parallel] {stop_reason}", flush=True)
            if time.monotonic() - last_poll >= POLL_S or not (queue or running):
                last_poll = time.monotonic()
                spent, n_logged = spent_since(datasets, experiment, since)
                if is_openrouter:
                    key_now = openrouter_key_status(llm_config.API_KEY)
                    if key_before and key_now and "usage" in key_before and "usage" in key_now:
                        key_spent = key_now["usage"] - key_before["usage"]
                print(f"[parallel] {utc_now()[11:19]} runs {n_logged}/{total_runs} | logged ${spent:.4f} | key "
                      f"${key_spent if key_spent is not None else float('nan'):.4f} | running {len(running)} "
                      f"queued {len(queue)} done {len(exit_codes)}", flush=True)
                if max(spent, key_spent or 0.0) >= max_cost_usd:
                    stop_all(f"budget reached (logged ${spent:.4f}, key ${key_spent or 0:.4f} >= ${max_cost_usd})")
            time.sleep(2)
    except KeyboardInterrupt:
        stop_all("interrupted by user")
        for proc, *_ in running.values():
            proc.wait()
        raise
    finally:
        key_after = openrouter_key_status(llm_config.API_KEY) if is_openrouter else None
        for ds in datasets:
            append_jsonl(experiment_dir(ds, experiment) / "sessions.jsonl", {
                "type": "parallel_launch", "launch_id": launch_id, "experiment": experiment,
                "datasets": [d.name for d in datasets], "started_at": since, "finished_at": utc_now(),
                "stop_reason": stop_reason or "completed", "workers": workers, "shard_size": shard_size,
                "max_cost_usd": max_cost_usd, "runs_planned": total_runs, "runs_logged": n_logged,
                "logged_cost_usd": spent, "openrouter_key_before": key_before, "openrouter_key_after": key_after,
                "worker_exit_codes": exit_codes,
            })
    print(f"[parallel] finished: {stop_reason or 'completed'} | logged {n_logged}/{total_runs} runs, ${spent:.4f} "
          f"| failed workers: {[n for n, c in exit_codes.items() if c != 0]}", flush=True)
    return 0 if stop_reason is None and all(c == 0 for c in exit_codes.values()) else 1
