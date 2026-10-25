"""LexGLUE: vanilla LLM vs RLM (custom system prompt). Run from the repo root.

  # 1. free checks, no LLM call: config, data, prompts, parser, OpenRouter endpoint/credit
  python -m lexglue.run_lexglue preflight [--show-prompts]

  # 2. pilot on your own DeepSeek key (LLM_PROVIDER=deepseek, the default)
  python -m lexglue.run_lexglue run --experiment pilot_deepseek --n 3 --max-cost 0.5

  # 3. cost projection of the full run from the pilot, at OpenRouter prices
  python -m lexglue.run_lexglue estimate --experiment pilot_deepseek \
      --project-model deepseek/deepseek-v4-flash-0731 --project-provider deepinfra/fp8

  # 4. full run on the OpenRouter key (asks for confirmation, stops at --max-cost)
  LLM_PROVIDER=openrouter python -m lexglue.run_lexglue run --experiment lexglue_v1 --max-cost 10
  #    ... or the same in parallel worker processes, one global budget
  LLM_PROVIDER=openrouter python -m lexglue.run_lexglue run-parallel --experiment lexglue_v1 \
      --max-cost 10 --workers 12 --shard-size 10

  # 5. metrics, paired tests, cost/efficiency tables -> lexglue/results/<experiment>/report/
  python -m lexglue.run_lexglue report --experiment lexglue_v1

A crashed or interrupted run is resumed by re-running the same command.
"""

import argparse
import json
import sys

import pandas as pd

from lexglue.config import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_N,
    DEFAULT_SPLIT,
    DEFAULT_START,
    METHODS,
    MULTIPLE_CHOICE,
    RESULTS_DIR,
    TASKS,
)


def _indices(args) -> list[int]:
    return args.indices if args.indices else list(range(args.start, args.start + args.n))


def _answer_json(task: str, gold) -> str:
    task_type = TASKS[task]["task_type"]
    key = {"multi_class": "label", "multi_label": "labels", "multiple_choice": "answer"}[task_type]
    return json.dumps({key: gold})


def cmd_preflight(args) -> None:
    from llm import config as llm_config
    from lexglue.data import label_names, load_records, load_split
    from lexglue.parsing import is_correct, parse_prediction
    from lexglue.prompts import build_task_spec
    from lexglue.tracking import openrouter_endpoint, openrouter_key_status

    problems = []
    print("== LLM configuration (llm/config.py)")
    print(f"  provider={llm_config.LLM_PROVIDER} model={llm_config.MODEL} base_url={llm_config.BASE_URL}")
    print(f"  extra_body={llm_config.EXTRA_BODY} temperature={llm_config.TEMPERATURE}")
    print(f"  api key set: {bool(llm_config.API_KEY)}")
    if not llm_config.API_KEY:
        problems.append("API key missing")

    indices = _indices(args)
    print(f"\n== Data and prompts: split={args.split}, {len(indices)} records/task (indices {indices[0]}..{indices[-1]})")
    total_chars = 0
    for task in args.tasks:
        ds = load_split(task, args.split)
        labels = None if TASKS[task]["task_type"] == MULTIPLE_CHOICE else label_names(task, ds.features)
        records = load_records(task, args.split, indices)
        chars = [len(r.document) + len(build_task_spec(r, labels)) for r in records]
        total_chars += sum(chars)
        for r in records:
            spec = build_task_spec(r, labels)
            # RLM str.format()s the system prompt: the escaped spec must come back unchanged
            escaped = spec.replace("{", "{{").replace("}", "}}")
            if escaped.format() != spec:
                problems.append(f"{r.row_id}: task spec does not survive str.format()")
            # the gold answer, written in the requested JSON format, must parse back to itself
            parsed = parse_prediction(task, labels, len(r.options), _answer_json(task, r.gold))
            if parsed.status != "ok" or not is_correct(task, parsed.prediction, r.gold):
                problems.append(f"{r.row_id}: gold {r.gold!r} does not round-trip through the parser")
            if labels is not None:
                gold = r.gold if isinstance(r.gold, list) else [r.gold]
                if not set(gold) <= set(labels):
                    problems.append(f"{r.row_id}: gold label not among candidate labels")
        print(f"  {task:11s} records={len(records)} labels={len(labels) if labels else 5} "
              f"approx_input_tokens/record(chars/4): mean={sum(chars) / len(chars) / 4:,.0f} max={max(chars) / 4:,.0f}")
        if args.show_prompts:
            print("-" * 80 + f"\n{build_task_spec(records[0], labels)}\n" + "-" * 80)
    print(f"  total single-pass input ~{total_chars / 4:,.0f} tokens (vanilla reads each document once)")

    if llm_config.LLM_PROVIDER == "openrouter":
        print("\n== OpenRouter (free endpoints, no completion is requested)")
        endpoint = openrouter_endpoint(llm_config.MODEL, llm_config.OPENROUTER_PROVIDER)
        if endpoint is None:
            problems.append(f"no endpoint {llm_config.OPENROUTER_PROVIDER!r} for {llm_config.MODEL!r}")
        else:
            print(f"  endpoint: {json.dumps(endpoint, indent=2)}")
            if "reasoning" not in (endpoint.get("supported_parameters") or []):
                problems.append("pinned endpoint does not list 'reasoning' as supported")
        key = openrouter_key_status(llm_config.API_KEY)
        print(f"  key status: {key}")
        if key and key.get("error"):
            problems.append(f"key check failed: {key['error']}")

    print("\n== Result:", "OK" if not problems else f"{len(problems)} PROBLEM(S)")
    for p in problems[:50]:
        print("  -", p)
    if problems:
        sys.exit(1)


def cmd_run(args) -> None:
    from llm import config as llm_config
    from lexglue.pipeline import run_experiment

    indices = _indices(args)
    n_runs = len(args.tasks) * len(indices) * len(args.methods) * args.repeats
    print(f"Plan: experiment={args.experiment} provider={llm_config.LLM_PROVIDER} model={llm_config.MODEL}\n"
          f"      extra_body={llm_config.EXTRA_BODY}\n"
          f"      tasks={args.tasks}\n      methods={args.methods} repeats={args.repeats} "
          f"records/task={len(indices)} -> {n_runs} runs, RLM max_iterations={args.max_iterations}, "
          f"budget={'none' if args.max_cost is None else f'${args.max_cost}'}")
    if llm_config.LLM_PROVIDER == "openrouter":
        if args.max_cost is None:
            sys.exit("Refusing to use the OpenRouter key without --max-cost.")
        if not args.yes and input("This spends OpenRouter credit. Type 'yes' to start: ").strip() != "yes":
            sys.exit("Aborted.")
    run_experiment(
        experiment=args.experiment,
        tasks=args.tasks,
        methods=args.methods,
        split=args.split,
        indices=indices,
        repeats=args.repeats,
        max_iterations=args.max_iterations,
        max_cost_usd=args.max_cost,
        price_per_mtok=tuple(args.price_per_mtok) if args.price_per_mtok else None,
        allow_config_change=args.allow_config_change,
        store_text=not args.no_store_text,
    )


def cmd_run_parallel(args) -> None:
    from llm import config as llm_config
    from lexglue.parallel import run_parallel

    indices = _indices(args)
    n_runs = len(args.tasks) * len(indices) * len(args.methods) * args.repeats
    print(f"Plan: experiment={args.experiment} provider={llm_config.LLM_PROVIDER} model={llm_config.MODEL}\n"
          f"      extra_body={llm_config.EXTRA_BODY}\n"
          f"      tasks={args.tasks}\n      methods={args.methods} repeats={args.repeats} "
          f"records/task={len(indices)} -> {n_runs} runs, RLM max_iterations={args.max_iterations}, "
          f"workers={args.workers}, shard size={args.shard_size}, global budget=${args.max_cost}")
    if llm_config.LLM_PROVIDER == "openrouter" and not args.yes \
            and input("This spends OpenRouter credit. Type 'yes' to start: ").strip() != "yes":
        sys.exit("Aborted.")
    sys.exit(run_parallel(
        experiment=args.experiment,
        tasks=args.tasks,
        methods=args.methods,
        split=args.split,
        indices=indices,
        repeats=args.repeats,
        max_iterations=args.max_iterations,
        max_cost_usd=args.max_cost,
        workers=args.workers,
        shard_size=args.shard_size,
        store_text=not args.no_store_text,
    ))


def _projection_pricing(args) -> dict | None:
    if not args.project_model:
        return None
    from lexglue.tracking import openrouter_endpoint
    endpoint = openrouter_endpoint(args.project_model, args.project_provider)
    if endpoint is None:
        sys.exit(f"No OpenRouter endpoint {args.project_provider!r} for {args.project_model!r}")
    print(f"Projection prices (per token): {endpoint['pricing_per_token']} "
          f"[{endpoint['provider_name']}, {endpoint['quantization']}]")
    return endpoint["pricing_per_token"]


def cmd_report(args) -> None:
    from lexglue.metrics import build_report

    exp_dir = RESULTS_DIR / args.experiment
    tables = build_report(exp_dir, args.split, _projection_pricing(args))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    q = tables["quality"]
    print("\n== Quality (95% bootstrap CI)")
    print(q[["task", "method", "repeat", "n", "n_failed_answers", "accuracy", "micro_f1",
             "micro_f1_ci_low", "micro_f1_ci_high", "macro_f1"]].round(4).to_string(index=False))
    if not tables["paired_vanilla_vs_rlm"].empty:
        print("\n== Vanilla vs RLM (paired)")
        print(tables["paired_vanilla_vs_rlm"].round(4).to_string(index=False))
    e = tables["efficiency_cost"]
    cols = [c for c in ("task", "method", "n_runs", "n_calls_mean", "total_tokens_mean", "reasoning_tokens_mean",
                        "wall_time_s_mean", "reported_cost_usd_sum", "estimated_cost_usd_sum",
                        "projected_cost_usd_sum", "rlm_iterations_mean", "rlm_hit_max_iterations_rate")
            if c in e]
    print("\n== Efficiency and cost")
    print(e[cols].round(5).to_string(index=False))
    print("\n== Overall")
    print(tables["overall"].round(5).to_string(index=False))
    print(f"\nCSV tables written to {exp_dir / 'report'}")


def cmd_estimate(args) -> None:
    """Scale a pilot's per-run cost to the full experiment."""
    from lexglue.metrics import efficiency_table, load_records

    pricing = _projection_pricing(args)
    if pricing is None:
        sys.exit("--project-model and --project-provider are required")
    df = load_records(RESULTS_DIR / args.experiment)
    e = efficiency_table(df, pricing)
    df = df.assign(projected=(
        (df["prompt_tokens"].fillna(0) - df["cached_prompt_tokens"].fillna(0)) * pricing["prompt"]
        + df["cached_prompt_tokens"].fillna(0) * (pricing["input_cache_read"] or pricing["prompt"])
        + df["completion_tokens"].fillna(0) * pricing["completion"]))
    per = df.groupby(["task", "method"])["projected"].agg(["mean", "max"]).reset_index()
    per["full_run_mean"] = per["mean"] * args.n * args.repeats
    per["full_run_if_all_like_max"] = per["max"] * args.n * args.repeats
    print(f"\nProjected cost of {args.n} records/task x {args.repeats} repeat(s), "
          f"from {len(df)} pilot runs:")
    print(per.round(5).to_string(index=False))
    total, worst = per["full_run_mean"].sum(), per["full_run_if_all_like_max"].sum()
    print(f"\nTOTAL expected ~${total:.2f}   (pessimistic, every run as costly as the worst pilot run: ${worst:.2f})")
    missing = sorted(set(TASKS) - set(df["task"]))
    if missing:
        print(f"Not in the pilot (not included above): {missing}")
    print(f"\nPilot wall time: mean {e['wall_time_s_mean'].mean():.1f}s per run")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_sample_args(p):
        p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
        p.add_argument("--split", default=DEFAULT_SPLIT)
        p.add_argument("--start", type=int, default=DEFAULT_START)
        p.add_argument("--n", type=int, default=DEFAULT_N, help="records per task")
        p.add_argument("--indices", type=int, nargs="+", help="explicit record indices (overrides --start/--n)")

    p = sub.add_parser("preflight", help="free checks, no LLM call")
    add_sample_args(p)
    p.add_argument("--show-prompts", action="store_true")
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("run", help="run the experiment (resumes if interrupted)")
    add_sample_args(p)
    p.add_argument("--experiment", required=True)
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    p.add_argument("--max-cost", type=float, help="stop the session when its cost (USD) reaches this")
    p.add_argument("--price-per-mtok", type=float, nargs=3, metavar=("IN", "OUT", "CACHE_READ"),
                   help="USD per 1M tokens for estimated_cost_usd (OpenRouter prices are fetched automatically)")
    p.add_argument("--allow-config-change", action="store_true")
    p.add_argument("--no-store-text", action="store_true", help="don't keep answer/reasoning text per call")
    p.add_argument("--yes", action="store_true", help="skip the OpenRouter confirmation prompt")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("run-parallel", help="run in parallel worker processes under one global budget")
    add_sample_args(p)
    p.add_argument("--experiment", required=True)
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    p.add_argument("--max-cost", type=float, required=True, help="global budget (USD) for this launch")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--shard-size", type=int, default=10, help="records per worker")
    p.add_argument("--no-store-text", action="store_true")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_run_parallel)

    for name, func in (("report", cmd_report), ("estimate", cmd_estimate)):
        p = sub.add_parser(name)
        p.add_argument("--experiment", required=True)
        p.add_argument("--split", default=DEFAULT_SPLIT)
        p.add_argument("--project-model", help="OpenRouter model id for projected cost")
        p.add_argument("--project-provider", help="OpenRouter endpoint tag, e.g. deepinfra/fp8")
        if name == "estimate":
            p.add_argument("--n", type=int, default=DEFAULT_N)
            p.add_argument("--repeats", type=int, default=1)
        p.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
