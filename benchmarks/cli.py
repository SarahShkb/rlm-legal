"""Contract-NLI and MAUD re-runs with full logging. Run from the repo root.

  # free checks (no LLM call): data, prompts, tools, parsers
  python -m benchmarks.cli preflight

  # both datasets in parallel workers, one global budget (OpenRouter: asks for confirmation)
  LLM_PROVIDER=openrouter python -m benchmarks.cli run-parallel --datasets contract_nli maud \
      --experiment rerun_v1_morph --max-cost 15 --workers 24

  # tables -> contract-nli/results/<experiment>/report/, maud/results/<experiment>/report/
  python -m benchmarks.cli report --dataset maud --experiment rerun_v1_morph

Re-running the same command resumes; finished (item, method, repeat) runs are skipped.
"""

import argparse
import sys

import pandas as pd


def get_dataset(name: str):
    if name == "contract_nli":
        from benchmarks.contract_nli_bench import ContractNLI
        return ContractNLI()
    if name == "maud":
        from benchmarks.maud_bench import Maud
        return Maud()
    raise SystemExit(f"unknown dataset {name!r}")


def _confirm(args, plan: str) -> None:
    from llm import config as llm_config
    print(f"Plan: provider={llm_config.LLM_PROVIDER} model={llm_config.MODEL} extra_body={llm_config.EXTRA_BODY}\n{plan}")
    if llm_config.LLM_PROVIDER == "openrouter":
        if args.max_cost is None:
            sys.exit("Refusing to use the OpenRouter key without --max-cost.")
        if not args.yes and input("This spends OpenRouter credit. Type 'yes' to start: ").strip() != "yes":
            sys.exit("Aborted.")


def cmd_preflight(args) -> None:
    from benchmarks.maud_bench import TOOLS, build_tools, tool_hints
    from legal_tools.tools_utils import ToolLog
    from rlm.utils.prompts import RLM_SYSTEM_PROMPT

    problems = []
    for name in args.datasets:
        ds = get_dataset(name)
        items = ds.items(ds.all_indices())
        chars = [len(i.document) for i in items]
        print(f"== {name}: {len(items)} items, methods {ds.methods}, document chars mean "
              f"{sum(chars) / len(chars):,.0f} max {max(chars):,}")
        for it in items:
            # the RLM system prompt is str.format()-ed; the task spec must survive it
            try:
                (RLM_SYSTEM_PROMPT + it.task_spec).format(custom_tools_section="")
            except (KeyError, IndexError, ValueError) as e:
                problems.append(f"{it.row_id}: task spec breaks str.format(): {e}")
            if name == "contract_nli":
                ok = ds.score(it, __import__("json").dumps({h: {"choice": g} for h, g in it.gold.items()}))
                if ok["n_correct_hypotheses"] != 17:
                    problems.append(f"{it.row_id}: gold does not round-trip through the scorer")
            else:
                ok = ds.score(it, '{"answer": "%s"}' % ", ".join(it.gold))
                if not ok["correct"]:
                    problems.append(f"{it.row_id}: gold {it.gold} does not round-trip through parse_raw_result")
        if name == "maud":
            # tools build on every contract (no API call)
            for it in items:
                for method, which in TOOLS.items():
                    if which:
                        try:
                            build_tools(it.document, which, ToolLog(), list_in_prompt=True)
                            tool_hints(which)
                        except Exception as e:  # noqa: BLE001
                            problems.append(f"{it.row_id} {method}: tools failed: {type(e).__name__}: {e}")
        if args.show_prompts:
            print("-" * 80 + f"\n{items[0].task_spec}\n" + "-" * 80)
    print("== Result:", "OK" if not problems else f"{len(problems)} PROBLEM(S)")
    for p in problems[:50]:
        print("  -", p)
    if problems:
        sys.exit(1)


def cmd_run(args) -> None:
    from benchmarks.runner import run_experiment
    ds = get_dataset(args.dataset)
    indices = args.indices or ds.all_indices()
    methods = args.methods or list(ds.methods)
    _confirm(args, f"      dataset={ds.name} experiment={args.experiment} items={len(indices)} methods={methods} "
                   f"repeats={args.repeats} -> {len(indices) * len(methods) * args.repeats} runs, budget ${args.max_cost}")
    run_experiment(ds, args.experiment, indices, methods, repeats=args.repeats,
                   max_iterations=args.max_iterations, max_cost_usd=args.max_cost,
                   allow_config_change=args.allow_config_change)


def cmd_run_parallel(args) -> None:
    from benchmarks.parallel import run_parallel
    datasets = [get_dataset(n) for n in args.datasets]
    methods = {d.name: list(d.methods) for d in datasets}
    indices = {d.name: d.all_indices() for d in datasets}
    n = sum(len(indices[d.name]) * len(methods[d.name]) for d in datasets) * args.repeats
    _confirm(args, "      " + "; ".join(f"{d.name}: {len(indices[d.name])} items x {methods[d.name]}" for d in datasets)
             + f"\n      repeats={args.repeats} -> {n} runs, workers={args.workers}, global budget ${args.max_cost}")
    sys.exit(run_parallel(datasets, args.experiment, methods, indices, args.repeats, args.max_iterations,
                          args.max_cost, args.workers, args.shard_size))


def cmd_report(args) -> None:
    from benchmarks.metrics import build_report
    from benchmarks.runner import experiment_dir
    ds = get_dataset(args.dataset)
    tables = build_report(experiment_dir(ds, args.experiment), ds.name)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    for name, t in tables.items():
        if name == "efficiency_cost":
            cols = [c for c in ("method", "n_runs", "n_calls_mean", "n_sub_calls_mean", "total_tokens_mean",
                                "reasoning_tokens_mean", "wall_time_s_mean", "reported_cost_usd_sum",
                                "rlm_iterations_mean", "rlm_hit_max_iterations_rate", "n_empty_responses",
                                "n_truncated_calls") if c in t]
            t = t[cols]
        print(f"\n== {name}")
        print(t.round(4).to_string(index=False) if not t.empty else "(empty)")
    print(f"\nCSV tables written to {experiment_dir(ds, args.experiment) / 'report'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("preflight")
    s.add_argument("--datasets", nargs="+", default=["contract_nli", "maud"])
    s.add_argument("--show-prompts", action="store_true")
    s.set_defaults(func=cmd_preflight)

    for name, func in (("run", cmd_run), ("run-parallel", cmd_run_parallel)):
        s = sub.add_parser(name)
        if name == "run":
            s.add_argument("--dataset", required=True, choices=["contract_nli", "maud"])
            s.add_argument("--indices", type=int, nargs="+")
            s.add_argument("--methods", nargs="+")
            s.add_argument("--allow-config-change", action="store_true")
        else:
            s.add_argument("--datasets", nargs="+", required=True, choices=["contract_nli", "maud"])
            s.add_argument("--workers", type=int, default=24)
            s.add_argument("--shard-size", type=int, default=4, help="items per worker")
        s.add_argument("--experiment", required=True)
        s.add_argument("--repeats", type=int, default=1)
        s.add_argument("--max-iterations", type=int, default=8)
        s.add_argument("--max-cost", type=float)
        s.add_argument("--yes", action="store_true")
        s.set_defaults(func=func)

    s = sub.add_parser("report")
    s.add_argument("--dataset", required=True, choices=["contract_nli", "maud"])
    s.add_argument("--experiment", required=True)
    s.set_defaults(func=cmd_report)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
