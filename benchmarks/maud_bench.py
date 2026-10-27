"""MAUD 152-sample set (one question per contract) for the benchmarks runner.

Sample, prompt, tools and answer parsing are those of the existing MAUD code:
get_sample_rows_for_maud_152 / build_query (maud/maud_data_sampling.py),
build_tools / tool_hints (legal_tools/run_tools.py) and parse_raw_result
(maud/maud_eval.py). Methods:
  vanilla     run_vanilla
  rlm         run_rlm_customized, no tools
  rlm_tool1   + tool 1 (section locator)
  rlm_tool2   + tool 2 (definitions resolver)
  rlm_tool12  + both tools
Every tool condition uses the latest tool setup (maud_152_tool2_fixed_list_hints):
usage hints in the root prompt, tool 2's defined-term list in the system prompt.
Must run from the repo root (the MAUD code uses relative paths).
"""

import json
from functools import cached_property

from legal_tools.run_tools import build_tools, tool_hints
from legal_tools.tools_utils import ToolLog
from llm.config import TEMPERATURE
from llm.rlm import run_rlm_customized
from llm.vanila import run_vanilla
from lexglue.tracking import REPO_ROOT
from maud.maud_data_sampling import build_query, get_maud_data_dir, get_sample_rows_for_maud_152
from maud.maud_eval import parse_raw_result

from benchmarks.runner import Dataset, Item

TOOLS = {"rlm": set(), "rlm_tool1": {"1"}, "rlm_tool2": {"2"}, "rlm_tool12": {"1", "2"}}
LIST_IN_PROMPT = True
HINTS = True


class Maud(Dataset):
    name = "maud"
    results_root = REPO_ROOT / "maud" / "results"
    methods = ("vanilla", "rlm", "rlm_tool1", "rlm_tool2", "rlm_tool12")
    source_files = [
        "maud/maud_data_sampling.py", "maud/maud_eval.py", "maud/dicts/question_full_text_map.json",
        "maud/dicts/question_answer_map_unique.json", "legal_tools/run_tools.py",
        "legal_tools/section_locator.py", "legal_tools/definitions_resolver.py",
        "legal_tools/tools_utils.py", "benchmarks/maud_bench.py",
    ]

    @cached_property
    def _sample(self):
        return get_sample_rows_for_maud_152(split="train")

    def all_indices(self) -> list[int]:
        return list(range(len(self._sample)))

    def items(self, indices: list[int]) -> list[Item]:
        with open("maud/dicts/question_full_text_map.json", encoding="utf-8") as f:
            full_text = json.load(f)
        with open("maud/dicts/question_answer_map_unique.json", encoding="utf-8") as f:
            options = json.load(f)
        contracts_dir = f"{get_maud_data_dir()}/contracts"
        out = []
        for i in indices:
            row = self._sample.iloc[i]
            with open(f"{contracts_dir}/{row.contract_name}.txt", encoding="utf-8-sig") as f:
                text = f.read()
            opts = options[row.question]
            out.append(Item(
                index=i,
                row_id=f"{row.contract_name}_{i}",   # same row_id format as the earlier MAUD runs
                document=text,
                task_spec=build_query(question_text=full_text[row.question],
                                      multi_select=bool(row.is_multi_select), options=opts),
                gold=sorted(row.gold_letters),
                meta={"contract_name": row.contract_name, "question": row.question,
                      "is_multi_select": bool(row.is_multi_select), "n_options": len(opts),
                      "n_tokens_contract": int(row.n_tokens_contract),
                      "complexity_tier": str(getattr(row, "complexity_tier", ""))},
            ))
        return out

    def call(self, item, method, *, experiment, max_iterations, runs_csv, log_dir, on_result):
        if method == "vanilla":
            return run_vanilla("maud_152_samples", item.row_id, item.document, TASK_SPEC=item.task_spec,
                               experiment=experiment, temperature=TEMPERATURE, runs_csv=runs_csv), {}
        which = TOOLS[method]
        log = ToolLog() if which else None
        tools = build_tools(item.document, which, log, list_in_prompt=LIST_IN_PROMPT) if which else None
        response = run_rlm_customized(
            "maud_152_samples", item.row_id, item.document, TASK_SPEC=item.task_spec,
            experiment=experiment, max_iterations=max_iterations, tools=tools,
            tool_hint=tool_hints(which) if which and HINTS else "",
            on_result=on_result, runs_csv=runs_csv, log_dir=log_dir,
        )
        extra = {"tools": sorted(which)}
        if log is not None:
            extra["tool_log_summary"] = log.summary()
            extra["tool_log"] = json.loads(log.to_json())
        return response, extra

    def score(self, item, response):
        parsed = parse_raw_result(response or "")
        pred = parsed.letters
        gold = frozenset(item.gold)
        exact = bool(pred) and pred == gold
        return {
            "prediction": sorted(pred) if pred else None,
            "parse_status": parsed.status,     # final | repl_recovered | repl_unresolved | unparsed | empty ...
            "correct": exact,                  # lenient (as maud_eval): recovered answers count
            "correct_strict": exact and parsed.status == "final",
            "jaccard": len(pred & gold) / len(pred | gold) if pred else 0.0,
            "score": float(exact),
        }
