"""Contract-NLI (test split, 123 NDAs x 17 hypotheses) for the benchmarks runner.

Prompt and record loading are the ones of utils/contract_nli_utils.py
(run_contract_nli_pipeline): CONTRACT_NLI_TASK_SPEC + all 17 hypotheses, one
call per document, no brace escaping (the spec already doubles its braces).
Scoring: every hypothesis of every document counts; a hypothesis that is
missing from the answer, or has an unknown label, is wrong. Answer keys are
matched after stripping surrounding punctuation/space: build_hypotheses_block
lists hypotheses as "-nda-11: ...", and models sometimes copy the dash
("-nda-11"); parse_status "normalized_keys" records when that was needed.
"""

import json
import re
from functools import cached_property

from llm.config import TEMPERATURE
from llm.rlm import run_rlm_customized
from llm.vanila import run_vanilla
from lexglue.parsing import _json_candidates
from lexglue.tracking import REPO_ROOT
from prompts.prompts import CONTRACT_NLI_TASK_SPEC
from utils.contract_nli_utils import build_hypotheses_block, load_one_record

from benchmarks.runner import Dataset, Item

LABELS = ("Entailment", "Contradiction", "NotMentioned")
_LABEL_LOOKUP = {re.sub(r"[^a-z]", "", l.lower()): l for l in LABELS}


def normalize_key(key) -> str:
    return re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", str(key).strip().lower())


def normalize_label(value) -> str | None:
    if isinstance(value, dict):
        value = value.get("choice")
    if not isinstance(value, str):
        return None
    return _LABEL_LOOKUP.get(re.sub(r"[^a-z]", "", value.lower()))


class ContractNLI(Dataset):
    name = "contract_nli"
    results_root = REPO_ROOT / "contract-nli" / "results"
    methods = ("vanilla", "rlm")
    source_files = ["prompts/prompts.py", "utils/contract_nli_utils.py", "benchmarks/contract_nli_bench.py"]
    path = str(REPO_ROOT / "contract-nli" / "main_repo" / "test.json")

    @cached_property
    def _data(self) -> dict:
        with open(self.path) as f:
            return json.load(f)

    def all_indices(self) -> list[int]:
        return list(range(len(self._data["documents"])))

    def items(self, indices: list[int]) -> list[Item]:
        task_spec = f"{CONTRACT_NLI_TASK_SPEC}\n\n{build_hypotheses_block(self.path)}\n\n"
        out = []
        for i in indices:
            row_id, text = load_one_record(self.path, doc_index=i)
            doc = self._data["documents"][i]
            gold = {h: a["choice"] for h, a in doc["annotation_sets"][0]["annotations"].items()}
            out.append(Item(index=i, row_id=row_id, document=text, task_spec=task_spec, gold=gold,
                            meta={"doc_id": doc["id"], "file_name": doc.get("file_name")}))
        return out

    def call(self, item, method, *, experiment, max_iterations, runs_csv, log_dir, on_result):
        if method == "vanilla":
            return run_vanilla("contract-nli", item.row_id, item.document, TASK_SPEC=item.task_spec,
                               experiment=experiment, temperature=TEMPERATURE, runs_csv=runs_csv), {}
        if method == "rlm":
            return run_rlm_customized("contract-nli", item.row_id, item.document, TASK_SPEC=item.task_spec,
                                      experiment=experiment, max_iterations=max_iterations,
                                      on_result=on_result, runs_csv=runs_csv, log_dir=log_dir), {}
        raise ValueError(f"unknown method {method!r}")

    def score(self, item, response):
        answer = None
        for obj in _json_candidates(response or ""):
            if isinstance(obj, dict) and any(normalize_key(k).startswith("nda-") for k in obj):
                answer = obj
                break
        if answer is None:
            prediction = {h: None for h in item.gold}
            status = "unparseable"
        else:
            by_key = {normalize_key(k): v for k, v in answer.items()}
            prediction = {h: normalize_label(by_key.get(h)) for h in item.gold}
            n_valid = sum(v is not None for v in prediction.values())
            status = "ok" if n_valid == len(item.gold) else ("partial" if n_valid else "invalid_labels")
            if status == "ok" and any(str(k) != normalize_key(k) for k in answer):
                status = "normalized_keys"
        n_correct = sum(prediction[h] == g for h, g in item.gold.items())
        return {
            "prediction": prediction,
            "parse_status": status,
            "n_hypotheses": len(item.gold),
            "n_answered": sum(v is not None for v in prediction.values()),
            "n_correct_hypotheses": n_correct,
            "score": round(n_correct / len(item.gold), 4),
            "correct": n_correct == len(item.gold),   # all 17 right
        }
