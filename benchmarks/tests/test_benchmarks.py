"""Offline tests (no paid API call):  .venv/bin/python -m pytest benchmarks/tests -q"""

import json
from pathlib import Path

import httpx
import openai
import pytest

import llm.vanila as llm_vanila
from benchmarks.contract_nli_bench import ContractNLI
from benchmarks.maud_bench import Maud
from benchmarks.metrics import build_report
from benchmarks.runner import Item, run_experiment
from lexglue.pipeline import read_jsonl

GOLD = {"nda-1": "Entailment", "nda-2": "NotMentioned", "nda-3": "Contradiction"}


def test_contract_nli_score():
    ds = ContractNLI()
    item = Item(0, "test-1", "doc", "spec", GOLD)
    r = ds.score(item, '```json\n{"nda-1": {"choice": "Entailment"}, "nda-2": {"choice": "Not Mentioned"},'
                       ' "nda-3": {"choice": "Entailment"}}\n```')
    assert (r["n_correct_hypotheses"], r["parse_status"], r["correct"]) == (2, "ok", False)
    r = ds.score(item, '{"nda-1": "Entailment"}')                     # missing hypotheses are wrong
    assert (r["n_correct_hypotheses"], r["n_answered"], r["parse_status"]) == (1, 1, "partial")
    assert ds.score(item, "no idea")["parse_status"] == "unparseable"
    r = ds.score(item, '{"-nda-1": {"choice": "Entailment"}, "-nda-2": {"choice": "NotMentioned"}, '
                       '" nda-3": {"choice": "Contradiction"}}')                # dash copied from the prompt
    assert (r["n_correct_hypotheses"], r["parse_status"]) == (3, "normalized_keys")


def test_maud_score_uses_maud_eval_parser():
    ds = Maud()
    item = Item(0, "contract_1_0", "doc", "spec", ["A", "C"])
    assert ds.score(item, "Answer: A, C")["correct"]
    assert ds.score(item, '{"answer": "A, C"}')["correct_strict"]
    r = ds.score(item, "Answer: A")
    assert not r["correct"] and r["jaccard"] == 0.5
    assert ds.score(item, "")["prediction"] is None


def fake_completion(request: httpx.Request) -> httpx.Response:
    ids = json.load(open(ContractNLI.path))["labels"]           # 17 ids, nda-1 .. nda-20 with gaps
    answer = json.dumps({h: {"choice": "Entailment"} for h in ids})
    return httpx.Response(200, json={
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": answer}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110, "cost": 0.001},
    })


def test_contract_nli_vanilla_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(ContractNLI, "results_root", tmp_path)
    monkeypatch.setattr(llm_vanila, "OpenAI", lambda **kw: openai.OpenAI(
        api_key="t", base_url="http://fake/v1", http_client=httpx.Client(transport=httpx.MockTransport(fake_completion))))
    logs_csv = Path(llm_vanila.__file__).resolve().parent.parent / "logs" / "runs.csv"
    before = logs_csv.read_bytes() if logs_csv.exists() else None

    run_experiment(ContractNLI(), "t", [0, 1], ["vanilla"])
    rows = read_jsonl(tmp_path / "t" / "records" / "contract_nli.jsonl")
    assert [r["status"] for r in rows] == ["ok", "ok"]
    assert rows[0]["reported_cost_usd"] == pytest.approx(0.001)
    gold = rows[0]["gold"]
    assert rows[0]["n_correct_hypotheses"] == sum(v == "Entailment" for v in gold.values())
    assert (logs_csv.read_bytes() if logs_csv.exists() else None) == before   # logs/runs.csv untouched

    run_experiment(ContractNLI(), "t", [0, 1], ["vanilla"])                  # resume: nothing re-run
    assert len(read_jsonl(tmp_path / "t" / "records" / "contract_nli.jsonl")) == 2
    tables = build_report(tmp_path / "t", "contract_nli")
    assert tables["quality"].loc[0, "n_hypotheses"] == 34


# ------------------------------------------------
# Final method failures and the per-run watchdog
# ------------------------------------------------

def test_method_failure_classification():
    from benchmarks import runner
    req = httpx.Request("POST", "http://x")
    too_big = openai.BadRequestError("The total text input size exceeds 8 MB", response=httpx.Response(400, request=req), body=None)
    assert runner.is_method_failure(too_big)
    assert runner.is_method_failure(runner.RunTimeout("x"))
    assert runner.is_method_failure(type("TokenLimitExceededError", (Exception,), {})())
    assert not runner.is_method_failure(openai.APIConnectionError(request=req))


def test_watchdog_turns_a_hung_run_into_a_final_failure(tmp_path, monkeypatch):
    import time
    from benchmarks import runner

    class Hung(ContractNLI):
        results_root = tmp_path

        def call(self, item, method, **kw):
            while True:                 # stands in for REPL code stuck in a regex
                time.sleep(0.1)

    monkeypatch.setattr(runner, "WATCHDOG", runner.Watchdog(timeout_s=1))
    run_experiment(Hung(), "t", [0], ["vanilla"])
    row = read_jsonl(tmp_path / "t" / "records" / "contract_nli.jsonl")[0]
    assert row["status"] == "failed" and row["error"].startswith("RunTimeout") and row["correct"] is False
    run_experiment(Hung(), "t", [0], ["vanilla"])              # final: not retried
    assert len(read_jsonl(tmp_path / "t" / "records" / "contract_nli.jsonl")) == 1
