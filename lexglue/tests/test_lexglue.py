"""Offline tests (no paid API call). Run from the repo root:
    .venv/bin/python -m pytest lexglue/tests -q
"""

import json
import threading
from types import SimpleNamespace

import httpx
import numpy as np
import openai
import pytest
from sklearn.metrics import f1_score

import llm.rlm as llm_rlm
import llm.vanila as llm_vanila
from lexglue import pipeline
from lexglue.config import ECTHR_LABEL_NAMES, TASKS
from lexglue.data import label_names, load_records, load_split
from lexglue.metrics import build_report, f1_scores
from lexglue.parsing import is_correct, parse_prediction
from lexglue.prompts import build_task_spec
from lexglue.tracking import CallLedger, estimate_cost

SCOTUS = ['Criminal Procedure', 'Civil Rights', 'First Amendment']


# ------------------------------------------------
# Parser
# ------------------------------------------------

@pytest.mark.parametrize("text,expected,status", [
    ('{"label": "Civil Rights"}', "Civil Rights", "ok"),
    ('```json\n{"label": "Civil Rights"}\n```', "Civil Rights", "ok"),
    ('Reasoning... final: {"label": "civil rights."}', "Civil Rights", "normalized"),
    ('"{\\"label\\": \\"Civil Rights\\"}"', "Civil Rights", "ok"),          # JSON string holding JSON
    ("{'label': 'Civil Rights'}", "Civil Rights", "ok"),                     # python dict repr
    ("Civil Rights", "Civil Rights", "no_json"),
    ('{"label": "Taxes"}', None, "invalid_label"),
    ("I cannot decide", None, "unparseable"),
    ("", None, "unparseable"),
    (None, None, "unparseable"),
])
def test_parse_multi_class(text, expected, status):
    r = parse_prediction("scotus", SCOTUS, 0, text)
    assert (r.prediction, r.status) == (expected, status)


def test_parse_multi_label():
    labels = ECTHR_LABEL_NAMES
    r = parse_prediction("ecthr_a", labels, 0, '{"labels": ["Article 3 - Prohibition of torture", "Article 6"]}')
    assert r.prediction == ["Article 3 - Prohibition of torture", "Article 6 - Right to a fair trial"]
    assert r.status == "normalized"
    assert parse_prediction("ecthr_a", labels, 0, '{"labels": []}').prediction == []
    assert parse_prediction("ecthr_a", labels, 0, '{"labels": null}').prediction == []
    r = parse_prediction("ecthr_a", labels, 0, '{"labels": ["Article 3", "Article 99"]}')
    assert (r.prediction, r.status, r.invalid) == (["Article 3 - Prohibition of torture"], "partial_invalid", ["Article 99"])
    r = parse_prediction("ecthr_a", labels, 0, '{"labels": ["Article 99"]}')
    assert (r.prediction, r.status) == (None, "invalid_label")
    assert is_correct("ecthr_a", [], [])
    assert not is_correct("ecthr_a", None, [])


def test_parse_case_hold():
    assert parse_prediction("case_hold", None, 5, '{"answer": 3}').prediction == 3
    assert parse_prediction("case_hold", None, 5, '{"answer": "3"}').prediction == 3
    assert parse_prediction("case_hold", None, 5, "3").status == "no_json"
    assert parse_prediction("case_hold", None, 5, '{"answer": 7}').status == "invalid_label"
    assert parse_prediction("case_hold", None, 5, '{"answer": 2.5}').status == "invalid_label"


@pytest.mark.parametrize("task", list(TASKS))
def test_gold_round_trip_and_spec(task):
    """For the real data: gold answer in the requested JSON parses back to gold, spec survives format()."""
    ds = load_split(task, "validation")
    labels = None if task == "case_hold" else label_names(task, ds.features)
    key = {"multi_class": "label", "multi_label": "labels", "multiple_choice": "answer"}[TASKS[task]["task_type"]]
    for rec in load_records(task, "validation", list(range(100))):
        r = parse_prediction(task, labels, len(rec.options), json.dumps({key: rec.gold}))
        assert r.status == "ok" and is_correct(task, r.prediction, rec.gold), rec.row_id
        spec = build_task_spec(rec, labels)
        assert spec.replace("{", "{{").replace("}", "}}").format(custom_tools_section="") == spec


# ------------------------------------------------
# run_rlm_customized: brace escaping + on_result (RLM stubbed, no API call)
# ------------------------------------------------

def test_rlm_customized_escapes_task_spec(monkeypatch):
    captured = {}

    class FakeRLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def completion(self, prompt, root_prompt):
            captured["root_prompt"] = root_prompt
            return SimpleNamespace(response='{"label": "x"}', execution_time=0.1, metadata=None, usage_summary=None)

    monkeypatch.setattr(llm_rlm, "RLM", FakeRLM)
    monkeypatch.setattr(llm_rlm, "log_run", lambda *a, **k: None)
    spec = 'Return ONLY: {"label": "<candidate label>"}'
    seen = []
    llm_rlm.run_rlm_customized("d", "r", "doc", spec, escape_task_spec_braces=True,
                               on_result=lambda result, rlm: seen.append(result))
    # exactly what rlm.utils.prompts.build_rlm_system_prompt does with it
    system = captured["custom_system_prompt"].format(custom_tools_section="")
    assert spec in system
    assert spec in captured["root_prompt"]
    assert len(seen) == 1


# ------------------------------------------------
# Metrics
# ------------------------------------------------

def test_f1_matches_sklearn():
    rng = np.random.default_rng(0)
    Y = rng.random((60, 8)) < 0.2
    P = rng.random((60, 8)) < 0.25
    micro, macro = f1_scores(Y, P)
    present = np.where(Y.any(0) | P.any(0))[0]
    assert micro == pytest.approx(f1_score(Y, P, average="micro", zero_division=0))
    assert macro == pytest.approx(f1_score(Y, P, average="macro", labels=present, zero_division=0))


# ------------------------------------------------
# Call ledger + full pipeline against a fake OpenAI-compatible API
# ------------------------------------------------

def fake_completion(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    answer = '{"label": "Amendments"}'
    return httpx.Response(200, json={
        "id": "fake-1", "object": "chat.completion", "created": 0, "model": body["model"],
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": answer, "reasoning_content": "thinking..."}}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200,
                  "prompt_cache_hit_tokens": 400, "prompt_cache_miss_tokens": 600,
                  "completion_tokens_details": {"reasoning_tokens": 150}},
    })


def fake_client(**kwargs):
    return openai.OpenAI(api_key="test", base_url="http://fake/v1",
                         http_client=httpx.Client(transport=httpx.MockTransport(fake_completion)))


def test_ledger_roles_and_usage():
    ledger = CallLedger()
    ledger.install()
    ledger.begin()
    client = fake_client()
    client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
    t = threading.Thread(target=lambda: client.chat.completions.create(model="m", messages=[]))
    t.start(); t.join()
    calls = ledger.end()
    assert [c["role"] for c in calls] == ["root", "sub"]
    c = calls[0]
    assert (c["prompt_tokens"], c["cached_prompt_tokens"], c["completion_tokens"], c["reasoning_tokens"]) == (1000, 400, 200, 150)
    assert c["reasoning"] == "thinking..."
    price = {"prompt": 1e-6, "completion": 2e-6, "input_cache_read": 0.1e-6}
    assert estimate_cost(c, price) == pytest.approx(600e-6 + 400 * 0.1e-6 + 200 * 2e-6)


def test_pipeline_end_to_end_offline(tmp_path, monkeypatch, capsys):
    from pathlib import Path
    from lexglue.parallel import plan_shards, spent_since

    runs_csv = Path(pipeline.__file__).resolve().parent.parent / "logs" / "runs.csv"
    before = runs_csv.read_bytes() if runs_csv.exists() else None
    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(llm_vanila, "OpenAI", fake_client)
    run = dict(experiment="t", tasks=["ledgar"], methods=["vanilla"], split="validation", indices=[0, 1],
               price_per_mtok=(1.0, 2.0, 0.1))
    pipeline.run_experiment(**run)

    records = pipeline.read_jsonl(tmp_path / "t" / "records" / "ledgar.jsonl")
    calls = pipeline.read_jsonl(tmp_path / "t" / "calls" / "ledgar.jsonl")
    assert [r["status"] for r in records] == ["ok", "ok"]
    assert len(calls) == 2
    r0 = records[0]
    assert r0["prediction"] == "Amendments" and r0["parse_status"] == "ok"
    assert r0["correct"] == (r0["gold"] == "Amendments")
    assert (r0["prompt_tokens"], r0["cached_prompt_tokens"], r0["reasoning_tokens"]) == (1000, 400, 150)
    assert r0["estimated_cost_usd"] == pytest.approx(600e-6 + 400 * 0.1e-6 + 200 * 2e-6)
    assert calls[0]["request_temperature"] == 0          # vanilla now sends the same temperature as the RLM

    # resume: nothing is re-run
    pipeline.run_experiment(**run)
    assert len(pipeline.read_jsonl(tmp_path / "t" / "records" / "ledgar.jsonl")) == 2
    sessions = pipeline.read_jsonl(tmp_path / "t" / "sessions.jsonl")
    assert sessions[-1]["records_skipped"] == 2

    # a changed configuration is refused under the same experiment name
    with pytest.raises(SystemExit):
        pipeline.run_experiment(**{**run, "max_iterations": 3})

    # log_run wrote its summary row inside the experiment folder, not into logs/runs.csv
    assert list((tmp_path / "t" / "runs_csv").glob("ledgar__*.csv"))
    assert (runs_csv.read_bytes() if runs_csv.exists() else None) == before

    # parallel launcher helpers: finished records are not re-planned, costs are summed
    assert plan_shards("t", ["ledgar", "scotus"], [0, 1, 2], ["vanilla"], 1, 2) == [
        ("scotus", [0, 1]), ("scotus", [2]), ("ledgar", [2])]
    spent, n = spent_since("t", ["ledgar"], "2000-01-01")
    assert n == 2 and spent == pytest.approx(2 * (600e-6 + 400 * 0.1e-6 + 200 * 2e-6))

    tables = build_report(tmp_path / "t", "validation")
    assert tables["quality"].loc[0, "n"] == 2
    assert (tmp_path / "t" / "report" / "all_records.csv").exists()


# ------------------------------------------------
# Empty provider answers are re-sent (llm/empty_response_retry.py, llm/vanila.py)
# ------------------------------------------------

def test_rlm_client_retries_empty_answer(monkeypatch):
    import llm.empty_response_retry as retry
    answers = iter([None, "", "```repl\nprint(1)\n```"])
    monkeypatch.setattr(retry, "_original_completion", lambda self, prompt, model=None: next(answers))
    assert retry._completion_with_retry(None, "p") == "```repl\nprint(1)\n```"
    always_empty = iter([None] * 3)
    monkeypatch.setattr(retry, "_original_completion", lambda self, prompt, model=None: next(always_empty))
    assert retry._completion_with_retry(None, "p") == ""      # no crash, no 4th attempt


def test_vanilla_retries_empty_answer(monkeypatch, tmp_path):
    n = {"calls": 0}

    def flaky(request):
        n["calls"] += 1
        response = fake_completion(request)
        if n["calls"] == 1:
            body = json.loads(response.content)
            body["choices"][0]["message"]["content"] = None
            return httpx.Response(200, json=body)
        return response

    monkeypatch.setattr(llm_vanila, "OpenAI", lambda **kw: openai.OpenAI(
        api_key="t", base_url="http://fake/v1", http_client=httpx.Client(transport=httpx.MockTransport(flaky))))
    out = llm_vanila.run_vanilla("d", "r", "doc", "spec", runs_csv=str(tmp_path / "runs.csv"))
    assert out == '{"label": "Amendments"}' and n["calls"] == 2
