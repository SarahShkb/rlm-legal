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


# ------------------------------------------------
# Small sub-LLMs (llm/sub_llm.py, modal_subllm/serve.py)
# ------------------------------------------------

SUB_ENV = {"MODAL_WORKSPACE": "ws", "SUBLLM_API_KEY": "vllm-key",
           "MODAL_PROXY_TOKEN_ID": "wk-id", "MODAL_PROXY_TOKEN_SECRET": "ws-secret"}


def test_sub_llm_registry_matches_modal_server():
    import modal_subllm.serve as serve
    from llm.sub_llm import SUB_LLMS
    for name, cfg in SUB_LLMS.items():
        if cfg["host"] == "modal":
            assert serve.MODELS[name]["repo"] == cfg["model"]
            assert hasattr(serve, name)                      # one web endpoint per model


def test_sub_llm_prompt_guard(monkeypatch):
    import llm.sub_llm as sub_llm
    from rlm.clients.openai import OpenAIClient
    sent = []
    monkeypatch.setattr(sub_llm, "_inner_completion", lambda self, prompt, model=None: sent.append(prompt) or "ok")
    for model in ("Equall/Saul-7B-Instruct-v1", "deepseek/whatever"):
        client = OpenAIClient(api_key="t", model_name=model, base_url="http://fake/v1", sub_llm=True)
        assert client.completion("x" * sub_llm.SUB_LLM_MAX_PROMPT_CHARS) == "ok"
        with pytest.raises(ValueError, match="limit is 20,000"):
            client.completion([{"role": "user", "content": "x" * (sub_llm.SUB_LLM_MAX_PROMPT_CHARS + 1)}])
    assert len(sent) == 2
    root = OpenAIClient(api_key="t", model_name="deepseek/whatever", base_url="http://fake/v1")
    assert root.completion("x" * 10**6) == "ok"              # the root is not limited, even when it is the same model


def test_sub_llm_method_end_to_end(tmp_path, monkeypatch):
    """A real RLM run on mocked HTTP: llm_query goes to the Modal sub-LLM with its own
    credentials and sampling; an oversized prompt never leaves the process."""
    import llm.rlm as llm_rlm
    for k, v in SUB_ENV.items():
        monkeypatch.setenv(k, v)
    root_requests, sub_requests = [], []

    def fake(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["model"] == "Equall/Saul-7B-Instruct-v1":
            sub_requests.append((request, body))
            content = "Answer: B"
        else:
            root_requests.append(body)
            if len(root_requests) == 1:
                content = ('```repl\nr = llm_query("Which choice? A or B")\nbig = llm_query("x" * 30000)\n'
                           'print(r, "|", big[:40])\nprint("y" * 5000)\n```')
            else:
                content = '```repl\nanswer["content"] = "Answer: " + r[-1]\nanswer["ready"] = True\n```'
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}})

    real_sync, real_async = openai.OpenAI, openai.AsyncOpenAI
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: real_sync(
        **kw, http_client=httpx.Client(transport=httpx.MockTransport(fake))))
    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kw: real_async(
        **kw, http_client=httpx.AsyncClient(transport=httpx.MockTransport(fake))))
    monkeypatch.setattr(llm_rlm, "API_KEY", "root-key")

    response = llm_rlm.run_rlm_customized(
        "t", "row", "THE CONTRACT", TASK_SPEC="Question?", max_iterations=3, runs_csv=str(tmp_path / "r.csv"),
        log_dir=str(tmp_path), sub_llm="saul", sub_llm_delegate=True)

    assert response == "Answer: B"
    system = root_requests[0]["messages"][0]["content"]
    metadata = root_requests[0]["messages"][1]["content"]
    # rlm's prompt is sent unchanged, the limits are appended after it
    assert "Sub-LLM context window ≈ 500K chars." in system and "act as an orchestrator, not a solver" in system
    assert "100k tokens" in metadata
    assert system.index("SUB-LLM:") < system.index("DELEGATION:") and "These limits replace" in system
    assert "at most 20,000 characters" in system
    assert "REPL OUTPUT LIMIT" in system and "REPL OUTPUT LIMIT" in metadata   # metadata holds the root prompt
    repl_output = next(m["content"] for m in root_requests[1]["messages"] if m["content"].startswith("REPL output"))
    assert "Error: sub-LLM prompt has 30,000" in repl_output and "... + [" in repl_output
    assert repl_output.count("y") < 1_000
    assert len(sub_requests) == 1                           # the 30,000-char prompt was refused locally
    request, body = sub_requests[0]
    assert str(request.url) == "https://ws--rlm-subllm-saul.modal.run/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer vllm-key"
    assert (request.headers["modal-key"], request.headers["modal-secret"]) == ("wk-id", "ws-secret")
    assert body["temperature"] == 0 and body["max_completion_tokens"] == 1024
    assert "reasoning_effort" not in body and "provider" not in body and "reasoning" not in body


def test_maud_sub_llm_method(monkeypatch):
    import benchmarks.maud_bench as maud_bench
    for k, v in SUB_ENV.items():
        monkeypatch.setenv(k, v)
    seen = {}
    monkeypatch.setattr(maud_bench, "run_rlm_customized", lambda *a, **kw: seen.update(kw) or "Answer: A")
    item = Item(0, "contract_1_0", "doc", "spec", ["A"])
    response, extra = Maud().call(item, "rlm_lawma", experiment="e", max_iterations=8, runs_csv="x", log_dir="y",
                                  on_result=None)
    assert response == "Answer: A" and seen["sub_llm"] == "lawma" and seen["sub_llm_delegate"] is True
    assert "tools" not in seen                               # sub-LLM conditions run without the legal tools
    assert extra["sub_llm_settings"]["model"] == "ricdomolm/lawma-8b"
    assert "vllm-key" not in json.dumps(extra)               # no credentials in the records
    assert set(Maud.default_methods) == {"vanilla", "rlm", "rlm_tool1", "rlm_tool2", "rlm_tool12"}


def test_maud_deepseek_anchor(monkeypatch):
    """rlm_deepseek: the root's model and endpoint, under the same sub-LLM limits, without reasoning."""
    import benchmarks.maud_bench as maud_bench
    import llm.sub_llm as sub_llm
    monkeypatch.setitem(sub_llm.SUB_LLMS, "deepseek", {**sub_llm.SUB_LLMS["deepseek"],
                                                      "model": "deepseek/m", "provider": "p/fp8"})
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    kw = sub_llm.backend_kwargs("deepseek")
    assert (kw["model_name"], kw["base_url"], kw["sub_llm"]) == ("deepseek/m", "https://openrouter.ai/api/v1", True)
    args = sub_llm.sampling_args("deepseek")
    assert args["max_tokens"] == sub_llm.SUB_LLM_MAX_TOKENS and args["extra_body"]["reasoning"] == {"enabled": False}
    assert args["extra_body"]["provider"] == {"order": ["p/fp8"], "allow_fallbacks": False, "require_parameters": True}
    assert maud_bench.SUB_LLM_METHODS["rlm_deepseek"] == "deepseek"


def test_maud_tools_with_sub_llm(monkeypatch):
    """rlm_tool1_llama: tool 1 and its hint, plus the Llama sub-LLM under the delegation config."""
    import benchmarks.maud_bench as maud_bench
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    seen = {}
    monkeypatch.setattr(maud_bench, "run_rlm_customized", lambda *a, **kw: seen.update(kw) or "Answer: A")
    monkeypatch.setattr(maud_bench, "build_tools", lambda text, which, log, **kw: {"toc": object()})
    item = Item(0, "contract_1_0", "doc", "spec", ["A"])
    response, extra = Maud().call(item, "rlm_tool1_llama", experiment="e", max_iterations=12, runs_csv="x",
                                  log_dir="y", on_result=None)
    assert seen["sub_llm"] == "llama" and seen["sub_llm_delegate"] is True and "toc" in seen["tools"]
    assert seen["tool_hint"] and extra["tools"] == ["1"] and extra["sub_llm"] == "llama"
    assert "rlm_tool1_llama" in Maud.methods and "rlm_tool1_llama" not in Maud.default_methods

