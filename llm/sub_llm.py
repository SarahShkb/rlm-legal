"""Small sub-LLMs for the RLM's llm_query() calls (depth 1); the root stays llm.config's model.

Each entry is an OpenAI-compatible endpoint: a vLLM server from
modal_subllm/serve.py, or OpenRouter. Every sub-LLM is used the same way,
whatever its server accepts (8192-token window on Modal, 131K on OpenRouter):
greedy, at most SUB_LLM_MAX_TOKENS answer tokens, and prompts longer than
SUB_LLM_MAX_PROMPT_CHARS are refused before they are sent. The refusal reaches
the REPL as an "Error: ..." string, like any failed sub-call, so the root can
re-chunk.

Imported for its side effects too: rlm's OpenAIClient is wrapped so that, for
clients built from backend_kwargs() here (flagged sub_llm=True, so a sub-LLM
that is the root model itself is told apart from the root), the prompt size is
checked and at most SUB_LLM_MAX_CONCURRENT sub-calls per process are in flight
(the benchmark workers are processes, so one GPU sees at most
workers x SUB_LLM_MAX_CONCURRENT).

The root is told these limits by text appended to its system prompt
(prompts.SUB_LLM_LIMITS); rlm's own prompt is left as it is.

  python -m llm.sub_llm check saul     # credentials are enforced, then one tiny completion
"""

import asyncio
import contextvars
import os
import sys
import threading
import time
from contextlib import contextmanager

import rlm.core.rlm as rlm_core
from rlm.clients.openai import OpenAIClient

from llm.config import TEMPERATURE

# 20,000 chars of MAUD contract text is at most 6,261 tokens with the Mistral/Saul
# tokenizer (5,409 with Llama 3), measured on 40 contracts: + 1,024 answer tokens < 8,192
SUB_LLM_MAX_PROMPT_CHARS = 20_000
SUB_LLM_CHUNK_CHARS = 16_000       # excerpt size suggested to the root: room for its instructions
SUB_LLM_MAX_TOKENS = 1024
SUB_LLM_MAX_CONCURRENT = 4
# REPL output the root sees per block when it must delegate (rlm's default: 20,000):
# enough for search results and a summary of sub-LLM answers, too little to read the contract
ROOT_OUTPUT_CHARS = 1_000

MODAL_LABEL = "rlm-subllm-{key}"   # web_server labels in modal_subllm/serve.py

# name -> model (the served model name on Modal) and where it is served
SUB_LLMS = {
    "saul": {"host": "modal", "model": "Equall/Saul-7B-Instruct-v1"},
    "mistral": {"host": "modal", "model": "mistralai/Mistral-7B-Instruct-v0.1"},
    "lawma": {"host": "modal", "model": "ricdomolm/lawma-8b"},
    # RLM-trained Qwen3-30B-A3B (LoRA, served under the adapter's name); also usable as root
    # (LLM_PROVIDER=modal, MODAL_ROOT=rlm30b)
    "rlm30b": {"host": "modal", "model": "rlm-qwen3-30b-a3b"},
    # Lawma's control, through OpenRouter (Meta refused the HF download): the only bf16 endpoint, pinned
    "llama": {"host": "openrouter", "model": "meta-llama/llama-3.1-8b-instruct", "provider": "coreweave/bf16"},
    # baseline for a Qwen3-30B-A3B root: the same model as sub-LLM (no reasoning mode)
    "qwen": {"host": "openrouter", "model": "qwen/qwen3-30b-a3b-instruct-2507", "provider": "nebius/fp8"},
    "qwen235": {"host": "openrouter", "model": "qwen/qwen3-235b-a22b-2507", "provider": "parasail/fp8"},
    # anchor: the root model and endpoint itself (LLM_PROVIDER=openrouter), under the same
    # limits and delegation rule as the small models; no reasoning, as they have none
    "deepseek": {"host": "openrouter", "model": os.environ.get("OPENROUTER_MODEL", ""),
                 "provider": os.environ.get("OPENROUTER_PROVIDER", ""), "reasoning": False},
}


def base_url(name: str) -> str:
    cfg = SUB_LLMS[name]
    if cfg["host"] == "openrouter":
        return "https://openrouter.ai/api/v1"
    override = os.environ.get(f"SUBLLM_{name.upper()}_URL")
    if override:
        return override.rstrip("/")
    workspace = os.environ.get("MODAL_WORKSPACE")
    if not workspace:
        raise ValueError(f"sub-LLM {name!r}: set MODAL_WORKSPACE (or SUBLLM_{name.upper()}_URL) in .env")
    return f"https://{workspace}--{MODAL_LABEL.format(key=name)}.modal.run/v1"


def backend_kwargs(name: str) -> dict:
    """kwargs for rlm's OpenAIClient (RLM other_backend_kwargs)."""
    cfg = SUB_LLMS[name]
    if cfg["host"] == "openrouter":
        if not cfg["model"] or not cfg["provider"]:
            raise ValueError(f"sub-LLM {name!r}: set OPENROUTER_MODEL and OPENROUTER_PROVIDER in .env")
        return {"model_name": cfg["model"], "base_url": base_url(name), "api_key": os.environ["OPENROUTER_API_KEY"],
                "sub_llm": True}
    missing = [v for v in ("SUBLLM_API_KEY", "MODAL_PROXY_TOKEN_ID", "MODAL_PROXY_TOKEN_SECRET") if not os.environ.get(v)]
    if missing:
        raise ValueError(f"sub-LLM {name!r}: set {', '.join(missing)} in .env")
    return {
        "model_name": cfg["model"],
        "base_url": base_url(name),
        "api_key": os.environ["SUBLLM_API_KEY"],                      # vllm serve --api-key
        "default_headers": {"Modal-Key": os.environ["MODAL_PROXY_TOKEN_ID"],
                            "Modal-Secret": os.environ["MODAL_PROXY_TOKEN_SECRET"]},
        "sub_llm": True,
    }


def sampling_args(name: str) -> dict:
    """RLM sub_sampling_args: the same for every sub-LLM (no reasoning, no provider extras on Modal)."""
    args = {"temperature": TEMPERATURE, "max_tokens": SUB_LLM_MAX_TOKENS}
    cfg = SUB_LLMS[name]
    extra = {}
    if cfg.get("provider"):
        extra["provider"] = {"order": [cfg["provider"]], "allow_fallbacks": False, "require_parameters": True}
    if cfg.get("reasoning") is False:
        extra["reasoning"] = {"enabled": False}
    if extra:
        args["extra_body"] = extra
    return args


def public_settings(name: str) -> dict:
    """What the manifest records about a sub-LLM (no credentials)."""
    return {**SUB_LLMS[name], "base_url": base_url(name), "sampling": sampling_args(name),
            "max_prompt_chars": SUB_LLM_MAX_PROMPT_CHARS, "max_concurrent_per_process": SUB_LLM_MAX_CONCURRENT}


# ------------------------------------------------
# Prompt-size guard and concurrency cap on rlm's OpenAIClient
# ------------------------------------------------

_slots = threading.BoundedSemaphore(SUB_LLM_MAX_CONCURRENT)
_inner_init = OpenAIClient.__init__
_inner_completion = OpenAIClient.completion
_inner_acompletion = OpenAIClient.acompletion


def _check_prompt_size(prompt) -> None:
    chars = len(prompt) if isinstance(prompt, str) else sum(len(str(m.get("content") or "")) for m in prompt)
    if chars > SUB_LLM_MAX_PROMPT_CHARS:
        raise ValueError(f"sub-LLM prompt has {chars:,} characters, the limit is {SUB_LLM_MAX_PROMPT_CHARS:,}; "
                         "split the text into smaller excerpts")


def _init(self, *args, sub_llm: bool = False, **kwargs):
    _inner_init(self, *args, **kwargs)
    self.sub_llm = sub_llm


def _completion(self, prompt, model=None):
    if not getattr(self, "sub_llm", False):
        return _inner_completion(self, prompt, model)
    _check_prompt_size(prompt)
    with _slots:
        return _inner_completion(self, prompt, model)


async def _acompletion(self, prompt, model=None):
    if not getattr(self, "sub_llm", False):
        return await _inner_acompletion(self, prompt, model)
    _check_prompt_size(prompt)
    # polled, not a blocking acquire in a thread: a cancelled task never holds a slot
    while not _slots.acquire(blocking=False):
        await asyncio.sleep(0.05)
    try:
        return await _inner_acompletion(self, prompt, model)
    finally:
        _slots.release()


OpenAIClient.__init__ = _init
OpenAIClient.completion = _completion
OpenAIClient.acompletion = _acompletion


# ------------------------------------------------
# Root REPL output cap (rlm truncates each output block at 20,000 chars by default)
# ------------------------------------------------

_output_cap = contextvars.ContextVar("root_output_cap", default=None)
_inner_format_iteration = rlm_core.format_iteration


def _format_iteration(iteration, max_character_length=20000):
    cap = _output_cap.get()
    return _inner_format_iteration(iteration, max_character_length if cap is None else cap)


rlm_core.format_iteration = _format_iteration


@contextmanager
def root_output_cap(chars: int | None):
    """Within the block, RLM iterations show the root at most `chars` of each REPL
    output block (None: rlm's default)."""
    token = _output_cap.set(chars)
    try:
        yield
    finally:
        _output_cap.reset(token)


# ------------------------------------------------
# Endpoint checks
# ------------------------------------------------

def wait_ready(name: str, timeout_s: float = 20 * 60) -> None:
    """Block until the endpoint answers (a cold Modal start loads the weights first)."""
    import openai

    if SUB_LLMS[name]["host"] != "modal":
        return
    kw = backend_kwargs(name)
    client = openai.OpenAI(api_key=kw["api_key"], base_url=kw["base_url"],
                           default_headers=kw.get("default_headers"), max_retries=0, timeout=300)
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            client.chat.completions.create(model=kw["model_name"], messages=[{"role": "user", "content": "Hi"}],
                                           max_tokens=1, temperature=0)
            print(f"[sub_llm] {name} is ready", flush=True)
            return
        except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as e:
            if time.monotonic() > deadline:
                raise
            print(f"[sub_llm] waiting for {name} ({type(e).__name__})", flush=True)
            time.sleep(15)


def check(name: str) -> None:
    import openai
    import requests

    kw = backend_kwargs(name)
    url = f"{kw['base_url']}/chat/completions"
    body = {"model": kw["model_name"], "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1}
    if SUB_LLMS[name]["host"] == "modal":
        r = requests.post(url, json=body, timeout=60)
        print(f"no credentials:        HTTP {r.status_code} (expected 401, rejected by Modal, no container started)")
        r = requests.post(url, json=body, headers=kw["default_headers"], timeout=600)
        print(f"proxy token, no key:   HTTP {r.status_code} (expected 401, rejected by vLLM)")
    wait_ready(name)
    client = openai.OpenAI(api_key=kw["api_key"], base_url=kw["base_url"], default_headers=kw.get("default_headers"))
    args = sampling_args(name)
    resp = client.chat.completions.create(
        model=kw["model_name"], extra_body=args.pop("extra_body", None),
        messages=[{"role": "user", "content": "In one sentence: what is a material adverse effect clause?"}], **args)
    print(f"answer ({resp.model}, {resp.usage.prompt_tokens}+{resp.usage.completion_tokens} tokens): "
          f"{resp.choices[0].message.content!r}")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "check" or sys.argv[2] not in SUB_LLMS:
        sys.exit(f"usage: python -m llm.sub_llm check <{'|'.join(SUB_LLMS)}>")
    check(sys.argv[2])
