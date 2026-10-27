"""Per-API-call ledger, cost estimation and run metadata for LexGLUE runs.

CallLedger wraps openai's Completions.create / AsyncCompletions.create, so it
sees every request of this process: the vanilla call, every RLM root iteration
and every llm_query / llm_query_batched sub-call, with the exact usage the
provider returned (incl. reasoning and cached tokens, OpenRouter's billed cost,
the provider that served it) and the latency measured around the request.
Root iterations run in the main thread, sub-calls in the LMHandler's server
threads, which is how `role` is assigned.
Records are processed one at a time, so begin()/end() delimit one record.
"""

import functools
import hashlib
import json
import logging
import platform
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

import requests
from openai.resources.chat.completions import AsyncCompletions, Completions

from llm.rlm_usage_patch import cached_prompt_tokens

OPENROUTER_API = "https://openrouter.ai/api/v1"
REPO_ROOT = Path(__file__).resolve().parent.parent

# Code whose exact state is fingerprinted in every session record
TRACKED_SOURCES = [
    "llm/config.py", "llm/rlm.py", "llm/vanila.py", "llm/rlm_usage_patch.py", "llm/empty_response_retry.py",
    "prompts/prompts.py", "utils/run_logger.py",
    "lexglue/config.py", "lexglue/data.py", "lexglue/prompts.py", "lexglue/parsing.py",
    "lexglue/tracking.py", "lexglue/metrics.py", "lexglue/pipeline.py", "lexglue/parallel.py",
    "lexglue/run_lexglue.py",
    "lexglue/data/eurlex_label_names.json",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _field(obj, name):
    """Attribute, or provider-specific extra field of a pydantic response object."""
    if obj is None:
        return None
    value = getattr(obj, name, None)
    if value is None:
        value = (getattr(obj, "model_extra", None) or {}).get(name)
    return value


def _reported_cost(usage) -> tuple[float | None, float | None]:
    """(billed cost, upstream inference cost) as reported by OpenRouter; None elsewhere."""
    cost = _field(usage, "cost")
    details = _field(usage, "cost_details") or {}
    if not isinstance(details, dict):
        details = getattr(details, "__dict__", {}) or {}
    upstream = details.get("upstream_inference_cost")
    return (None if cost is None else float(cost)), (None if upstream is None else float(upstream))


class CallLedger:
    def __init__(self, store_text: bool = True):
        self.store_text = store_text  # keep full answer + reasoning text of every call
        self._lock = threading.Lock()
        self._calls: list[dict] = []
        self._retries: list[dict] = []
        self.last_retries: list[dict] = []   # SDK retries of the record that just ended
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        ledger = self
        sync_create = Completions.create
        async_create = AsyncCompletions.create

        @functools.wraps(sync_create)
        def create(self_, *args, **kwargs):
            started, t0 = utc_now(), time.perf_counter()
            try:
                response = sync_create(self_, *args, **kwargs)
            except Exception as e:
                ledger._record(kwargs, None, started, time.perf_counter() - t0, e)
                raise
            ledger._record(kwargs, response, started, time.perf_counter() - t0, None)
            return response

        @functools.wraps(async_create)
        async def acreate(self_, *args, **kwargs):
            started, t0 = utc_now(), time.perf_counter()
            try:
                response = await async_create(self_, *args, **kwargs)
            except Exception as e:
                ledger._record(kwargs, None, started, time.perf_counter() - t0, e)
                raise
            ledger._record(kwargs, response, started, time.perf_counter() - t0, None)
            return response

        Completions.create = create
        AsyncCompletions.create = acreate

        class RetryHandler(logging.Handler):
            # openai re-sends a request after a timeout / 429 / 5xx and only logs
            # "Retrying request to ..." at INFO; the failed attempt may still be billed
            def emit(self, record):
                if record.getMessage().startswith("Retrying request"):
                    with ledger._lock:
                        ledger._retries.append({"at": utc_now(), "message": record.getMessage()})

        sdk_log = logging.getLogger("openai._base_client")
        sdk_log.setLevel(logging.INFO)
        sdk_log.addHandler(RetryHandler())
        self._installed = True

    def begin(self) -> None:
        with self._lock:
            self._calls = []
            self._retries = []

    def end(self) -> list[dict]:
        with self._lock:
            calls, self._calls = self._calls, []
            self.last_retries, self._retries = self._retries, []
        return calls

    def _record(self, kwargs: dict, response, started: str, latency: float, error: Exception | None) -> None:
        messages = kwargs.get("messages") or []
        messages_json = json.dumps(messages, ensure_ascii=False, default=str)
        call = {
            "started_at": started,
            "latency_s": round(latency, 4),
            "role": "root" if threading.current_thread() is threading.main_thread() else "sub",
            "request_model": kwargs.get("model"),
            "request_temperature": kwargs.get("temperature"),
            "request_reasoning_effort": kwargs.get("reasoning_effort"),
            "request_max_tokens": kwargs.get("max_completion_tokens") or kwargs.get("max_tokens"),
            "request_extra_body": kwargs.get("extra_body"),
            "n_messages": len(messages),
            "messages_chars": len(messages_json),
            "messages_sha256": sha256_text(messages_json),
            "error": None,
        }
        if error is not None:
            call["error"] = f"{type(error).__name__}: {error}"
            call["error_status_code"] = getattr(error, "status_code", None)
        else:
            usage = getattr(response, "usage", None)
            choice = response.choices[0] if getattr(response, "choices", None) else None
            message = getattr(choice, "message", None)
            content = (getattr(message, "content", None) or "") if message else ""
            reasoning = (_field(message, "reasoning_content") or _field(message, "reasoning") or "") if message else ""
            details = getattr(usage, "completion_tokens_details", None)
            cost, upstream_cost = _reported_cost(usage)
            call.update({
                "response_id": getattr(response, "id", None),
                "response_model": getattr(response, "model", None),
                "provider": _field(response, "provider"),
                "system_fingerprint": getattr(response, "system_fingerprint", None),
                "finish_reason": getattr(choice, "finish_reason", None),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "cached_prompt_tokens": cached_prompt_tokens(usage),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "reasoning_tokens": getattr(details, "reasoning_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
                "reported_cost_usd": cost,
                "upstream_cost_usd": upstream_cost,
                "usage_raw": usage.model_dump() if usage is not None else None,
                "content_chars": len(content),
                "reasoning_chars": len(reasoning),
            })
            if self.store_text:
                call["content"] = content
                call["reasoning"] = reasoning
            if not content and message is not None:
                # empty answer: keep the whole message (tool_calls, refusal, provider fields)
                call["message_raw"] = message.model_dump()
        with self._lock:
            self._calls.append(call)


# ------------------------------------------------
# Pricing / OpenRouter account
# ------------------------------------------------

def openrouter_endpoint(model: str, provider_tag: str) -> dict | None:
    """Public (no key) description of one OpenRouter endpoint, with per-token prices."""
    r = requests.get(f"{OPENROUTER_API}/models/{model}/endpoints", timeout=30)
    r.raise_for_status()
    for e in r.json().get("data", {}).get("endpoints", []):
        if e.get("tag") == provider_tag:
            p = e.get("pricing", {})
            return {
                "model": model,
                "provider_name": e.get("provider_name"),
                "tag": e.get("tag"),
                "quantization": e.get("quantization"),
                "context_length": e.get("context_length"),
                "max_completion_tokens": e.get("max_completion_tokens"),
                "supported_parameters": e.get("supported_parameters"),
                "status": e.get("status"),
                "pricing_per_token": {
                    "prompt": float(p.get("prompt") or 0),
                    "completion": float(p.get("completion") or 0),
                    "input_cache_read": float(p["input_cache_read"]) if p.get("input_cache_read") else None,
                },
                "fetched_at": utc_now(),
            }
    return None


def openrouter_key_status(api_key: str) -> dict | None:
    """Credit usage of the key (free endpoint). Label and account identifiers are dropped."""
    try:
        r = requests.get(f"{OPENROUTER_API}/key", headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        return {"error": f"{type(e).__name__}: {e}"}
    data = r.json().get("data", {})
    for field in ("label", "creator_user_id", "workspace_id", "organization_id"):
        data.pop(field, None)
    data["fetched_at"] = utc_now()
    return data


def estimate_cost(call: dict, pricing: dict | None) -> float | None:
    """Cost from token counts and per-token prices (cache hits at the cache-read price)."""
    if not pricing or call.get("prompt_tokens") is None:
        return None
    cached = call.get("cached_prompt_tokens") or 0
    cache_price = pricing["input_cache_read"] if pricing.get("input_cache_read") is not None else pricing["prompt"]
    return (
        (call["prompt_tokens"] - cached) * pricing["prompt"]
        + cached * cache_price
        + (call.get("completion_tokens") or 0) * pricing["completion"]
    )


# ------------------------------------------------
# Run metadata
# ------------------------------------------------

def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def code_state() -> dict:
    files = {}
    for rel in TRACKED_SOURCES:
        path = REPO_ROOT / rel
        files[rel] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty_files": [l for l in _git("status", "--porcelain").splitlines() if l],
        "git_diff_sha256": sha256_text(_git("diff", "HEAD")),
        "source_sha256": files,
    }


def environment_info() -> dict:
    versions = {}
    for pkg in ("openai", "rlms", "datasets", "scikit-learn", "scipy", "numpy", "pandas", "requests"):
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            versions[pkg] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "argv": sys.argv,
        "packages": versions,
    }
