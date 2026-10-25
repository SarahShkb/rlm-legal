"""Fix RLM usage accounting in the `rlms` package (local editable copy and PyPI 0.1.3).

Bug: passing `sub_sampling_args` without `other_backends` makes rlms create a
second client for sub-calls with the SAME model name as the root client.
`LMHandler.get_usage_summary()` merges per-client summaries with `dict.update`
keyed by model name, so the sub-call client's stats overwrite the root model's.
Whenever a run made any llm_query() call, all root iterations were dropped from
the logged tokens / calls / cost.

This module (imported for its side effects from llm/rlm.py):
  1. sums usage over every distinct client instead of overwriting by model name;
  2. also tracks cached prompt tokens per client, exposed on the returned
     UsageSummary as `cached_input_tokens` (DeepSeek bills cache hits cheaper);
  3. counts sub-calls (llm_query / llm_query_batched requests from the REPL,
     one per prompt) and exposes `sub_calls` and `root_calls` on the summary.
"""

from collections import defaultdict

from rlm.clients.openai import OpenAIClient
from rlm.core.lm_handler import LMHandler, LMRequestHandler
from rlm.core.types import ModelUsageSummary, UsageSummary


def cached_prompt_tokens(usage) -> int:
    """Cached prompt tokens from an OpenAI-compatible `usage` object (0 if not reported)."""
    if usage is None:
        return 0
    # DeepSeek
    hit = getattr(usage, "prompt_cache_hit_tokens", None)
    if hit is None and getattr(usage, "model_extra", None):
        hit = usage.model_extra.get("prompt_cache_hit_tokens")
    if hit is not None:
        return int(hit)
    # OpenAI / vLLM
    details = getattr(usage, "prompt_tokens_details", None)
    return int(getattr(details, "cached_tokens", 0) or 0)


_original_track_cost = OpenAIClient._track_cost


def _track_cost_with_cache(self, response, model: str):
    _original_track_cost(self, response, model)
    if not hasattr(self, "model_cached_tokens"):
        self.model_cached_tokens = defaultdict(int)
    self.model_cached_tokens[model] += cached_prompt_tokens(getattr(response, "usage", None))


def _summed_usage_summary(self) -> UsageSummary:
    clients = {
        id(c): c
        for c in [self.default_client, self.other_backend_client, *self.clients.values()]
        if c is not None
    }
    merged: dict[str, ModelUsageSummary] = {}
    cached = 0
    for client in clients.values():
        cached += sum(getattr(client, "model_cached_tokens", {}).values())
        for model, s in client.get_usage_summary().model_usage_summaries.items():
            prev = merged.get(model)
            if prev is None:
                merged[model] = s
                continue
            costs = [c for c in (prev.total_cost, s.total_cost) if c is not None]
            merged[model] = ModelUsageSummary(
                total_calls=prev.total_calls + s.total_calls,
                total_input_tokens=prev.total_input_tokens + s.total_input_tokens,
                total_output_tokens=prev.total_output_tokens + s.total_output_tokens,
                total_cost=sum(costs) if costs else None,
            )
    summary = UsageSummary(model_usage_summaries=merged)
    summary.cached_input_tokens = cached
    # Every request reaching the socket server comes from the REPL (a sub-call);
    # root iterations call the client directly via LMHandler.completion().
    total_calls = sum(s.total_calls for s in merged.values())
    summary.sub_calls = getattr(self, "sub_call_count", 0)
    summary.root_calls = total_calls - summary.sub_calls
    return summary


_original_handle_single = LMRequestHandler._handle_single
_original_handle_batched = LMRequestHandler._handle_batched


def _count_single(self, request, handler):
    handler.sub_call_count = getattr(handler, "sub_call_count", 0) + 1
    return _original_handle_single(self, request, handler)


def _count_batched(self, request, handler):
    handler.sub_call_count = getattr(handler, "sub_call_count", 0) + len(request.prompts or [])
    return _original_handle_batched(self, request, handler)


OpenAIClient._track_cost = _track_cost_with_cache
LMHandler.get_usage_summary = _summed_usage_summary
LMRequestHandler._handle_single = _count_single
LMRequestHandler._handle_batched = _count_batched
