"""Re-send a request whose answer came back empty (imported for its side effects).

Some OpenRouter providers of DeepSeek V4.1 Flash (seen on DeepInfra and
InferenceNet, ~5% of RLM root calls) occasionally return `content: null` with
finish_reason "stop": the reasoning is there, but the ~150 answer tokens that
were generated (and billed) are lost when the provider splits reasoning from
content. The RLM library then crashes on the None (re.finditer(None)).

This is a malformed response, not a model answer, so the identical request is
sent again, up to MAX_EMPTY_RETRIES more times. Every attempt goes through the
real client, so its tokens/cost are tracked (RLM usage, lexglue call ledger).
If all attempts are empty, "" is returned (the RLM sees an empty turn instead
of crashing); vanilla handles this in llm/vanila.py.
"""

from rlm.clients.openai import OpenAIClient

MAX_EMPTY_RETRIES = 2

_original_completion = OpenAIClient.completion
_original_acompletion = OpenAIClient.acompletion


def _completion_with_retry(self, prompt, model=None):
    for attempt in range(MAX_EMPTY_RETRIES + 1):
        content = _original_completion(self, prompt, model)
        if content:
            return content
        print(f"[empty_response_retry] empty answer (attempt {attempt + 1}/{MAX_EMPTY_RETRIES + 1})")
    return ""


async def _acompletion_with_retry(self, prompt, model=None):
    for attempt in range(MAX_EMPTY_RETRIES + 1):
        content = await _original_acompletion(self, prompt, model)
        if content:
            return content
        print(f"[empty_response_retry] empty answer (attempt {attempt + 1}/{MAX_EMPTY_RETRIES + 1})")
    return ""


OpenAIClient.completion = _completion_with_retry
OpenAIClient.acompletion = _acompletion_with_retry
