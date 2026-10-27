"""Run the root model with a smaller context window than its server allows.

With ROOT_CONTEXT_TOKENS set (llm/config.py), every root request is counted with
the root model's own chat template and tokenizer (ROOT_TOKENIZER) before it is
sent, and treated as a vLLM server started with --max-model-len would treat it:
a prompt that leaves no room for an answer is refused (ContextWindowExceededError,
the run fails), otherwise the answer may use the rest of the window.

Imported for its side effects from llm/rlm.py. Clients built for a sub-LLM
(llm/sub_llm.py, flagged sub_llm=True) are not limited: they have their own window.
"""

import functools

from rlm.clients.openai import OpenAIClient

from llm.config import ROOT_CONTEXT_TOKENS, ROOT_TOKENIZER


# calibrated on 19 MAUD vanilla prompts (2 messages each): the text count + 2 x 15 matched
# OpenRouter's prompt_tokens exactly for deepseek-v4.1-flash-1017; if part of it is per
# request rather than per message, long histories are slightly over-counted (safe side)
MESSAGE_OVERHEAD_TOKENS = 15


class ContextWindowExceededError(Exception):
    """The root's prompt does not fit in ROOT_CONTEXT_TOKENS."""


@functools.cache
def _tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(ROOT_TOKENIZER)


def prompt_tokens(prompt) -> int:
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    tok = _tokenizer()
    if tok.chat_template:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return len(tok(text, add_special_tokens=False)["input_ids"])
    # no chat template shipped (e.g. deepseek-ai/DeepSeek-V4.1-Flash-1017): the text of each
    # message + MESSAGE_OVERHEAD_TOKENS for its role markers
    return sum(len(tok(str(m.get("content") or ""), add_special_tokens=False)["input_ids"]) + MESSAGE_OVERHEAD_TOKENS
               for m in messages)


def _answer_budget(self, prompt) -> int:
    n = prompt_tokens(prompt)
    if n >= ROOT_CONTEXT_TOKENS:
        raise ContextWindowExceededError(
            f"root prompt has {n:,} tokens, the root context window is {ROOT_CONTEXT_TOKENS:,}")
    return ROOT_CONTEXT_TOKENS - n


def _limited(inner):
    def completion(self, prompt, model=None):
        if not ROOT_CONTEXT_TOKENS or getattr(self, "sub_llm", False):
            return inner(self, prompt, model)
        args = self.sampling_args
        self.sampling_args = {**(args or {}), "max_tokens": _answer_budget(self, prompt)}
        try:
            return inner(self, prompt, model)
        finally:
            self.sampling_args = args
    return completion


def _alimited(inner):
    async def acompletion(self, prompt, model=None):
        if not ROOT_CONTEXT_TOKENS or getattr(self, "sub_llm", False):
            return await inner(self, prompt, model)
        args = self.sampling_args
        self.sampling_args = {**(args or {}), "max_tokens": _answer_budget(self, prompt)}
        try:
            return await inner(self, prompt, model)
        finally:
            self.sampling_args = args
    return acompletion


OpenAIClient.completion = _limited(OpenAIClient.completion)
OpenAIClient.acompletion = _alimited(OpenAIClient.acompletion)
