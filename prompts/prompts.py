#----------------------------------------
# specific tasks prompt 
#----------------------------------------

CONTRACT_NLI_TASK_SPEC = """
TASK: Document-level Natural Language Inference on a legal contract (NDA).

For each hypothesis, classify it as exactly one of:
"Entailment", "Contradiction", "NotMentioned".

Return ONLY:
{{
  "<hypothesis_id>": {{
    "choice": "<Entailment|Contradiction|NotMentioned>",
  }}
}}
"""


#----------------------------------------
# custom system prompt components
#----------------------------------------

RLM_FINISH_GUIDE = """
Use the REPL to inspect the context and reasoning.

When you have enough evidence, STOP exploring and terminate.

To terminate, execute a REPL block that sets:

answer["content"] = "<final JSON object>"
answer["ready"] = True

The value of answer["content"] must be ONLY the final JSON object.
Do not put Python code, markdown, or explanation inside answer["content"].

IMPORTANT: End the loop when answer["ready"] is True and return answer["content"].
"""

CRITICAL_OUTPUT_RULE = """
- In every turn except the final one, you MUST output exactly one ```repl``` block.
- Do NOT output free-form prose without a repl block – it breaks the REPL state.
- The final answer (the one the user sees) must be a plain JSON object, NOT inside a repl block.
- To prepare the final answer, set answer["content"] and answer["ready"] = True inside a repl block in an earlier turn.
- If you have already set answer["ready"] = True, the next turn should output ONLY the JSON (no repl).
- IMPORTANT: End the loop when answer["ready"] is True and return answer["content"].
"""

CRASH_PROOF_FINAL_SUBMISSION = """
When you are within the last turn and you intend to set answer["ready"] = True, 
your code MUST be structured as follows:
try:
    # Your LLM calls and parsing
    ...
    # set answer
except Exception:
    # Hardcoded fallback (e.g., using the last valid state or a conservative default)
    answer["content"] = json.dumps({{...}})  # fallback JSON
    answer["ready"] = True


IMPORTANT: return answer["content"]. NEVER output a repl block in the final turn.
"""

REPL_STDOUT_VISIBILITY = """
Every time you execute a ```repl``` block, its printed output (stdout) will be visible to you 
in the next turn's prompt, between the "Turn n/N" messages. 
If your REPL produces no output, you will see an empty observation.
You can always rely on seeing the output; if you don't see it, it means the REPL produced no output or errored.
"""

PRESERVE_VALID_ANSWERS = """
If you have already set answer["ready"] = True and answer["content"] contains a valid JSON object,
DO NOT overwrite it in later turns unless you have strong evidence that it is wrong.
Instead, simply output the answer["content"] JSON object as your final message (without a repl block).
"""

SUB_LLM_VERIFICATION = """
After every REPL block you execute, the output (stdout) will be shown to you in the next turn.
If you do not see any output (or see an error), it means your REPL code was not valid.
In that case, you should treat that turn as having failed and try again with corrected code.
"""


#----------------------------------------
# small sub-LLM (llm/sub_llm.py): filled in with .format(), the same text for every sub-LLM
#----------------------------------------

SUB_LLM_LIMITS = """
In this run, llm_query() / llm_query_batched() are answered by a separate language model with a limited input size.
These limits replace the sub-LLM capacity stated above and in the context metadata (≈500K characters, ~100K characters per prompt, ~100k tokens per call), which does not apply here:
- Each prompt (your instructions + the text) must be at most {max_chars:,} characters. A longer prompt is not sent: you get a string starting with "Error:" instead. Pass excerpts of at most ~{chunk_chars:,} characters, never the whole `context`.
- Each answer is cut off after about {max_tokens:,} tokens: ask for short, structured outputs (e.g. the supporting sentence and the letter(s) it supports).
- The sub-LLM sees only the prompt you write: restate the question and the full list of answer choices in every prompt.
- An answer starting with "Error:" means the call failed: shorten the prompt or retry; it is not evidence.
"""

SUB_LLM_DELEGATION = """
Answer from the sub-LLM's reading of the contract, not from your own:
- Decide which parts of `context` the sub-LLM must read. When the question names something distinctive (a defined term, a clause title), Python string / regex search can find candidate sections quickly. Keyword hits are only candidates: contracts often express the same idea in different words, so when the question is about the meaning or effect of the language, or a search finds nothing convincing, split the whole `context` into consecutive excerpts of at most ~{chunk_chars:,} characters and send all of them to llm_query_batched.
- Every sub-LLM prompt holds one excerpt, the full question and all answer choices, and asks which choice(s) the excerpt supports (or "none") and for the sentence that supports it.
- Choose the final answer by combining the sub-LLM answers (e.g. the answer backed by the most specific quoted language, or the majority). Do not print long contract passages to read them yourself: print only the sub-LLM answers and short quotes.
"""

# with SUB_LLM_DELEGATION: appended to the system prompt and repeated in the root prompt
SUB_LLM_OUTPUT_LIMIT = """
In this run you see only the first {output_chars:,} characters of each REPL output block (this replaces the ~20K characters stated above); the rest is cut off. You cannot read the contract by printing it: the sub-LLM reads it for you through llm_query() / llm_query_batched(). Use print() for search results (positions, counts, section titles) and for a compact summary of the sub-LLM answers: keep the answers in Python variables and print e.g. the count of each answer letter and one or two short supporting quotes. Base your final answer on the sub-LLM answers.
"""
