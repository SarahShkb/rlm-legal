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