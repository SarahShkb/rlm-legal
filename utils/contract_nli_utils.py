import json

#----------------------------------------
# prompt components
#----------------------------------------

# TASK_SPEC: the label set, the JSON output schema, the evidence-citation requirement. 
# Task-defining, not mechanism-defining.
TASK_SPEC = """
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

RLM_FINISH_GUIDE = """
Use the REPL to inspect the contract and reason about the hypothesis.

When you have enough evidence, STOP exploring and terminate.

To terminate, execute a REPL block that sets:

answer["content"] = "<final JSON object>"
answer["ready"] = True

The value of answer["content"] must be ONLY the final JSON object.
Do not put Python code, markdown, or explanation inside answer["content"].
"""

CRITICAL_OUTPUT_RULE = """
- In every turn except the final one, you MUST output exactly one ```repl``` block.
- Do NOT output free-form prose without a repl block – it breaks the REPL state.
- The final answer (the one the user sees) must be a plain JSON object, NOT inside a repl block.
- To prepare the final answer, set answer["content"] and answer["ready"] = True inside a repl block in an earlier turn.
- If you have already set answer["ready"] = True, the next turn should output ONLY the JSON (no repl).
"""

CRASH_PROOF_FINAL_SUBMISSION = """
When you are within the last 3 turns and you intend to set answer["ready"] = True, 
your code MUST be structured as follows:
try:
    # Your LLM calls and parsing
    ...
    # set answer
except Exception:
    # Hardcoded fallback (e.g., using the last valid state or a conservative default)
    answer["content"] = json.dumps({{...}})  # fallback JSON
    answer["ready"] = True
"""

REPL_STDOUT_VISIBILITY = """
Every time you execute a ```repl``` block, its printed output (stdout) will be visible to you 
in the next turn's prompt, between the "Turn N/20" messages. 
If your REPL produces no output, you will see an empty observation.
You can always rely on seeing the output; if you don't see it, it means the REPL produced no output or errored.
"""

PRESERVE_VALID_ANSWERS = """
If you have already set answer["ready"] = True and answer["content"] contains a valid JSON object,
DO NOT overwrite it in later turns unless you have strong evidence that it is wrong.
Instead, simply output the JSON object as your final message (without a repl block).
"""

SUB_LLM_VERIFICATION = """
After every REPL block you execute, the output (stdout) will be shown to you in the next turn.
If you do not see any output (or see an error), it means your REPL code was not valid.
In that case, you should treat that turn as having failed and try again with corrected code.
"""

#--------------------------------------------------------
# methods
#--------------------------------------------------------

# HYPOTHESES_BLOCK: the specific hypothesis (or hypotheses) for this record, 
# verbatim from the dataset.
def build_hypotheses_block(contractnli_json_path: str) -> str:
    with open(contractnli_json_path) as f:
        data = json.load(f)
    hypotheses = data["labels"]
    lines = ["HYPOTHESES:"]
    for key, value in hypotheses.items():
        lines.append(f'-{key}: {value["hypothesis"]}')
    return "\n".join(lines)

# get one record of hypothesis
def get_hypothesis(contractnli_json_path: str, h_id: str = "nda-1") -> str:
    with open(contractnli_json_path) as f:
        data = json.load(f)
    hypes = data["labels"]
    return f"({hypes[h_id]['short_description']}){hypes[h_id]['hypothesis']}"
 
 
def load_one_record(contractnli_json_path: str, doc_index: int = 0):
    with open(contractnli_json_path) as f:
        data = json.load(f)
    doc = data["documents"][doc_index]
    contract_text = doc["text"]
    contract_id = doc["id"]
    data_split_type = contractnli_json_path.split("/")[-1].split(".")[0]  # "train" | "dev" |"test"
    row_id = f"{data_split_type}-{contract_id}"
    return row_id, contract_text


def get_actual_label(contractnli_json_path: str, doc_index: int = 0, h_id: str = "nda-1") -> str:
    with open(contractnli_json_path) as f:
        data = json.load(f)
    doc = data["documents"][doc_index]
    actual_label = doc["annotation_sets"][0]["annotations"][h_id]["choice"]
    return actual_label

def get_document_index_by_id(contractnli_json_path: str, doc_id: str) -> int:
    with open(contractnli_json_path) as f:
        data = json.load(f)
    for i, doc in enumerate(data["documents"]):
        print(f"doc_id: {doc['id']}, target: {doc_id}")
        if str(doc["id"]) == doc_id:
            return i
    raise ValueError(f"Document ID {doc_id} not found in {contractnli_json_path}.")