import json

#----------------------------------------
# prompt components
#----------------------------------------

# TASK_SPEC: the label set, the JSON output schema, the evidence-citation requirement. 
# Task-defining, not mechanism-defining.
TASK_SPEC = """
TASK: Document-level Natural Language Inference on a legal contract (NDA).
For each hypothesis below, classify it as exactly one of: "Entailment", \
"Contradiction", "NotMentioned", and list the verbatim contract sentence(s) \
that support the decision (empty list if NotMentioned).
Return ONLY a JSON object of the form:
{{"<hypothesis_id>": {{"choice": "<Entailment|Contradiction|NotMentioned>", "spans": [<...>]}}, ...}}
"""

RLM_FINISH_GUIDE = """
You MUST use `repl` code blocks to inspect and reason over the context.

When you are ready to answer, DO NOT output the answer as Python code.

Instead, in your final `repl` block, execute exactly this pattern:

answer["content"] = <a JSON string containing your final ContractNLI result>
answer["ready"] = True

The value of answer["content"] must contain ONLY the final JSON object.
It must not contain markdown fences, Python code, explanations, or commentary.

Once you have enough evidence, stop exploring and submit the answer.

Budget awareness:
Do not begin a new exploration step when only 2 iterations remain.
Use the remaining iterations to finalize the answer and explicitly set:
answer["content"] = ...
answer["ready"] = True
"""


#--------------------------------------------------------
# methods
#--------------------------------------------------------

# HYPOTHESES_BLOCK: the specific hypothesis (or hypotheses) for this record, 
# verbatim from the dataset.
# def build_hypotheses_block(hypotheses: dict) -> str:
#     lines = ["HYPOTHESES:"]
#     for hyp_id, hyp_text in hypotheses.items():
#         lines.append(f'- {hyp_id}: "{hyp_text}"')
#     return "\n".join(lines)

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