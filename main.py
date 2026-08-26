import json
import os

from dotenv import load_dotenv
from openai import OpenAI

# rlm
from rlm.core.rlm import RLM
from rlm.logger import RLMLogger
from rlm.utils.prompts import RLM_SYSTEM_PROMPT

# logger
from utils.run_logger import log_run

load_dotenv()
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
TEMPERATURE = 0 


# prompt components

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

#-----------------------------------------
# condition 1: vanilla LLM calls (no RLM)
#-----------------------------------------
def run_vanilla(row_id: str, contract_text: str, hypotheses: str, actual_label: str) -> str:
    client = OpenAI(
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        base_url=DEEPSEEK_BASE_URL,)

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": f"{TASK_SPEC}\n\n{hypotheses}\n\nCONTRACT:\n{contract_text}"},
        ],
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}}
    )

    log_run(
        response,
        actual=actual_label,
        method="vanilla",
        model=DEEPSEEK_MODEL,
        experiment="rlm-comparing-vanilla-vs-rlmgGeneric-vs-rlmCustomized",
        dataset="contract-nli",
        rlm={},
        total_tokens=response.usage.total_tokens,
        output_tokens=response.usage.completion_tokens,
        input_tokens=response.usage.prompt_tokens,
        row_id=row_id,
    )
    return response.choices[0].message.content


#-----------------------------------------------
# condition 2: RLM calls (Generic system prompt)
#-----------------------------------------------
def run_rlm_generic(row_id: str, contract_text: str, hypotheses: str, actual_label: str) -> str:
    root_prompt = f"{TASK_SPEC}{RLM_FINISH_GUIDE}\n\n{hypotheses}"
    rlm = RLM(
        backend="openai",
        backend_kwargs={
            "model_name": DEEPSEEK_MODEL,
            "api_key": DEEPSEEK_API_KEY,
            "base_url": DEEPSEEK_BASE_URL,
        },
        verbose=True,
        sampling_args={"temperature": TEMPERATURE},
        logger=RLMLogger(log_dir="./logs"),
        max_iterations=20,
            max_tokens=200_000,
            max_errors=3,
    )
    result = rlm.completion(prompt=contract_text, root_prompt=root_prompt)
    log_run(
        result,
        actual=actual_label,
        method="RLM-generic",
        model=DEEPSEEK_MODEL,
        rlm=rlm,
        experiment="rlm-comparing-vanilla-vs-rlmgGeneric-vs-rlmCustomized",
        dataset="contract-nli",
        row_id=row_id,
    )
    return result.response


#-----------------------------------------------
# condition 3: RLM calls (Custom system prompt)
#-----------------------------------------------
def run_rlm_customized(row_id: str, contract_text: str, hypotheses: str, actual_label: str) -> str:
    custom_system_prompt = (
        RLM_SYSTEM_PROMPT
        + f"\n\n{TASK_SPEC}\n\n{RLM_FINISH_GUIDE}\n\n"
        + "Reminder: llm_query() calls are isolated completions with no memory "
        "of this system prompt -- restate the label set and the relevant "
        "hypothesis text inside any llm_query() prompt you write."
    )
    rlm = RLM(
        backend="openai",
        backend_kwargs={
            "model_name": DEEPSEEK_MODEL,
            "api_key": DEEPSEEK_API_KEY,
            "base_url": DEEPSEEK_BASE_URL,
        },
        verbose=True,
        sampling_args={"temperature": TEMPERATURE},
        max_iterations=20,
        max_tokens=200_000,
        max_errors=3,
        logger=RLMLogger(log_dir="./logs"),
        custom_system_prompt=custom_system_prompt
    )
    result = rlm.completion(prompt=contract_text, root_prompt=hypotheses)
    log_run(
        result,
        actual=actual_label,
        method="RLM-customized",
        model=DEEPSEEK_MODEL,
        experiment="rlm-comparing-vanilla-vs-rlmgGeneric-vs-rlmCustomized",
        dataset="contract-nli",
        rlm=rlm,
        row_id=row_id,
    )
    return result.response



# Phase 4 of checklist.md: testing ONLY 1 sample from Contrat-NLI dev dataset
def main():
    CONTRACT_PATH = "./contract-nli/dev.json" 
    hypothesis_id = "nda-1"
    hypotheses = get_hypothesis(CONTRACT_PATH, h_id=hypothesis_id)

    for i in range(3):
        row_id, contract_text = load_one_record(CONTRACT_PATH, doc_index=i+3)
        actual_label = get_actual_label(CONTRACT_PATH, doc_index=i, h_id=hypothesis_id)
        print("=== 1. VANILLA ===")
        print(run_vanilla(f"{row_id}-{hypothesis_id}", contract_text, hypotheses, actual_label))
    
        print("\n=== 2. RLM DEFAULT SYSTEM PROMPT ===")
        print(run_rlm_generic(f"{row_id}-{hypothesis_id}", contract_text, hypotheses, actual_label))
    
        print("\n=== 3. RLM CUSTOM SYSTEM PROMPT ===")
        print(run_rlm_customized(f"{row_id}-{hypothesis_id}", contract_text, hypotheses, actual_label))

 
if __name__ == "__main__":
    main()

