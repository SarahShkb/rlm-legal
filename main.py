import os

from dotenv import load_dotenv
from openai import OpenAI

# rlm
from rlm.core.rlm import RLM
from rlm.logger import RLMLogger
from rlm.utils.prompts import RLM_SYSTEM_PROMPT

#utils
from utils.contract_nli_utils import (
    RLM_FINISH_GUIDE,
    TASK_SPEC,
    get_actual_label,
    get_document_index_by_id,
    get_hypothesis,
    load_one_record,
)

# logger
from utils.run_logger import log_run

load_dotenv()
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
TEMPERATURE = 0 


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

    target_doc_ids = ["19", "20", "32"]
    target_hypotheses = [
        ["nda-15", "nda-19", "nda-7", "nda-2", "nda-11", "nda-16"],
        ["nda-15", "nda-16", "nda-1", "nda-2", "nda-11", "nda-10"],
        ["nda-15", "nda-16", "nda-7", "nda-2", "nda-11", "nda-19"]]

    for i, doc_id in enumerate(target_doc_ids):

        doc_index = get_document_index_by_id(CONTRACT_PATH, doc_id)
        row_id, contract_text = load_one_record(CONTRACT_PATH, doc_index=doc_index)
        for hypothesis_id in target_hypotheses[i]:
            hypotheses = get_hypothesis(CONTRACT_PATH, h_id=hypothesis_id)
            actual_label = get_actual_label(CONTRACT_PATH, doc_index=doc_index, h_id=hypothesis_id)
            print("=== 1. VANILLA ===")
            print(run_vanilla(f"{row_id}-{hypothesis_id}", contract_text, hypotheses, actual_label))
        
            print("\n=== 2. RLM DEFAULT SYSTEM PROMPT ===")
            print(run_rlm_generic(f"{row_id}-{hypothesis_id}", contract_text, hypotheses, actual_label))
        
            print("\n=== 3. RLM CUSTOM SYSTEM PROMPT ===")
            print(run_rlm_customized(f"{row_id}-{hypothesis_id}", contract_text, hypotheses, actual_label))

 
if __name__ == "__main__":
    main()

