import os

from dotenv import load_dotenv
from openai import OpenAI

# rlm
from rlm.core.rlm import RLM
from rlm.logger import RLMLogger
from rlm.utils.prompts import RLM_SYSTEM_PROMPT

#utils
from utils.contract_nli_utils import (
    CRASH_PROOF_FINAL_SUBMISSION,
    CRITICAL_OUTPUT_RULE,
    PRESERVE_VALID_ANSWERS,
    REPL_STDOUT_VISIBILITY,
    RLM_FINISH_GUIDE,
    SUB_LLM_VERIFICATION,
    TASK_SPEC,
    build_hypotheses_block,
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
def run_vanilla(row_id: str, contract_text: str, hypotheses: str) -> str:
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
        extra_body={"thinking": {"type": "enabled"}}
    )

    log_run(
        response,
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
def run_rlm_generic(row_id: str, contract_text: str, hypotheses: str) -> str:
    root_prompt = f"{TASK_SPEC}{RLM_FINISH_GUIDE}\n\n{hypotheses}\n\n"
    root_prompt += "Remember: your last message must be the JSON answer, not a repl block. "
    root_prompt += "If you have already set answer['ready']=True, simply output the JSON object."
    root_prompt += "\n\nIf you are unsure, always prefer to set answer['ready']=True at least one turn before the last turn."
    root_prompt += f"\n\nVERIFICATION: {SUB_LLM_VERIFICATION}\n\n"
    rlm = RLM(
        backend="openai",
        backend_kwargs={
            "model_name": DEEPSEEK_MODEL,
            "api_key": DEEPSEEK_API_KEY,
            "base_url": DEEPSEEK_BASE_URL,
        },
        verbose=True,
        sampling_args={
            "temperature": TEMPERATURE,
            "extra_body":{"thinking": {"type": "enabled"}}
        },
        sub_sampling_args={
            "temperature": TEMPERATURE,
            "reasoning_effort": "low",
            "extra_body":{"thinking": {"type": "enabled"}}
        },
        logger=RLMLogger(log_dir="./logs"),
        max_iterations=5,
            max_tokens=500_000,
            max_errors=3,
    )
    result = rlm.completion(prompt=contract_text, root_prompt=root_prompt)
    log_run(
        result,
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
def run_rlm_customized(row_id: str, contract_text: str, hypotheses: str) -> str:
    custom_system_prompt = (
        RLM_SYSTEM_PROMPT
        + f"\n\n{TASK_SPEC}\n\n{RLM_FINISH_GUIDE}\n\n\n\n"
        + "Reminder: llm_query() calls are isolated completions with no memory "
        "of this system prompt -- restate the label set and the relevant "
        "hypothesis text inside any llm_query() prompt you write."
        + f"\n\nCRITICAL OUTPUT RULE: {CRITICAL_OUTPUT_RULE}"
        + f"\n\nCRASH-PROOF FINAL SUBMISSION: {CRASH_PROOF_FINAL_SUBMISSION}"
        + f"\n\nREPL STDOUT VISIBILITY: {REPL_STDOUT_VISIBILITY}"
        + f"\n\nPRESERVE VALID ANSWERS: {PRESERVE_VALID_ANSWERS}"
    )
    root_prompt = f"{TASK_SPEC}{RLM_FINISH_GUIDE}\n\n{hypotheses}\n\n"
    root_prompt += "Remember: your last message must be the JSON answer, not a repl block. "
    root_prompt += "If you have already set answer['ready']=True, simply output the JSON object."
    root_prompt += "\n\nIf you are unsure, always prefer to set answer['ready']=True at least one turn before the last turn."
    root_prompt += f"\n\nVERIFICATION: {SUB_LLM_VERIFICATION}\n\n"
    rlm = RLM(
        backend="openai",
        backend_kwargs={
            "model_name": DEEPSEEK_MODEL,
            "api_key": DEEPSEEK_API_KEY,
            "base_url": DEEPSEEK_BASE_URL,
        },
        verbose=True,
        sampling_args={
            "temperature": TEMPERATURE,
            "extra_body":{"thinking": {"type": "enabled"}}
        },
        sub_sampling_args={
            "temperature": TEMPERATURE,
            "reasoning_effort": "low",
            "extra_body":{"thinking": {"type": "enabled"}}
        },
        max_iterations=5,
        max_tokens=500_000,
        max_errors=3,
        logger=RLMLogger(log_dir="./logs"),
        custom_system_prompt=custom_system_prompt
    )
    result = rlm.completion(prompt=contract_text, root_prompt=root_prompt)
    log_run(
        result,
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

    # target_doc_ids = ["19", "20", "32"]

    for i in range(5):

        row_id, contract_text = load_one_record(CONTRACT_PATH, doc_index=i)
        hypotheses = build_hypotheses_block(CONTRACT_PATH)
        print("=== 1. VANILLA ===")
        print(run_vanilla(f"{row_id}", contract_text, hypotheses))
        
        # print("\n=== 2. RLM DEFAULT SYSTEM PROMPT ===")
        # print(run_rlm_generic(f"{row_id}", contract_text, hypotheses))
        
        # print("\n=== 3. RLM CUSTOM SYSTEM PROMPT ===")
        # print(run_rlm_customized(f"{row_id}", contract_text, hypotheses))

 
if __name__ == "__main__":
    main()

