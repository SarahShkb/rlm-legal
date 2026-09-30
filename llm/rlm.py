import os

from dotenv import load_dotenv

# rlm
from rlm.core.rlm import RLM
from rlm.logger import RLMLogger
from rlm.utils.prompts import RLM_SYSTEM_PROMPT

from prompts.prompts import (
    CRASH_PROOF_FINAL_SUBMISSION,
    CRITICAL_OUTPUT_RULE,
    PRESERVE_VALID_ANSWERS,
    REPL_STDOUT_VISIBILITY,
    RLM_FINISH_GUIDE,
    SUB_LLM_VERIFICATION,
)

# logger
from utils.run_logger import log_run

load_dotenv()
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
DEEPSEEK_MODEL = os.environ.get('DEEPSEEK_MODEL_NAME')
DEEPSEEK_BASE_URL = os.environ.get('DEEPSEEK_BASE_URL')
TEMPERATURE = 0 


#-----------------------------------------------
# condition 2: RLM calls (Generic system prompt)
#-----------------------------------------------
def run_rlm_generic(dataset: str, row_id: str, contract_text: str, TASK_SPEC: str) -> str:
    root_prompt = f"{TASK_SPEC}\n\n{RLM_FINISH_GUIDE}\n\n"
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
        dataset=dataset,
        row_id=row_id,
    )
    return result.response


#-----------------------------------------------
# condition 3: RLM calls (Custom system prompt)
#-----------------------------------------------
def run_rlm_customized(
        dataset: str,
        row_id: str,
        contract_text: str,
        TASK_SPEC: str,
        experiment="rlm-comparing-vanilla-vs-rlmgGeneric-vs-rlmCustomized",
        max_iterations=5
    ) -> str:
    custom_system_prompt = (
        RLM_SYSTEM_PROMPT
        + f"\n\n{TASK_SPEC}\n\n{RLM_FINISH_GUIDE}\n\n\n\n"
        + "Reminder: llm_query() calls are isolated completions with no memory "
        "of this system prompt -- restate the label set and the relevant "
        "data inside any llm_query() prompt you write."
        + f"\n\nCRITICAL OUTPUT RULE: {CRITICAL_OUTPUT_RULE}"
        + f"\n\nCRASH-PROOF FINAL SUBMISSION: {CRASH_PROOF_FINAL_SUBMISSION}"
        + f"\n\nREPL STDOUT VISIBILITY: {REPL_STDOUT_VISIBILITY}"
        + f"\n\nPRESERVE VALID ANSWERS: {PRESERVE_VALID_ANSWERS}"
    )
    root_prompt = f"{TASK_SPEC}\n\n{RLM_FINISH_GUIDE}\n\n"
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
        max_iterations=max_iterations,
        max_tokens=500_000,
        max_errors=3,
        logger=RLMLogger(log_dir="./logs"),
        custom_system_prompt=custom_system_prompt,
    )
    result = rlm.completion(prompt=contract_text, root_prompt=root_prompt)
    log_run(
        result,
        method="RLM-customized",
        model=DEEPSEEK_MODEL,
        experiment=experiment,
        dataset=dataset,
        rlm=rlm,
        row_id=row_id,
    )
    return result.response
