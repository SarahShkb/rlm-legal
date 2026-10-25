
# rlm
import llm.rlm_usage_patch  # noqa: F401  (fixes root-model usage being overwritten by sub-calls)
import llm.empty_response_retry  # noqa: F401  (re-sends requests whose answer came back empty)
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

from llm.config import API_KEY, BASE_URL, EXTRA_BODY, MODEL, TEMPERATURE


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
            "model_name": MODEL,
            "api_key": API_KEY,
            "base_url": BASE_URL,
        },
        verbose=True,
        sampling_args={
            "temperature": TEMPERATURE,
            "extra_body": EXTRA_BODY
        },
        sub_sampling_args={
            "temperature": TEMPERATURE,
            "reasoning_effort": "low",
            "extra_body": EXTRA_BODY
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
        model=MODEL,
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
        max_iterations=5,
        tools=None,
        tool_hint: str = "",
        escape_task_spec_braces: bool = False,
        on_result=None,
        runs_csv: str = "logs/runs.csv",
        log_dir: str = "./logs",
    ) -> str:
    """escape_task_spec_braces: RLM runs str.format() on the system prompt, so a
    TASK_SPEC written with single literal braces (e.g. a JSON example) must have
    them doubled there; the root prompt is not formatted and gets TASK_SPEC as-is.
    on_result: optional callback(result, rlm), called before logging, for callers
    that need the RLMChatCompletion (trajectory, execution time, logger path).
    runs_csv / log_dir: where the summary row and the RLM trajectory are written."""
    system_task_spec = (
        TASK_SPEC.replace("{", "{{").replace("}", "}}") if escape_task_spec_braces else TASK_SPEC
    )
    custom_system_prompt = (
        RLM_SYSTEM_PROMPT
        + f"\n\n{system_task_spec}\n\n{RLM_FINISH_GUIDE}\n\n\n\n"
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
    if tool_hint:
        root_prompt += f"TOOLS: {tool_hint}\n\n"
    rlm = RLM(
        backend="openai",
        backend_kwargs={
            "model_name": MODEL,
            "api_key": API_KEY,
            "base_url": BASE_URL,
        },
        verbose=True,
        sampling_args={
            "temperature": TEMPERATURE,
            "extra_body": EXTRA_BODY
        },
        sub_sampling_args={
            "temperature": TEMPERATURE,
            "reasoning_effort": "low",
            "extra_body": EXTRA_BODY
        },
        max_iterations=max_iterations,
        max_tokens=500_000,
        max_errors=3,
        logger=RLMLogger(log_dir=log_dir),
        custom_system_prompt=custom_system_prompt,
        custom_tools=tools,
    )
    result = rlm.completion(prompt=contract_text, root_prompt=root_prompt)
    if on_result is not None:
        on_result(result, rlm)
    log_run(
        result,
        method="RLM-customized",
        model=MODEL,
        experiment=experiment,
        dataset=dataset,
        rlm=rlm,
        row_id=row_id,
        csv_path=runs_csv,
    )
    return result.response
