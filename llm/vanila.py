from openai import OpenAI

# logger
from llm.empty_response_retry import MAX_EMPTY_RETRIES
from llm.rlm_usage_patch import cached_prompt_tokens
from utils.run_logger import log_run

from llm.config import API_KEY, BASE_URL, EXTRA_BODY, MODEL, TEMPERATURE


#-----------------------------------------
# condition 1: vanilla LLM calls (no RLM)
#-----------------------------------------
def run_vanilla(
        dataset: str,
        row_id: str,
        contract_text: str,
        TASK_SPEC: str,
        experiment="rlm-comparing-vanilla-vs-rlmgGeneric-vs-rlmCustomized",
        temperature: float | None = None,
        context_label: str = "CONTRACT",
        runs_csv: str = "logs/runs.csv",
    ) -> str:
    """temperature: None keeps the old behaviour (not sent, provider default);
    pass TEMPERATURE to match the RLM sampling args.
    context_label: heading before the document (LexGLUE documents aren't contracts).
    runs_csv: where log_run appends the summary row."""
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,)

    sampling = {} if temperature is None else {"temperature": temperature}
    # an empty answer is a malformed provider response: re-send (see llm/empty_response_retry.py)
    tokens = {"total": 0, "output": 0, "input": 0, "cached": 0}
    for _ in range(MAX_EMPTY_RETRIES + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": f"{TASK_SPEC}\n\n{context_label}:\n{contract_text}"},
            ],
            stream=False,
            extra_body=EXTRA_BODY,
            **sampling,
        )
        tokens["total"] += response.usage.total_tokens
        tokens["output"] += response.usage.completion_tokens
        tokens["input"] += response.usage.prompt_tokens
        tokens["cached"] += cached_prompt_tokens(response.usage)
        if response.choices[0].message.content:
            break

    log_run(
        response,
        method="vanilla",
        model=MODEL,
        experiment=experiment,
        dataset=dataset,
        rlm={},
        total_tokens=tokens["total"],
        output_tokens=tokens["output"],
        input_tokens=tokens["input"],
        cached_input_tokens=tokens["cached"],
        row_id=row_id,
        csv_path=runs_csv,
    )
    return response.choices[0].message.content
