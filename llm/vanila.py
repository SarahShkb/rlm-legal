from openai import OpenAI

# logger
from utils.run_logger import log_run

from llm.config import API_KEY, BASE_URL, EXTRA_BODY, MODEL, TEMPERATURE


#-----------------------------------------
# condition 1: vanilla LLM calls (no RLM)
#-----------------------------------------
def run_vanilla(dataset: str,row_id: str, contract_text: str, TASK_SPEC: str, experiment="rlm-comparing-vanilla-vs-rlmgGeneric-vs-rlmCustomized") -> str:
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": f"{TASK_SPEC}\n\nCONTRACT:\n{contract_text}"},
        ],
        stream=False,
        extra_body=EXTRA_BODY
    )

    log_run(
        response,
        method="vanilla",
        model=MODEL,
        experiment=experiment,
        dataset=dataset,
        rlm={},
        total_tokens=response.usage.total_tokens,
        output_tokens=response.usage.completion_tokens,
        input_tokens=response.usage.prompt_tokens,
        row_id=row_id,
    )
    return response.choices[0].message.content
