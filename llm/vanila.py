import os

from dotenv import load_dotenv
from openai import OpenAI

# logger
from utils.run_logger import log_run

load_dotenv()
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
DEEPSEEK_MODEL = os.environ.get('DEEPSEEK_MODEL_NAME')
DEEPSEEK_BASE_URL = os.environ.get('DEEPSEEK_BASE_URL')
TEMPERATURE = 0 


#-----------------------------------------
# condition 1: vanilla LLM calls (no RLM)
#-----------------------------------------
def run_vanilla(dataset: str,row_id: str, contract_text: str, TASK_SPEC: str) -> str:
    client = OpenAI(
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        base_url=DEEPSEEK_BASE_URL,)

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": f"{TASK_SPEC}\n\nCONTRACT:\n{contract_text}"},
        ],
        stream=False,
        extra_body={"thinking": {"type": "enabled"}}
    )

    log_run(
        response,
        method="vanilla",
        model=DEEPSEEK_MODEL,
        experiment="rlm-comparing-vanilla-vs-rlmgGeneric-vs-rlmCustomized",
        dataset=dataset,
        rlm={},
        total_tokens=response.usage.total_tokens,
        output_tokens=response.usage.completion_tokens,
        input_tokens=response.usage.prompt_tokens,
        row_id=row_id,
    )
    return response.choices[0].message.content
