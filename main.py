from rlms import RLM
import os
from dotenv import load_dotenv


load_dotenv()
OPEN_ROUTER_API_KEY = os.getenv('OPEN_ROUTER_API_KEY')

rlm = RLM(
    backend="openrouter",
    backend_kwargs={
        "api_key": OPEN_ROUTER_API_KEY,
        "model_name": "qwen/qwen3-embedding-8b"
    },
)

print(rlm.completion("Print me the first 100 powers of two, each on a newline.").response)