import os

from dotenv import load_dotenv

# load_dotenv does not override variables already exported (e.g. by the sbatch script)
load_dotenv()

# "deepseek" (default, local machine via .env) or "local" (vLLM server on the HPC)
LLM_PROVIDER = os.environ.get('LLM_PROVIDER', 'deepseek').lower()

if LLM_PROVIDER == 'deepseek':
    API_KEY = os.environ.get('DEEPSEEK_API_KEY')
    BASE_URL = os.environ.get('OPENAI_BASE_URL', 'https://api.deepseek.com')
    MODEL = os.environ.get('MODEL_NAME', 'deepseek-v4-flash')
    # DeepSeek-specific request field to enable thinking mode
    EXTRA_BODY = {"thinking": {"type": "enabled"}}
elif LLM_PROVIDER == 'local':
    # vLLM checks requests against the key passed to `vllm serve --api-key`
    API_KEY = os.environ.get('OPENAI_API_KEY', 'local')
    BASE_URL = os.environ['OPENAI_BASE_URL']
    MODEL = os.environ['MODEL_NAME']
    EXTRA_BODY = {}
else:
    raise ValueError(f"Unknown LLM_PROVIDER={LLM_PROVIDER!r}, expected 'deepseek' or 'local'")

TEMPERATURE = 0
