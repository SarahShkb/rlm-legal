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
elif LLM_PROVIDER == 'openrouter':
    # Paid key: set OPENROUTER_API_KEY, OPENROUTER_MODEL and OPENROUTER_PROVIDER in .env.
    # Model and provider are required on purpose, so a run never falls back to a
    # moving alias (e.g. "~deepseek/...-latest") or to a differently quantized host.
    API_KEY = os.environ.get('OPENROUTER_API_KEY')
    if not API_KEY:
        raise ValueError("LLM_PROVIDER=openrouter but OPENROUTER_API_KEY is not set")
    BASE_URL = 'https://openrouter.ai/api/v1'
    MODEL = os.environ.get('OPENROUTER_MODEL')            # e.g. "deepseek/deepseek-v4-flash-0731"
    OPENROUTER_PROVIDER = os.environ.get('OPENROUTER_PROVIDER')  # endpoint tag, e.g. "deepinfra/fp8"
    if not MODEL or not OPENROUTER_PROVIDER:
        raise ValueError("LLM_PROVIDER=openrouter needs OPENROUTER_MODEL and OPENROUTER_PROVIDER")
    EXTRA_BODY = {
        # OpenRouter's equivalent of DeepSeek's {"thinking": {"type": "enabled"}}
        "reasoning": {"enabled": True},
        # every call is served by the same deployment, or fails loudly
        "provider": {
            "order": [OPENROUTER_PROVIDER],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
    }
elif LLM_PROVIDER == 'local':
    # vLLM checks requests against the key passed to `vllm serve --api-key`
    API_KEY = os.environ.get('OPENAI_API_KEY', 'local')
    BASE_URL = os.environ['OPENAI_BASE_URL']
    MODEL = os.environ['MODEL_NAME']
    EXTRA_BODY = {}
else:
    raise ValueError(f"Unknown LLM_PROVIDER={LLM_PROVIDER!r}, expected 'deepseek', 'openrouter' or 'local'")

TEMPERATURE = 0
