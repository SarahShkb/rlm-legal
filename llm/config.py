import os

from dotenv import load_dotenv

# load_dotenv does not override variables already exported (e.g. by the sbatch script)
load_dotenv()

# "deepseek" (default, local machine via .env) or "local" (vLLM server on the HPC)
LLM_PROVIDER = os.environ.get('LLM_PROVIDER', 'deepseek').lower()

# extra HTTP headers for the root's client (the Modal proxy token when LLM_PROVIDER=modal)
DEFAULT_HEADERS = None

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
    # "0" for models without a reasoning mode (e.g. qwen3-30b-a3b-instruct-2507): with
    # require_parameters, a reasoning field makes every endpoint refuse the request
    OPENROUTER_REASONING = os.environ.get('OPENROUTER_REASONING', '1') == '1'
    EXTRA_BODY = {
        # OpenRouter's equivalent of DeepSeek's {"thinking": {"type": "enabled"}}
        **({"reasoning": {"enabled": True}} if OPENROUTER_REASONING else {}),
        # every call is served by the same deployment, or fails loudly
        "provider": {
            "order": [OPENROUTER_PROVIDER],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
    }
elif LLM_PROVIDER == 'modal':
    # root served by modal_subllm/serve.py: MODAL_ROOT = its key there (e.g. "rlm30b"),
    # MODEL_NAME = its served model name; same credentials as the Modal sub-LLMs
    MODAL_ROOT = os.environ['MODAL_ROOT']
    API_KEY = os.environ['SUBLLM_API_KEY']
    BASE_URL = (os.environ.get(f'SUBLLM_{MODAL_ROOT.upper()}_URL')
                or f"https://{os.environ['MODAL_WORKSPACE']}--rlm-subllm-{MODAL_ROOT}.modal.run/v1")
    MODEL = os.environ['MODEL_NAME']
    EXTRA_BODY = {}
    DEFAULT_HEADERS = {"Modal-Key": os.environ['MODAL_PROXY_TOKEN_ID'],
                       "Modal-Secret": os.environ['MODAL_PROXY_TOKEN_SECRET']}
elif LLM_PROVIDER == 'local':
    # vLLM checks requests against the key passed to `vllm serve --api-key`
    API_KEY = os.environ.get('OPENAI_API_KEY', 'local')
    BASE_URL = os.environ['OPENAI_BASE_URL']
    MODEL = os.environ['MODEL_NAME']
    EXTRA_BODY = {}
else:
    raise ValueError(f"Unknown LLM_PROVIDER={LLM_PROVIDER!r}, expected 'deepseek', 'openrouter', 'modal' or 'local'")

TEMPERATURE = 0


# Root context window enforced client-side (llm/root_window.py), e.g. 32768 to run a
# 262K-context model as a 32K one; counted with ROOT_TOKENIZER (a Hugging Face repo id)
ROOT_CONTEXT_TOKENS = int(os.environ.get('ROOT_CONTEXT_TOKENS', '0')) or None
ROOT_TOKENIZER = os.environ.get('ROOT_TOKENIZER')
if ROOT_CONTEXT_TOKENS and not ROOT_TOKENIZER:
    raise ValueError("ROOT_CONTEXT_TOKENS needs ROOT_TOKENIZER (the root model's Hugging Face repo id)")
