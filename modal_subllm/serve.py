"""vLLM servers on Modal for the RLM sub-LLM (llm_query) experiments on MAUD.

Every model is served the same way: vLLM VLLM_VERSION on one GPU, bf16,
--max-model-len 8192, vLLM's own sampling defaults (not the model's
generation_config.json, which may ship its own temperature / top_p), weights
pinned to a commit and read from a Modal Volume (no download at start-up).

Access control, two layers:
  - requires_proxy_auth: Modal rejects requests without a proxy auth token
    (Modal-Key / Modal-Secret headers) BEFORE a container starts, so a stranger
    who finds the URL cannot wake (and bill) a GPU;
  - vllm serve --api-key: the server itself also checks a bearer key.
max_containers=1 caps each model at one GPU whatever the load.

Setup (once):
  modal secret create subllm-api-key VLLM_API_KEY=<random key, same as SUBLLM_API_KEY in .env>
  modal secret create huggingface-secret HF_TOKEN=<Hugging Face token>
  # Modal dashboard -> Settings -> Proxy Auth Tokens -> new token
  #   -> MODAL_PROXY_TOKEN_ID / MODAL_PROXY_TOKEN_SECRET in .env
  modal run modal_subllm/serve.py::download --model saul     # CPU container, no GPU billed
  modal run modal_subllm/serve.py::download --model mistral  # (and lawma)
  modal deploy modal_subllm/serve.py                          # idle endpoints cost nothing

  python -m llm.sub_llm check saul      # 401 without credentials, then one tiny completion
  modal app stop rlm-subllm             # when done

Endpoints: https://<workspace>--rlm-subllm-<model>.modal.run/v1
"""

import os
import subprocess

import modal

APP_NAME = "rlm-subllm"
VLLM_VERSION = "0.31.0"
HF_HUB_VERSION = "1.29.0"
MAX_MODEL_LEN = 8192
DTYPE = "bfloat16"
GPU = "L40S"                 # 48 GB: 8B bf16 weights + KV cache for ~25 full 8K sequences
SCALEDOWN_S = 120            # idle seconds before the GPU container stops
MAX_INPUTS = 64              # concurrent requests per container (vLLM batches them)
STARTUP_S = 15 * 60
PORT = 8000
MODELS_DIR = "/models"

# key -> HF repo and the commit it is pinned to; the repo id is also the served model name
MODELS = {
    "saul": {"repo": "Equall/Saul-7B-Instruct-v1", "revision": "2133ba7923533934e78f73848045299dd74f08d2"},
    "mistral": {"repo": "mistralai/Mistral-7B-Instruct-v0.1", "revision": "ec5deb64f2c6e6fa90c1abf74a91d5c93a9669ca"},
    # Llama 3.1 8B (llama3 rope scaling in its config) fine-tuned on 260 US court classification tasks;
    # its chat template is the MMLU-style "...\nAnswer:" prompt it was trained on
    "lawma": {"repo": "ricdomolm/lawma-8b", "revision": "3b088a47c72e6ce572c273ba923af14f7dfe40ef"},
}
# HF-format files only: no pytorch_model*.bin duplicates, no original/ checkpoints,
# no mistral-format tokenizer (tokenizer.model.v1), so every model goes through
# the HF tokenizer and its own chat template
ALLOW_PATTERNS = [
    "config.json", "generation_config.json", "model.safetensors.index.json", "model-*.safetensors",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model",
]

app = modal.App(APP_NAME)
weights = modal.Volume.from_name("rlm-subllm-weights", create_if_missing=True)
vllm_cache = modal.Volume.from_name("rlm-subllm-vllm-cache", create_if_missing=True)

download_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(f"huggingface_hub=={HF_HUB_VERSION}")
serve_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(f"vllm=={VLLM_VERSION}")
    .env({"HF_HUB_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1"})
)


@app.function(image=download_image, volumes={MODELS_DIR: weights},
              secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
              timeout=60 * 60)
def download(model: str) -> None:
    from huggingface_hub import snapshot_download

    spec = MODELS[model]
    path = snapshot_download(spec["repo"], revision=spec["revision"], local_dir=f"{MODELS_DIR}/{model}",
                             allow_patterns=ALLOW_PATTERNS, token=os.environ["HF_TOKEN"])
    weights.commit()
    print(f"{spec['repo']}@{spec['revision'][:10]} -> {path}: {sorted(os.listdir(path))}")


def _start_vllm(model: str) -> None:
    cmd = [
        "vllm", "serve", f"{MODELS_DIR}/{model}",
        "--served-model-name", MODELS[model]["repo"],
        "--api-key", os.environ["VLLM_API_KEY"],
        "--dtype", DTYPE,
        "--max-model-len", str(MAX_MODEL_LEN),
        "--generation-config", "vllm",
        "--host", "0.0.0.0",
        "--port", str(PORT),
    ]
    subprocess.Popen(cmd)


SERVE = dict(
    image=serve_image,
    gpu=GPU,
    volumes={MODELS_DIR: weights, "/root/.cache/vllm": vllm_cache},
    secrets=[modal.Secret.from_name("subllm-api-key", required_keys=["VLLM_API_KEY"])],
    scaledown_window=SCALEDOWN_S,
    max_containers=1,
    timeout=20 * 60,
)


@app.function(**SERVE)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.web_server(port=PORT, startup_timeout=STARTUP_S, label=f"{APP_NAME}-saul", requires_proxy_auth=True)
def saul():
    _start_vllm("saul")


@app.function(**SERVE)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.web_server(port=PORT, startup_timeout=STARTUP_S, label=f"{APP_NAME}-mistral", requires_proxy_auth=True)
def mistral():
    _start_vllm("mistral")


@app.function(**SERVE)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.web_server(port=PORT, startup_timeout=STARTUP_S, label=f"{APP_NAME}-lawma", requires_proxy_auth=True)
def lawma():
    _start_vllm("lawma")

