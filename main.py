from rlms import RLM

rlm = RLM(
    backend="openai",
    backend_kwargs={
        "model_name": "deepseek-v4-flash",
    },
    verbose=True,
)
