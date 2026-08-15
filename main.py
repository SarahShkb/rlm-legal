from rlms import RLM
from utils.run_logger import log_run

rlm = RLM(
    backend="openai",
    backend_kwargs={
        "model_name": "deepseek-v4-flash",
    },
    verbose=True,
)

dataset = []

for row_id, example in enumerate(dataset):

    result = rlm.completion(
        prompt=example["context"],
        root_prompt=example["question"],
    )

    log_run(
        result,
        rlm,
        experiment="rlm_ecourt",
        dataset="ECtHR_A",
        row_id=row_id,
    )

    print(f"Finished example {row_id}")