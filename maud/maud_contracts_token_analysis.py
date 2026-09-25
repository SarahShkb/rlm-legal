import json
import os
import re
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from maud.maud_utils import get_main_def, get_maud_data_dir

# config

TOKENIZER_NAME = os.environ.get('DEEPSEEK_TOKENIZER_NAME')
NUM_EXPECTED_CONTRACTS = 152
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
main_df = get_main_def(split="train")
maud_data_dir = get_maud_data_dir()
contract_dir = Path(maud_data_dir) / "contracts"
files = list(contract_dir.glob("contract_*.txt"))

# extract contract id vs path in an array
def get_contract_array()-> list[dict]:
    contracts = []
    for file_path in files:
        match = re.fullmatch(
            r"contract_(\d+)\.txt",
            file_path.name
        )
        if not match:
            raise ValueError(
                f"Unexpected filename: {file_path.name}"
            )
        contract_id = int(match.group(1))
        contracts.append(
            {
                "contract_id": contract_id,
                "path": file_path,
            }
        )
    return contracts


def get_contracts_token_count(contracts: list[dict] | None = None) -> list[dict]:
    results = []

    for i, contract in enumerate(
        sorted(contracts, key=lambda x: x["contract_id"]),
        start=1
    ):

        contract_id = contract["contract_id"]
        path = contract["path"]
        contract_text = path.read_text(encoding="utf-8")

        encoded = tokenizer(
            contract_text,
            add_special_tokens=False,
            truncation=False,
            return_length=True,
        )
        n_tokens = encoded["length"][0]

        results.append(
            {
                "contract_id": contract_id,
                "file": path.name,
                "n_chars": len(contract_text),
                "n_tokens": n_tokens,
            }
        )
    return results

# from shortest to longest contract
# if you want from longest to shortest, just reverse the list
# by adding reverse=True to the sorted() function
def sort_contracts_by_token_count(results: list[dict], reverse: bool = False) -> list[dict]:
    return sorted(results, key=lambda x: x["n_tokens"], reverse=reverse)


def save_contracts_token_count_to_json(save_path: str="maud_contract_lengths.json"):
    contracts = get_contract_array()
    results = get_contracts_token_count(contracts)
    sorted_results = sort_contracts_by_token_count(results)

    with open(
    save_path,
    "w",
    encoding="utf-8"
    ) as f:

        json.dump(
            {
                "tokenizer": TOKENIZER_NAME,
                "contracts": sorted_results,
                "sorted_contract_ids": [contract["contract_id"] for contract in sorted_results],
            },
            f,
            indent=2,
        )


    print("\nSaved: maud_contract_lengths.json")
