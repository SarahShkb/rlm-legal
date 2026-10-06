import json
import os
import re
from pathlib import Path

from transformers import AutoTokenizer

from utils.maud_utils import get_main_def, get_maud_data_dir, get_text_token_count

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
        n_tokens = get_text_token_count(contract_text, tokenizer)

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


def save_contracts_token_count_to_json(save_path: str="maud/dicts/maud_contract_lengths.json"):
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


    print("\nSaved: maud/dicts/maud_contract_lengths.json")




#   "sorted_contract_ids": [
#     88,
#     119,
#     6,
#     59,
#     21,
#     112,
#     106,
#     43,
#     138,
#     11,
#     125,
#     55,
#     97,
#     32,
#     113,
#     30,
#     87,
#     115,
#     136,
#     102,
#     124,
#     78,
#     83,
#     75,
#     65,
#     133,
#     47,
#     85,
#     127,
#     82,
#     16,
#     141,
#     39,
#     64,
#     58,
#     130,
#     67,
#     139,
#     14,
#     63,
#     96,
#     101,
#     86,
#     116,
#     129,
#     74,
#     95,
#     104,
#     12,
#     25,
#     146,
#     98,
#     3,
#     72,
#     9,
#     89,
#     62,
#     77,
#     110,
#     36,
#     18,
#     69,
#     61,
#     137,
#     40,
#     107,
#     0,
#     132,
#     7,
#     15,
#     49,
#     48,
#     4,
#     93,
#     144,
#     122,
#     22,
#     114,
#     71,
#     45,
#     60,
#     131,
#     52,
#     1,
#     103,
#     142,
#     35,
#     135,
#     29,
#     148,
#     149,
#     28,
#     53,
#     13,
#     143,
#     90,
#     134,
#     31,
#     17,
#     80,
#     117,
#     56,
#     26,
#     37,
#     120,
#     151,
#     123,
#     50,
#     19,
#     91,
#     128,
#     99,
#     34,
#     126,
#     42,
#     79,
#     100,
#     109,
#     70,
#     41,
#     23,
#     54,
#     2,
#     150,
#     92,
#     38,
#     147,
#     108,
#     20,
#     111,
#     57,
#     140,
#     5,
#     51,
#     46,
#     66,
#     118,
#     94,
#     76,
#     105,
#     44,
#     27,
#     8,
#     121,
#     24,
#     81,
#     73,
#     84,
#     10,
#     68,
#     145,
#     33
#   ]