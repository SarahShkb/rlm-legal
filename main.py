from maud.maud_contracts_token_analysis import (
    save_contracts_token_count_to_json,
)
from utils.maud_utils import run_maud_pipeline
from utils.contract_nli_utils import run_contract_nli_pipeline


def main():
    # phase 4.1
    #run_contract_nli_pipeline()

    # phase 4.2
    #run_maud_pipeline()
    save_contracts_token_count_to_json()

 
if __name__ == "__main__":
    main()

