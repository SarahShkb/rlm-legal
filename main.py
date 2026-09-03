from utils.contract_nli_utils import run_contract_nli_pipeline
from utils.maud_utils import run_maud_pipeline


def main():
    # phase 4.1
    #run_contract_nli_pipeline()

    # phase 4.2
    run_maud_pipeline(limit=1)

 
if __name__ == "__main__":
    main()

