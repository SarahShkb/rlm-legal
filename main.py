from maud.maud_data_sampling import run_maud_152_sample_pipeline
from maud.maud_type_of_consideration import run_maud_type_of_consideration_pipeline
from utils.contract_nli_utils import run_contract_nli_pipeline


def main():
    # phase 4.1
    #run_contract_nli_pipeline()

    # phase 4.2
    #run_maud_pipeline()

    # phase 4.3
    run_maud_152_sample_pipeline()

    # phase 5: using tools
    run_maud_152_sample_pipeline(tools={"1"})

 
if __name__ == "__main__":
    main()

