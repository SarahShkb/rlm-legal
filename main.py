from maud.maud_data_sampling import run_maud_152_sample_pipeline
from maud.maud_type_of_consideration import run_maud_type_of_consideration_pipeline
from utils.contract_nli_utils import run_contract_nli_pipeline


def main():
    # # phase 4.1
    # run_contract_nli_pipeline()


    # # phase 4.2
    # run_maud_152_sample_pipeline()


    # # phase 5: using tools
    # # just tool 1 (section locator)
    # run_maud_152_sample_pipeline(tools={"1"})
    # # just tool 2 (definitions resolver)
    # run_maud_152_sample_pipeline(tools={"2"})
    # tool 2 after the resolver fixes, with the term list in the system prompt + usage hints
    run_maud_152_sample_pipeline(tools={"2"}, list_in_prompt=True, hints=True,
                                 experiment="maud_152_tool2_fixed_list_hints")
    # # both tools
    # run_maud_152_sample_pipeline(tools={"1", "2"})

 
if __name__ == "__main__":
    main()

