import json

from llm.rlm import run_rlm_customized, run_rlm_generic
from llm.vanila import run_vanilla
from prompts.prompts import CONTRACT_NLI_TASK_SPEC

#--------------------------------------------------------
# methods
#--------------------------------------------------------

# HYPOTHESES_BLOCK: the specific hypothesis (or hypotheses) for this record, 
# verbatim from the dataset.
def build_hypotheses_block(contractnli_json_path: str) -> str:
    with open(contractnli_json_path) as f:
        data = json.load(f)
    hypotheses = data["labels"]
    lines = ["HYPOTHESES:"]
    for key, value in hypotheses.items():
        lines.append(f'-{key}: {value["hypothesis"]}')
    return "\n".join(lines)

# get one record of hypothesis
def get_hypothesis(contractnli_json_path: str, h_id: str = "nda-1") -> str:
    with open(contractnli_json_path) as f:
        data = json.load(f)
    hypes = data["labels"]
    return f"({hypes[h_id]['short_description']}){hypes[h_id]['hypothesis']}"
 
 
def load_one_record(contractnli_json_path: str, doc_index: int = 0):
    with open(contractnli_json_path) as f:
        data = json.load(f)
    doc = data["documents"][doc_index]
    contract_text = doc["text"]
    contract_id = doc["id"]
    data_split_type = contractnli_json_path.split("/")[-1].split(".")[0]  # "train" | "dev" |"test"
    row_id = f"{data_split_type}-{contract_id}"
    return row_id, contract_text


def get_actual_label(contractnli_json_path: str, doc_index: int = 0, h_id: str = "nda-1") -> str:
    with open(contractnli_json_path) as f:
        data = json.load(f)
    doc = data["documents"][doc_index]
    actual_label = doc["annotation_sets"][0]["annotations"][h_id]["choice"]
    return actual_label


def get_document_index_by_id(contractnli_json_path: str, doc_id: str) -> int:
    with open(contractnli_json_path) as f:
        data = json.load(f)
    for i, doc in enumerate(data["documents"]):
        print(f"doc_id: {doc['id']}, target: {doc_id}")
        if str(doc["id"]) == doc_id:
            return i
    raise ValueError(f"Document ID {doc_id} not found in {contractnli_json_path}.")



def run_contract_nli_pipeline():
    CONTRACT_PATH = "./contract-nli/main_repo/test.json" 
    TASK_SPEC = CONTRACT_NLI_TASK_SPEC
    with open(CONTRACT_PATH) as f:
        n_docs = len(json.load(f)["documents"])


    # 123 = full records
    for i in range(n_docs):
        row_id, contract_text = load_one_record(CONTRACT_PATH, doc_index=i)
        hypotheses = build_hypotheses_block(CONTRACT_PATH)
        task_prompt = f"{TASK_SPEC}\n\n{hypotheses}\n\n"
        print("=== 1. VANILLA ===")
        print(run_vanilla("contract-nli", f"{row_id}", contract_text, TASK_SPEC=task_prompt))
        
        print("\n=== 2. RLM DEFAULT SYSTEM PROMPT ===")
        print(run_rlm_generic("contract-nli", f"{row_id}", contract_text, TASK_SPEC=task_prompt))
        
        print("\n=== 3. RLM CUSTOM SYSTEM PROMPT ===")
        print(run_rlm_customized("contract-nli", f"{row_id}", contract_text, TASK_SPEC=task_prompt))