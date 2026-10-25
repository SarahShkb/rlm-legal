"""LexGLUE experiment configuration: tasks, sample, defaults.

Ported from Phase0/LexGLUE_performance.ipynb (LEXGLUE_TASKS_DICT). The label
names and task descriptions come from the official zero-shot LexGLUE repo
(https://github.com/coastalcph/zeroshot_lexglue).
"""

from pathlib import Path

LEXGLUE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = LEXGLUE_DIR / "results"
EURLEX_LABELS_PATH = LEXGLUE_DIR / "data" / "eurlex_label_names.json"
HF_DATASET = "coastalcph/lex_glue"

# Same sample as Phase0: the first 100 records of the validation split of each task.
DEFAULT_SPLIT = "validation"
DEFAULT_START = 0
DEFAULT_N = 100

# Same value as the latest MAUD runs (run_maud_152_sample_pipeline)
DEFAULT_MAX_ITERATIONS = 8

# "vanilla" -> llm.vanila.run_vanilla, "rlm" -> llm.rlm.run_rlm_customized
METHODS = ("vanilla", "rlm")

MULTI_CLASS = "multi_class"
MULTI_LABEL = "multi_label"
MULTIPLE_CHOICE = "multiple_choice"

ECTHR_LABEL_NAMES = [
    'Article 2 - Right to life',
    'Article 3 - Prohibition of torture',
    'Article 5 - Right to liberty and security',
    'Article 6 - Right to a fair trial',
    'Article 8 - Right to respect for private and family life',
    'Article 9 - Freedom of thought, conscience and religion',
    'Article 10 - Freedom of expression',
    'Article 11 - Freedom of assembly and association',
    'Article 14 - Prohibition of discrimination',
    'Article P1-1 - Protection of property',
]

# label_names:
#   list     -> readable names, aligned with the dataset's ClassLabel ids
#   None     -> the dataset's ClassLabel names are already readable
#   "eurovoc"-> EuroVoc descriptors from EURLEX_LABELS_PATH
# none_class: LexGLUE scoring convention for ECtHR and UNFAIR-ToS, where an empty
#   label set is common: "no label" is scored as an extra class.
TASKS: dict[str, dict] = {
    "scotus": {
        "name": "SCOTUS",
        "task_type": MULTI_CLASS,
        "description": "Classify the United States Supreme Court opinion according to its primary legal issue area.",
        "label_names": ['Criminal Procedure', 'Civil Rights', 'First Amendment', 'Due Process', 'Privacy',
                        'Attorneys', 'Unions', 'Economic Activity', 'Judicial Power', 'Federalism',
                        'Interstate Relations', 'Federal Taxation', 'Miscellaneous'],
        "none_class": False,
    },
    "ledgar": {
        "name": "LEDGAR",
        "task_type": MULTI_CLASS,
        "description": "Classify the contract provision according to its clause category.",
        "label_names": None,
        "none_class": False,
    },
    "ecthr_a": {
        "name": "ECtHR_A",
        "task_type": MULTI_LABEL,
        "description": "Predict which Articles of the European Convention on Human Rights the applicant alleges were violated.",
        "label_names": ECTHR_LABEL_NAMES,
        "none_class": True,
    },
    "ecthr_b": {
        "name": "ECtHR_B",
        "task_type": MULTI_LABEL,
        "description": "Predict which Articles of the European Convention on Human Rights the Court found to have been violated.",
        "label_names": ECTHR_LABEL_NAMES,
        "none_class": True,
    },
    "eurlex": {
        "name": "EURLEX",
        "task_type": MULTI_LABEL,
        "description": "Assign all applicable EuroVoc legal concepts to this European Union legal document.",
        "label_names": "eurovoc",
        "none_class": False,
    },
    "unfair_tos": {
        "name": "UNFAIR_ToS",
        "task_type": MULTI_LABEL,
        "description": "Identify all categories of potentially unfair contractual terms that apply to this Terms of Service sentence.",
        "label_names": None,
        "none_class": True,
    },
    "case_hold": {
        "name": "CaseHOLD",
        "task_type": MULTIPLE_CHOICE,
        "description": (
            "You are given an excerpt from a judicial decision in which the holding of a cited case "
            "has been replaced by the <HOLDING> placeholder, and five candidate holdings. Select the "
            "candidate holding that correctly fills the <HOLDING> placeholder."
        ),
        "label_names": None,
        "none_class": False,
    },
}
