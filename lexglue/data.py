"""Loading LexGLUE records and their gold labels (as readable label names)."""

import hashlib
import json
from dataclasses import dataclass, field

from datasets import load_dataset

from lexglue.config import (
    EURLEX_LABELS_PATH,
    HF_DATASET,
    MULTI_CLASS,
    MULTI_LABEL,
    MULTIPLE_CHOICE,
    TASKS,
)


@dataclass
class Record:
    task: str
    split: str
    index: int                       # position in the HF split
    row_id: str                      # "<task>-<split>-<index>"
    document: str                    # what the model reads (vanilla: in the user message, RLM: `context`)
    gold: str | list[str] | int      # readable label / list of labels / CaseHOLD index
    options: list[str] = field(default_factory=list)  # CaseHOLD candidate holdings

    @property
    def document_sha256(self) -> str:
        return hashlib.sha256(self.document.encode("utf-8")).hexdigest()


def load_split(task: str, split: str):
    return load_dataset(HF_DATASET, task, split=split)


def dataset_label_ids(task: str, features) -> list[str]:
    """ClassLabel names as stored in the dataset (e.g. '2', 'P1-1', '100163')."""
    task_type = TASKS[task]["task_type"]
    if task_type == MULTI_CLASS:
        return features["label"].names
    if task_type == MULTI_LABEL:
        return features["labels"].feature.names
    raise KeyError(f"{task} has no label vocabulary (task type {task_type})")


def label_names(task: str, features) -> list[str]:
    """Readable label names, aligned with dataset_label_ids()."""
    ids = dataset_label_ids(task, features)
    names = TASKS[task]["label_names"]
    if names is None:
        names = ids
    elif names == "eurovoc":
        with open(EURLEX_LABELS_PATH, encoding="utf-8") as f:
            eurovoc = json.load(f)
        names = [eurovoc[i] for i in ids]
    if len(names) != len(ids):
        raise ValueError(f"{task}: {len(names)} label names for {len(ids)} dataset labels")
    if len(set(names)) != len(names):
        raise ValueError(f"{task}: label names are not unique")
    return list(names)


def _document_text(task: str, item: dict) -> str:
    if task == "case_hold":
        return item["context"]
    text = item["text"]
    # ECtHR facts are a list of paragraphs
    return "\n".join(text) if isinstance(text, list) else text


def load_records(task: str, split: str, indices: list[int]) -> list[Record]:
    ds = load_split(task, split)
    task_type = TASKS[task]["task_type"]
    names = None if task_type == MULTIPLE_CHOICE else label_names(task, ds.features)

    records = []
    for i in indices:
        item = ds[i]
        if task_type == MULTI_CLASS:
            gold = names[item["label"]]
        elif task_type == MULTI_LABEL:
            gold = sorted(names[j] for j in item["labels"])
        else:
            gold = int(item["label"])
        records.append(Record(
            task=task,
            split=split,
            index=i,
            row_id=f"{task}-{split}-{i}",
            document=_document_text(task, item),
            gold=gold,
            options=list(item["endings"]) if task == "case_hold" else [],
        ))
    return records


def dataset_fingerprint(task: str, split: str) -> str:
    """HF fingerprint of the split; changes if the underlying data files change."""
    return load_split(task, split)._fingerprint
