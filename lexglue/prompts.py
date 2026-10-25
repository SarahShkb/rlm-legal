"""LexGLUE task specs (TASK_SPEC) shared by the vanilla and RLM conditions.

Adapted from the Phase0 prompts (multi_class_classification_prompt,
multi_label_classification_prompt, multiple_choice_QA_prompt). Two changes:
  * the answer is a JSON object, as RLM_FINISH_GUIDE / CRITICAL_OUTPUT_RULE in
    prompts/prompts.py require ("answer['content'] must be ONLY the final JSON
    object"); vanilla gets the same format so both are parsed identically;
  * the document is not inlined here: run_vanilla appends it after the spec,
    the RLM receives it as its `context` variable.
Braces are single here; run_rlm_customized(escape_task_spec_braces=True)
doubles them for the system prompt, which RLM passes through str.format().
"""

from lexglue.config import MULTI_CLASS, MULTI_LABEL, TASKS
from lexglue.data import Record

# Heading run_vanilla puts before the document
CONTEXT_LABEL = "DOCUMENT"


def _label_block(labels: list[str]) -> str:
    return "\n".join(f"- {label}" for label in labels)


def multi_class_spec(task: str, labels: list[str]) -> str:
    return f"""
TASK: {TASKS[task]["description"]}

This is a multi-class legal document classification task.
Choose exactly ONE label from the candidate labels below.

CANDIDATE LABELS:
{_label_block(labels)}

The document to classify is provided separately (it is not part of these instructions).

Return ONLY a JSON object, with the label copied exactly as written in the list above:
{{"label": "<candidate label>"}}
""".strip()


def multi_label_spec(task: str, labels: list[str]) -> str:
    return f"""
TASK: {TASKS[task]["description"]}

This is a multi-label legal document classification task.
Select EVERY candidate label that applies. Zero, one, or multiple labels may apply.

CANDIDATE LABELS:
{_label_block(labels)}

The document to classify is provided separately (it is not part of these instructions).

Return ONLY a JSON object, with each label copied exactly as written in the list above:
{{"labels": ["<candidate label>", "..."]}}
If no label applies, return {{"labels": []}}.
""".strip()


def multiple_choice_spec(task: str, options: list[str]) -> str:
    holdings = "\n".join(f"{i}: {h}" for i, h in enumerate(options))
    valid = ", ".join(str(i) for i in range(len(options)))
    return f"""
TASK: {TASKS[task]["description"]}

The judicial decision excerpt is provided separately (it is not part of these instructions).

CANDIDATE HOLDINGS:
{holdings}

Return ONLY a JSON object with the index of the correct holding (one of {valid}):
{{"answer": <index>}}
""".strip()


def build_task_spec(record: Record, labels: list[str] | None) -> str:
    task_type = TASKS[record.task]["task_type"]
    if task_type == MULTI_CLASS:
        return multi_class_spec(record.task, labels)
    if task_type == MULTI_LABEL:
        return multi_label_spec(record.task, labels)
    return multiple_choice_spec(record.task, record.options)
