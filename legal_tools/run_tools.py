"""
run_maud_tools.py — minimal example: one MAUD question with RLM + any tool combination.

Copy the parts you need into your existing runner. Keep everything else
(model, max_iterations, system prompt, answer recovery) identical to your
baseline run, so the only difference between conditions is the tools.

    export DEEPSEEK_API_KEY=...
    python run_maud_tools.py --maud ~/maud --contract 34 --question "..." --tools 1,2
    python run_maud_tools.py --maud ~/maud --contract 34 --question "..." --tools none
"""

import argparse
import os

import legal_tools.definitions_resolver as tool2
import legal_tools.section_locator as tool1
from legal_tools.tools_utils import (
    ToolLog,
    context_offset,
    load_maud_contract,
    merge_custom_tools,
)
from llm.rlm import RLM


def build_tools(text: str, which: set[str], log: ToolLog, context: str | None = None) -> dict:
    """custom_tools dict for the chosen tools, all bound to ONE contract."""
    parts = []
    sec = tool1.SectionIndex(text, base=context_offset(text, context)) if "1" in which else None
    if "1" in which:
        parts.append(tool1.rlm_custom_tools(text, log, index=sec))
    if "2" in which:
        # with tool 1 present, tool 2 reuses its section map (labels, cross-ref following, terms_in)
        parts.append(tool2.rlm_custom_tools(text, log, sections=sec, context=context))
    return merge_custom_tools(*parts)


def make_rlm(custom_tools: dict | None, custom_system_prompt: str | None = None) -> RLM:
    kwargs = {
        "backend": "openai",                                   # DeepSeek is OpenAI-compatible
        "backend_kwargs": {
            "model_name": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),   # use your model name
            "api_key": os.environ["DEEPSEEK_API_KEY"],
            "base_url": "https://api.deepseek.com",
        },
        "environment": "local",
        "max_depth": 1,              # same as your baseline
        "max_iterations": 8,         # same as your MAUD baseline
    }
    if custom_system_prompt:
        # must contain the placeholder {custom_tools_section}, or the tool descriptions are lost
        kwargs["custom_system_prompt"] = custom_system_prompt
    if custom_tools:
        kwargs["custom_tools"] = custom_tools
        kwargs["custom_sub_tools"] = {}   # sub-calls get no tools (explicit; matters once max_depth > 1)
    return RLM(**kwargs)


def run_maud_with_tools(maud_dir: str, contract: str, question: str, which: set[str]) -> None:
    text = load_maud_contract(maud_dir, contract)       # this exact string is the RLM `context`
    log = ToolLog()
    tools = build_tools(text, which, log) if which else None
    rlm = make_rlm(tools)

    # Optional usage hint: identical wording in every tool condition.
    hints = [m.TOOL_PROMPT for k, m in (("1", tool1), ("2", tool2)) if k in which]
    if "2" in which and "1" not in which:
        hints[-1] = tool2.TOOL_PROMPT_NO_SECTIONS
    root = "\n\n".join([question] + hints)

    result = rlm.completion(text, root_prompt=root)
    print("ANSWER:", result.response)
    print("TOOLS:", log.summary())
    with open(f"toollog_contract_{contract}_tools-{','.join(sorted(which))}.json", "w", encoding="utf-8") as f:
        f.write(log.to_json())

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maud", required=True)
    ap.add_argument("--contract", required=True, help="e.g. 34 or contract_34")
    ap.add_argument("--question", required=True, help="question + answer options, as in your baseline")
    ap.add_argument("--tools", default="1", help="none | 1 | 2 | 1,2")
    a = ap.parse_args()

    which = set() if a.tools == "none" else set(a.tools.split(","))
    text = load_maud_contract(a.maud, a.contract)       # this exact string is the RLM `context`
    log = ToolLog()
    tools = build_tools(text, which, log) if which else None
    print(tools)
    # rlm = make_rlm(tools)

    # # Optional usage hint: identical wording in every tool condition.
    # hints = [m.TOOL_PROMPT for k, m in (("1", tool1), ("2", tool2)) if k in which]
    # if "2" in which and "1" not in which:
    #     hints[-1] = tool2.TOOL_PROMPT_NO_SECTIONS
    # root = "\n\n".join([a.question] + hints)

    # result = rlm.completion(text, root_prompt=root)
    # print("ANSWER:", result.response)
    # print("TOOLS:", log.summary())
    # with open(f"toollog_contract_{a.contract}_tools-{a.tools.replace(',', '')}.json", "w", encoding="utf-8") as f:
    #     f.write(log.to_json())


if __name__ == "__main__":
    main()