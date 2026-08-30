import json
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "logs/rlm_2026-08-28_16-03-34_ea2507ac.jsonl"
OUTPUT_FILE = "rlm_2026-08-28_16-03-34_ea2507ac_debug_trace.md"


# ============================================================
# HELPERS
# ============================================================

def safe_text(value):
    """Convert any value into readable text."""
    if value is None:
        return "(None)"

    if isinstance(value, str):
        return value

    return json.dumps(
        value,
        indent=2,
        ensure_ascii=False,
        default=str
    )


def write_section(f, title, content, language="text"):
    """Write a Markdown section with a fenced code block."""

    if content is None:
        return

    text = safe_text(content)

    f.write(f"\n### {title}\n\n")
    f.write(f"```{language}\n")
    f.write(text)

    # Make sure the closing fence starts on a new line
    if not text.endswith("\n"):
        f.write("\n")

    f.write("```\n")


# ============================================================
# MAIN
# ============================================================

input_path = Path(INPUT_FILE)
output_path = Path(OUTPUT_FILE)

if not input_path.exists():
    raise FileNotFoundError(
        f"Input file not found: {input_path.resolve()}\n"
        "Please change INPUT_FILE at the top of this script."
    )


with input_path.open("r", encoding="utf-8") as infile, \
     output_path.open("w", encoding="utf-8") as outfile:

    outfile.write("# RLM Debug Trace\n\n")
    outfile.write(f"Source: `{input_path.name}`\n\n")

    for line_number, line in enumerate(infile, start=1):

        line = line.strip()

        if not line:
            continue

        try:
            entry = json.loads(line)

        except json.JSONDecodeError as e:
            outfile.write("\n---\n\n")
            outfile.write(f"## Invalid JSON - line {line_number}\n\n")
            outfile.write("```text\n")
            outfile.write(str(e))
            outfile.write("\n```\n")
            continue

        # ----------------------------------------------------
        # METADATA
        # ----------------------------------------------------

        iteration = entry.get("iteration", line_number)
        entry_type = entry.get("type", "unknown")
        timestamp = entry.get("timestamp", "")
        iteration_time = entry.get("iteration_time", "")

        # ----------------------------------------------------
        # COLLAPSIBLE ITERATION
        # ----------------------------------------------------

        outfile.write("\n<details>\n\n")

        outfile.write(
            f"<summary><b>Iteration {iteration}</b>"
            f" | Type: {entry_type}"
            f" | Time: {iteration_time}</summary>\n\n"
        )

        if timestamp:
            outfile.write(f"**Timestamp:** `{timestamp}`\n\n")

        # ----------------------------------------------------
        # PROMPT
        # ----------------------------------------------------

        write_section(
            outfile,
            "Prompt",
            entry.get("prompt"),
            "text"
        )

        # ----------------------------------------------------
        # RAW MODEL RESPONSE
        # ----------------------------------------------------

        write_section(
            outfile,
            "Raw Model Response",
            entry.get("response"),
            "text"
        )

        # ----------------------------------------------------
        # EXTRACTED PYTHON CODE
        # ----------------------------------------------------

        code_blocks = entry.get("code_blocks")

        if code_blocks:
            outfile.write("\n### Extracted Python Code\n\n")

            if isinstance(code_blocks, list):

                for i, code in enumerate(code_blocks, start=1):

                    outfile.write(f"#### Code Block {i}\n\n")
                    outfile.write("```python\n")

                    if isinstance(code, str):
                        outfile.write(code)
                    else:
                        outfile.write(
                            json.dumps(
                                code,
                                indent=2,
                                ensure_ascii=False,
                                default=str
                            )
                        )

                    outfile.write("\n```\n")

            else:
                outfile.write("```python\n")
                outfile.write(safe_text(code_blocks))
                outfile.write("\n```\n")

        # ----------------------------------------------------
        # FINAL ANSWER
        # ----------------------------------------------------

        write_section(
            outfile,
            "Final Answer",
            entry.get("final_answer"),
            "text"
        )

        # Close collapsible iteration
        outfile.write("\n</details>\n\n")


print("=" * 60)
print("Done!")
print(f"Readable debug trace saved to:")
print(output_path.resolve())
print("=" * 60)