import re
import json
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
LLM_LOGS_DIR = WORKSPACE_DIR / "backend" / "logs" / "llm_logs"


def convert_log_file(file_path: Path):
    content = file_path.read_text(encoding="utf-8")

    # Extract metadata header
    header_match = re.search(
        r"(# 🤖 LLM Request Log:.*?\n- \*\*Timestamp:\*\*.*?\n- \*\*Model:\*\*.*?\n(?:- \*\*Provider:\*\*.*?\n)?)",
        content,
    )
    header_str = header_match.group(1).strip() if header_match else "# 🤖 LLM Request Log"

    # Extract Input section
    input_match = re.search(r"## 📥 Input\s*```json\s*(.*?)\s*```", content, flags=re.DOTALL)
    input_json_str = input_match.group(1) if input_match else ""

    # Extract Output section
    output_match = re.search(r"## 📤 Output\s*```markdown\s*(.*?)\s*```", content, flags=re.DOTALL)
    output_str = output_match.group(1) if output_match else ""

    # Extract Reasoning section
    reasoning_count_match = re.search(r"- \*\*Reasoning Tokens Count:\*\*\s*`(\d+)`", content)
    r_count = reasoning_count_match.group(1) if reasoning_count_match else "0"

    r_summary_match = re.search(r"- \*\*Reasoning Summary:\*\*\s*```markdown\s*(.*?)\s*```", content, flags=re.DOTALL)
    r_summary_str = r_summary_match.group(1) if r_summary_match else ""

    r_content_match = re.search(r"- \*\*Full Reasoning Content:\*\*\s*```markdown\s*(.*?)\s*```|- \*\*Reasoning Content:\*\*\s*```markdown\s*(.*?)\s*```", content, flags=re.DOTALL)
    if r_content_match:
        r_content_str = r_content_match.group(1) or r_content_match.group(2) or ""
    else:
        r_content_str = ""

    # Extract Stats section
    stats_match = re.search(r"## 📊 Stats\s*(.*)", content, flags=re.DOTALL)
    stats_str = stats_match.group(1).strip() if stats_match else ""

    # Parse messages
    messages = []
    if input_json_str:
        try:
            messages = json.loads(input_json_str)
        except Exception:
            messages = [{"role": "user", "content": input_json_str}]

    # Build new Markdown
    md = []
    md.append(header_str)
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 📥 Input")
    md.append("")

    if isinstance(messages, list) and messages:
        for idx, msg in enumerate(messages):
            if isinstance(msg, dict):
                role = str(msg.get("role") or f"message_{idx+1}").capitalize()
                msg_content = str(msg.get("content") or "")
            else:
                role = f"Message {idx+1}"
                msg_content = str(msg)

            md.append("<details>")
            md.append(f"<summary>Role: {role}</summary>")
            md.append("")
            md.append("```markdown")
            md.append(msg_content)
            md.append("```")
            md.append("</details>")
            md.append("")
    else:
        md.append("<details>")
        md.append("<summary>Role: Input</summary>")
        md.append("")
        md.append("```markdown")
        md.append(input_json_str)
        md.append("```")
        md.append("</details>")
        md.append("")

    md.append("---")
    md.append("")
    md.append("## 📤 Output")
    md.append("")
    md.append("<details>")
    md.append("<summary>Completion Output</summary>")
    md.append("")
    md.append("```markdown")
    md.append(output_str if output_str else "*(No completion output emitted)*")
    md.append("```")
    md.append("</details>")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 🧠 Reasoning & Summary")
    md.append(f"- **Reasoning Tokens Count:** `{r_count}`")
    md.append("")
    if r_summary_str:
        md.append("<details>")
        md.append("<summary>Reasoning Summary</summary>")
        md.append("")
        md.append("```markdown")
        md.append(r_summary_str)
        md.append("```")
        md.append("</details>")
        md.append("")
    if r_content_str:
        md.append("<details>")
        md.append("<summary>Full Reasoning Content</summary>")
        md.append("")
        md.append("```markdown")
        md.append(r_content_str)
        md.append("```")
        md.append("</details>")
        md.append("")
    if not r_summary_str and not r_content_str:
        md.append("*(No separate reasoning summary or content emitted)*")
        md.append("")

    md.append("---")
    md.append("")
    md.append("## 📊 Stats")
    md.append(stats_str)
    md.append("")

    file_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Re-formatted: {file_path.name}")


def main():
    md_files = sorted(list(LLM_LOGS_DIR.rglob("*.md")))
    print(f"Converting {len(md_files)} log files to <details><summary> markdown format...")
    for f in md_files:
        convert_log_file(f)
    print("All log files re-formatted successfully!")


if __name__ == "__main__":
    main()
