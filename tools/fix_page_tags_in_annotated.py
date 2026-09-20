import re
import sys
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_DIR / "data" / "sequence_labelling_annotated"

TARGET_TAGS = ["stem", "option_text", "explanation", "section"]

def prune_and_concat(content: str) -> str:
    # 1. Prune page metadata blocks
    content = re.sub(r'</?page_metadata>\s*\{[^{}]*\}\s*(?:</page_metadata>)?', '', content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r'</?page_metadata>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'<\|page_metadata\|>\s*\{[^{}]*\}\s*<\|end_metadata\|>', '', content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r'<\|page_metadata\|>|<\|end_metadata\|>', '', content, flags=re.IGNORECASE)

    # 2. Prune <pages>, </pages>, <page>, </page>
    content = re.sub(r'</?pages>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'</?page>', '', content, flags=re.IGNORECASE)

    # 3. Concat continuous tags previously separated by page tags
    for tag in TARGET_TAGS:
        pattern = re.compile(rf'</{tag}>(\s*)<{tag}>', flags=re.IGNORECASE)
        while pattern.search(content):
            def repl(m):
                ws = m.group(1)
                if '\n' in ws:
                    newlines = ws.count('\n')
                    if newlines >= 2:
                        ws = '\n\n'
                return ws
            content = pattern.sub(repl, content)

    # 4. Clean up any trailing excessive empty lines (> 2 newlines)
    content = re.sub(r'\n{3,}', '\n\n', content)
    if not content.endswith('\n'):
        content += '\n'

    return content


def main():
    xml_files = sorted(DATA_DIR.rglob("*.xml"))
    print(f"Discovered {len(xml_files)} XML files in {DATA_DIR}")

    updated_count = 0
    figure_checks_passed = 0

    for xml_path in xml_files:
        orig_text = xml_path.read_text(encoding="utf-8")
        orig_figs = len(re.findall(r'<figure\s+id=', orig_text))
        
        cleaned_text = prune_and_concat(orig_text)
        
        clean_figs = len(re.findall(r'<figure\s+id=', cleaned_text))
        assert orig_figs == clean_figs, f"Figure mismatch in {xml_path}: {orig_figs} vs {clean_figs}"
        figure_checks_passed += 1

        if cleaned_text != orig_text:
            xml_path.write_text(cleaned_text, encoding="utf-8")
            updated_count += 1

    print(f"Successfully processed {len(xml_files)} XML files.")
    print(f"Updated {updated_count} files.")
    print(f"Verified all figure counts for {figure_checks_passed} files.")


if __name__ == "__main__":
    main()
