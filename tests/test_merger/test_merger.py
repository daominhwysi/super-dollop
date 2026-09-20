import re
from pathlib import Path
from typing import Tuple

from sequence_labelling.parser.long_parser.sequence_reconciler import merge_chunk_xmls


def _load_fixture(*parts: str) -> str:
    fixture_root = Path(__file__).resolve().parent / "fixtures"
    return (fixture_root.joinpath(*parts)).read_text(encoding="utf-8").strip()


def _load_chunk_fixture(*parts: str) -> Tuple[str, str]:
    content = _load_fixture(*parts)
    input_match = re.search(r"<input>(.*?)</input>", content, re.DOTALL)
    parsed_match = re.search(r"<parsed>(.*?)</parsed>", content, re.DOTALL)
    raw = input_match.group(1).strip() if input_match else ""
    parsed = parsed_match.group(1).strip() if parsed_match else content.strip()
    return raw, parsed


def _write_merged_output(*parts: str, merged_xml: str) -> None:
    output_root = Path(__file__).resolve().parent / "outputs"
    output_dir = output_root.joinpath(*parts)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "merged_output.xml").write_text(merged_xml, encoding="utf-8")


def test_load_all_fixtures_and_write_outputs():
    """
    Iterates through all fixture directories in tests/test_merger/fixtures/,
    loads chunks (<input> and <parsed>), runs the merger, and writes merged output
    to tests/test_merger/outputs/<fixture_name>/merged_output.xml.
    """
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    output_root = Path(__file__).resolve().parent / "outputs"

    fixture_dirs = sorted([d for d in fixtures_dir.iterdir() if d.is_dir()])
    assert len(fixture_dirs) > 0, "No fixture directories found"

    for f_dir in fixture_dirs:
        fixture_name = f_dir.name
        chunk_files = sorted(list(f_dir.glob("chunk_*.xml")))
        if not chunk_files:
            continue

        raw_inputs = []
        parsed_xmls = []

        for chunk_file in chunk_files:
            raw_text, parsed_xml = _load_chunk_fixture(fixture_name, chunk_file.name)
            raw_inputs.append(raw_text)
            parsed_xmls.append(parsed_xml)

        result = merge_chunk_xmls(parsed_xmls, raw_chunk_inputs=raw_inputs)
        merged_xml = result["merged_xml"]

        # Write output to outputs/<fixture_name>/merged_output.xml
        out_dir = output_root / fixture_name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "merged_output.xml").write_text(merged_xml, encoding="utf-8")

        # Verify against expected_output.xml if present
        expected_file = f_dir / "expected_output.xml"
        if expected_file.exists():
            expected_xml = expected_file.read_text(encoding="utf-8").strip()
            # Verify detagged source fidelity or exact XML match
            assert merged_xml.strip() == expected_xml or re.sub(r"\s+", "", merged_xml) == re.sub(r"\s+", "", expected_xml)


def test_merge_chunk_xmls_simulates_real_time_overlap_window():
    raw_a, parsed_a = _load_chunk_fixture("simulates_real_time_overlap_window", "chunk_a.xml")
    raw_b, parsed_b = _load_chunk_fixture("simulates_real_time_overlap_window", "chunk_b.xml")

    result = merge_chunk_xmls([parsed_a, parsed_b], raw_chunk_inputs=[raw_a, raw_b])
    merged = result["merged_xml"]

    assert result["total_questions"] == 16
    assert merged.count("<section>Live Stream</section>") == 1 or merged.count("<section>Live Stream</section>") == 2
    assert merged.count("<stimulus>Shared passage appears in both chunks.</stimulus>") == 1
    for q in range(1, 17):
        assert merged.count(f"<question_label>**{q}.**</question_label>") == 1


def test_merge_chunk_xmls_skips_hallucinated_question_when_raw_input_missing():
    chunk_1 = """
<question_label>**1.**</question_label>
<stem>First real question.</stem>
<question_label>**2.**</question_label>
<stem>Second real question.</stem>
""".strip()

    chunk_2 = """
<question_label>**2.**</question_label>
<stem>Duplicate in overlap.</stem>
<question_label>**3.**</question_label>
<stem>Hallucinated question should be removed.</stem>
""".strip()

    raw_inputs = [
        "**1.**\nFirst real question.\n**2.**\nSecond real question.",
        "**2.**\nSecond real question.",
    ]

    result = merge_chunk_xmls([chunk_1, chunk_2], raw_chunk_inputs=raw_inputs)
    merged = result["merged_xml"]

    assert result["total_questions"] == 2
    assert merged.count("<question_label>**1.**</question_label>") == 1
    assert merged.count("<question_label>**2.**</question_label>") == 1
    assert "<question_label>**3.**</question_label>" not in merged


def test_merge_chunk_xmls_real_vietnamese_standardized_exam_overlap_and_overlap_stimulus():
    raw_a, parsed_a = _load_chunk_fixture("vietnamese_math_overlap", "chunk_a.xml")
    raw_b, parsed_b = _load_chunk_fixture("vietnamese_math_overlap", "chunk_b.xml")

    result = merge_chunk_xmls([parsed_a, parsed_b], raw_chunk_inputs=[raw_a, raw_b])
    _write_merged_output("vietnamese_math_overlap", merged_xml=result["merged_xml"])

    assert result["total_questions"] == 13
    assert (
        result["merged_xml"].count(
            "<stimulus>BÀI ĐỌC HIỂU: Một học sinh chuẩn bị đi thi, cần xác định quãng đường đi bộ khi đi từ nhà đến trường trong giờ cao điểm.</stimulus>"
        )
        == 1
    )

    for q in range(1, 14):
        assert result["merged_xml"].count(f"<question_label>**{q}.**</question_label>") == 1


def test_merge_chunk_xmls_real_vietnamese_standardized_exam_prunes_hallucinated_tail():
    raw_a, parsed_a = _load_chunk_fixture("vietnamese_history_hallucinated", "chunk_a.xml")
    raw_b, parsed_b = _load_chunk_fixture("vietnamese_history_hallucinated", "chunk_b.xml")

    result = merge_chunk_xmls([parsed_a, parsed_b], raw_chunk_inputs=[raw_a, raw_b])
    merged = result["merged_xml"]
    _write_merged_output(
        "vietnamese_history_hallucinated",
        merged_xml=merged,
    )

    assert result["total_questions"] == 5
    assert merged.count("<question_label>**3.**") == 1
    assert merged.count("<question_label>**4.**") == 1
    assert merged.count("<question_label>**5.**") == 1
    assert "<question_label>**6.**" not in merged
