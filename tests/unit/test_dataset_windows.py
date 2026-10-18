import json

import pytest

from model.module.dataset import MultiWindowBIODataset, _word_windows


def test_word_windows_single_window_when_it_fits():
    assert _word_windows([2, 2, 2], budget=10, overlap=2) == [(0, 3)]


def test_word_windows_cover_every_word_and_respect_budget():
    lengths = [3] * 20  # 60 subwords
    windows = _word_windows(lengths, budget=20, overlap=6)
    assert windows[0][0] == 0
    assert windows[-1][1] == len(lengths)
    for start, end in windows:
        assert sum(lengths[start:end]) <= 20
    for (_, prev_end), (next_start, _) in zip(windows, windows[1:]):
        assert next_start <= prev_end  # consecutive windows overlap or touch


def test_word_windows_keep_oversized_word_alone():
    assert _word_windows([50, 2], budget=10, overlap=0) == [(0, 1), (1, 2)]


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            "aisingapore/SEA-LION-ModernBERT-300M", local_files_only=True
        )
    except Exception:
        pytest.skip("SEA-LION tokenizer is not in the local Hugging Face cache")


def _write_chunks(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def test_long_record_is_split_into_windows_within_max_length(tmp_path, tokenizer):
    tokens = ["Câu", "hỏi", "số", "631", "A.", "This", "chance"] * 400
    tags = ["B-QUESTION_LABEL", "I-QUESTION_LABEL"] + ["O"] * (len(tokens) - 2)
    path = tmp_path / "chunks.jsonl"
    _write_chunks(path, [{"tokens": tokens, "ner_tags": tags}])

    tag_to_id = {"O": 0, "B-QUESTION_LABEL": 1, "I-QUESTION_LABEL": 2}
    ds = MultiWindowBIODataset(path, tag_to_id, tokenizer=tokenizer, max_length=512, window_overlap_tokens=128)

    assert len(ds) > 1
    for i in range(len(ds)):
        item = ds[i]
        assert item["input_ids"].shape[0] <= 512
        assert item["labels"].shape == item["input_ids"].shape
    # The first window starts at the start of the record
    assert int(ds.index[0][1]) == 0
    assert int(ds.index[-1][2]) == len(tokens)


def test_short_record_is_one_window_identical_to_direct_tokenization(tmp_path, tokenizer):
    tokens = ["Câu", "1", "Nguyễn", "không", "được"]
    tags = ["B-QUESTION_LABEL", "I-QUESTION_LABEL", "O", "O", "O"]
    path = tmp_path / "chunks.jsonl"
    _write_chunks(path, [{"tokens": tokens, "ner_tags": tags}])

    ds = MultiWindowBIODataset(path, {"O": 0, "B-QUESTION_LABEL": 1, "I-QUESTION_LABEL": 2}, tokenizer=tokenizer, max_length=8192)
    assert len(ds) == 1
    expected = tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=8192)["input_ids"]
    assert ds[0]["input_ids"].tolist() == expected
