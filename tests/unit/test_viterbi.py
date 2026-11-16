import pytest
import numpy as np
import torch
from model.inference.viterbi import build_bio_transition_matrix, ConstrainedViterbiDecoder
from model.inference.predict import (
    generate_hann_window,
    compute_entropy_confidence_weights,
    _unify_contiguous_discourse_spans,
)


def test_transition_matrix_defaults():
    id_to_tag = {
        0: "O",
        1: "B-question_label",
        2: "I-question_label",
        3: "B-stem",
        4: "I-stem"
    }
    M, start_trans = build_bio_transition_matrix(id_to_tag)

    # Soft impossible penalty default is -10.0
    assert start_trans[2] == -10.0  # I-question_label cannot start sequence
    assert start_trans[4] == -10.0  # I-stem cannot start sequence
    assert start_trans[0] == 0.0    # O can start
    assert start_trans[1] == 0.0    # B-question_label can start

    # O -> I-stem is invalid
    assert M[0, 4] == -10.0
    # B-stem -> I-question_label is invalid
    assert M[3, 2] == -10.0
    # B-stem -> I-stem is valid
    assert M[3, 4] == 0.0
    # I-stem -> I-stem is valid
    assert M[4, 4] == 0.0
    # B-question_label -> B-stem is valid
    assert M[1, 3] == 0.0


def test_viterbi_decoder_initialization_flexibility():
    tags = ["O", "B-stem", "I-stem"]
    # Via bio_labels list
    dec1 = ConstrainedViterbiDecoder(bio_labels=tags, soft_impossible_penalty=-10.0)
    assert dec1.K == 3
    assert dec1.penalty == -10.0

    # Via id_to_tag dict
    id_to_tag = {0: "O", 1: "B-stem", 2: "I-stem"}
    dec2 = ConstrainedViterbiDecoder(id_to_tag=id_to_tag)
    assert dec2.K == 3
    assert dec2.penalty == -10.0


def test_c_and_numpy_backend_parity():
    id_to_tag = {
        0: "O",
        1: "B-question_label",
        2: "I-question_label",
        3: "B-stem",
        4: "I-stem"
    }
    decoder = ConstrainedViterbiDecoder(id_to_tag=id_to_tag, soft_impossible_penalty=-10.0)
    np.random.seed(42)
    T, K = 128, len(id_to_tag)
    logits = np.random.randn(T, K).astype(np.float32)

    path_c = decoder.decode(logits)
    path_np = decoder._decode_numpy(logits, T, K)

    assert path_c == path_np, "C and NumPy backends must yield identical decoding paths"


def test_soft_penalty_override():
    id_to_tag = {0: "O", 1: "B-stem", 2: "I-stem"}
    decoder = ConstrainedViterbiDecoder(id_to_tag=id_to_tag, soft_impossible_penalty=-10.0)

    # Construct logits where token 0 is strongly 'O'
    # token 1 has 'I-stem' emission overwhelmingly high (+50.0 vs 0.0)
    logits = np.array([
        [10.0, 0.0, -10.0],  # Token 0 -> predicts O
        [-20.0, -20.0, 50.0]  # Token 1 -> I-stem has +70 confidence over O
    ], dtype=np.float32)

    path = decoder.decode(logits)
    # Token 1 overrides the -10.0 transition penalty because +50 > -10 penalty
    assert path[0] == 0  # O
    assert path[1] == 2  # I-stem (soft override allowed)


def test_generate_hann_window():
    chunk_len = 16
    window = generate_hann_window(chunk_len)
    assert window.shape == (16,)
    assert window.min().item() >= 0.1
    assert window.max().item() <= 1.0

    # Symmetry check
    np.testing.assert_allclose(window.numpy(), window.numpy()[::-1], atol=1e-5)
    # Center tokens are higher than edge tokens
    assert window[8].item() > window[0].item()
    assert window[0].item() == pytest.approx(0.1, abs=0.05)


def test_entropy_confidence_weights_penalizes_blind_window():
    # Token 0: high-entropy uniform logits (blind mid-passage window)
    # Token 1: low-entropy sharp logits (anchor window seeing structural boundary)
    logits = torch.tensor([
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [-5.0, -5.0, -5.0, -5.0, 12.0],
    ], dtype=torch.float32)
    log_probs, conf_w = compute_entropy_confidence_weights(logits)
    assert log_probs.shape == (2, 5)
    assert conf_w.shape == (2,)
    assert conf_w[0].item() == pytest.approx(0.05, abs=1e-4)
    assert conf_w[1].item() > 0.98
    assert conf_w[1].item() / conf_w[0].item() > 19.0


def test_hard_viterbi_global_backtracking_across_passage():
    id_to_tag = {
        0: "O",
        1: "B-question_label",
        2: "I-question_label",
        3: "B-stem",
        4: "I-stem",
        5: "B-stimulus",
        6: "I-stimulus",
    }
    T = 100
    logits = np.full((T, 7), -15.0, dtype=np.float32)
    # Token 0: passage start (could be B-stem or B-stimulus)
    logits[0, 3] = -0.5  # B-stem
    logits[0, 5] = -0.6  # B-stimulus
    # Tokens 1..69: blind window slightly favors I-stem (-0.6) over I-stimulus (-0.8)
    logits[1:70, 4] = -0.6
    logits[1:70, 6] = -0.8
    # Tokens 70..99: anchor window seeing downstream questions strongly favors I-stimulus (-0.05 vs -4.0)
    logits[70:100, 4] = -4.0
    logits[70:100, 6] = -0.05

    hard_decoder = ConstrainedViterbiDecoder(
        id_to_tag=id_to_tag,
        use_soft_constraints=False,
        enforce_domain_constraints=True,
    )
    path = hard_decoder.decode(logits)
    # Backtracking propagates stimulus back to token 0 across the entire 100-token passage
    assert path[0] == 5  # B-stimulus
    assert all(p == 6 for p in path[1:])  # All I-stimulus

    # Also verify allow_leading_i=True permits chunk evaluation starting at I-stimulus
    chunk_decoder = ConstrainedViterbiDecoder(id_to_tag=id_to_tag, allow_leading_i=True)
    assert chunk_decoder.start_trans[6] == 0.0


def test_unify_contiguous_discourse_spans_multi_paragraph_and_optionless_q():
    # 1. Multi-paragraph passage before 2 questions split into <stem> + <stimulus> across \n\n
    text1 = "Paragraph 1 of IELTS.\n\nParagraph 2 of IELTS.\n\nCâu 1. Ask 1?\nCâu 2. Ask 2?"
    p1_end = text1.index("\n\nParagraph 2")
    p2_start = p1_end + 2
    p2_end = text1.index("\n\nCâu 1.")
    q1_start = p2_end + 2
    q1_end = q1_start + len("Câu 1.")
    q2_start = text1.index("Câu 2.")
    q2_end = q2_start + len("Câu 2.")
    segs1 = [
        {"label": "stem", "start": 0, "end": p1_end, "text": text1[:p1_end]},
        {"label": "stimulus", "start": p2_start, "end": p2_end, "text": text1[p2_start:p2_end]},
        {"label": "question_label", "start": q1_start, "end": q1_end, "text": "Câu 1."},
        {"label": "question_label", "start": q2_start, "end": q2_end, "text": "Câu 2."},
    ]
    unified1 = _unify_contiguous_discourse_spans(text1, segs1)
    assert len(unified1) == 3
    assert unified1[0]["label"] == "stimulus"
    assert unified1[0]["start"] == 0 and unified1[0]["end"] == p2_end

    # 2. Optionless short-answer question (<question_label> + <stem>) followed by a new <stimulus>
    text2 = "Câu 5. Điền số thích hợp: _____\n\nDựa vào bài đọc trả lời câu 6, 7:\n\nCâu 6. Hỏi 6?\nCâu 7. Hỏi 7?"
    q5_end = len("Câu 5.")
    st5_start = q5_end + 1
    st5_end = text2.index("\n\nDựa vào")
    stim_start = st5_end + 2
    stim_end = text2.index("\n\nCâu 6.")
    q6_start = stim_end + 2
    q7_start = text2.index("Câu 7.")
    segs2 = [
        {"label": "question_label", "start": 0, "end": q5_end, "text": "Câu 5."},
        {"label": "stem", "start": st5_start, "end": st5_end, "text": text2[st5_start:st5_end]},
        {"label": "stimulus", "start": stim_start, "end": stim_end, "text": text2[stim_start:stim_end]},
        {"label": "question_label", "start": q6_start, "end": q6_start + 6, "text": "Câu 6."},
        {"label": "question_label", "start": q7_start, "end": q7_start + 6, "text": "Câu 7."},
    ]
    unified2 = _unify_contiguous_discourse_spans(text2, segs2)
    assert len(unified2) == 5
    assert unified2[1]["label"] == "stem"
    assert unified2[2]["label"] == "stimulus"
