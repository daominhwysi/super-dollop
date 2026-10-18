import pytest
import numpy as np
import torch
from model.inference.viterbi import build_bio_transition_matrix, ConstrainedViterbiDecoder
from model.inference.predict import generate_hann_window


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
