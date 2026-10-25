import torch
from transformers import ModernBertConfig, ModernBertModel

import model.train as train_module
from model.module.head import EnhancedBertForTokenClassification

TAG_TO_ID = {"O": 0, "B-QUESTION_LABEL": 1, "I-QUESTION_LABEL": 2}
ID_TO_TAG = {v: k for k, v in TAG_TO_ID.items()}


def _tiny_model():
    cfg = ModernBertConfig(
        vocab_size=200,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        cls_token_id=1,
        sep_token_id=2,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=8192,
        output_hidden_states=True,
    )
    cfg.num_labels = len(TAG_TO_ID)
    cfg.id2label = ID_TO_TAG
    cfg.label2id = TAG_TO_ID
    return EnhancedBertForTokenClassification(config=cfg, base_model=ModernBertModel(cfg))


def test_no_decay_keywords_match_modernbert_norm_names():
    model = _tiny_model()
    excluded = [
        name for name, _ in model.base_model.named_parameters()
        if any(key in name for key in train_module.NO_DECAY_KEYWORDS)
    ]
    assert excluded, "ModernBERT norm parameters must be matched for no weight decay"
    assert all("norm" in name or "bias" in name for name in excluded)
    assert any(name.endswith("mlp_norm.weight") for name in excluded)


def test_evaluate_split_returns_decoded_sequences_aligned_to_gold_labels():
    model = _tiny_model()
    input_ids = torch.randint(3, 200, (2, 12))
    labels = torch.tensor([
        [1, 2, -100, 0, 0, -100, 0, 1, 2, 0, -100, -100],
        [0, 1, 2, 2, 0, -100, -100, -100, -100, -100, -100, -100],
    ])
    loader = [{"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "labels": labels}]

    loss, preds, gold = train_module.evaluate_split(
        model, loader, torch.device("cpu"), False, torch.float32, ID_TO_TAG
    )

    assert loss > 0
    assert [len(p) for p in preds] == [len(g) for g in gold] == [8, 5]
    assert all(tag in TAG_TO_ID for seq in preds for tag in seq)
    assert gold[0] == ["B-QUESTION_LABEL", "I-QUESTION_LABEL", "O", "O", "O", "B-QUESTION_LABEL", "I-QUESTION_LABEL", "O"]


def test_distributed_batch_sampler_padding_and_step_balance():
    lengths = [100, 200, 300, 400, 500, 600, 700]  # 7 items, batch_size=2 -> 4 batches total
    num_replicas = 3
    # 4 batches % 3 replicas != 0 -> needs wrap-around padding so all ranks get 2 batches
    samplers = [
        train_module.DistributedLengthGroupedBatchSampler(
            lengths=lengths,
            batch_size=2,
            num_replicas=num_replicas,
            rank=r,
            drop_last=False,
            shuffle=False,
            seed=42,
        )
        for r in range(num_replicas)
    ]
    rank_batches = [list(s) for s in samplers]
    # Every rank must yield exactly the same number of batches to prevent DDP deadlock
    assert len(rank_batches[0]) == len(rank_batches[1]) == len(rank_batches[2]) == 2
    for s in samplers:
        assert len(s) == 2

