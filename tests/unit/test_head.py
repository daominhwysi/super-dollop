import pytest
import torch
import torch.nn as nn
from model.module.head import (
    FocalLoss,
    SigmoidWeightedLayerPooling,
    WeightedLayerPooling,
    LocalConvRefinement,
    TightenedMultiSampleDropout,
    MultiSampleDropout,
    EnhancedTokenClassificationHead,
    SequenceLabellingHead,
    EnhancedBertForTokenClassification
)


def test_sigmoid_weighted_layer_pooling():
    B, S, H = 2, 16, 64
    pooler = SigmoidWeightedLayerPooling(num_layers=4)
    assert pooler.weights.shape == (4,)
    assert torch.all(pooler.weights == 0.0)

    # Equal weighting at step 0
    layers = [torch.randn(B, S, H) for _ in range(4)]
    out = pooler(layers)
    assert out.shape == (B, S, H)

    # Check normalization: sigmoid(0) = 0.5, 4 * 0.5 = 2.0 -> each weight is 0.25
    expected = sum(layers) / 4.0
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_local_conv_refinement():
    B, S, H = 2, 32, 64
    conv = LocalConvRefinement(hidden_size=H, kernel_size=7, dropout_prob=0.0)
    x = torch.randn(B, S, H)
    out = conv(x)
    assert out.shape == (B, S, H)

    # Check Dirac identity on depthwise conv
    center_idx = 7 // 2
    for i in range(H):
        assert conv.depthwise.weight.data[i, 0, center_idx].item() == 1.0
        # other indices are 0
        for k in range(7):
            if k != center_idx:
                assert conv.depthwise.weight.data[i, 0, k].item() == 0.0


def test_tightened_multi_sample_dropout():
    B, S, H, num_labels = 2, 16, 64, 15
    msd = TightenedMultiSampleDropout(dropout_rates=(0.05, 0.10, 0.15, 0.20, 0.25))
    classifier = nn.Linear(H, num_labels)
    x = torch.randn(B, S, H)

    # In train mode
    msd.train()
    branch_logits = msd(x, classifier)
    assert len(branch_logits) == 5
    for l in branch_logits:
        assert l.shape == (B, S, num_labels)

    # In eval mode
    msd.eval()
    eval_logits = msd(x, classifier)
    assert len(eval_logits) == 1
    assert eval_logits[0].shape == (B, S, num_labels)


def test_enhanced_token_classification_head():
    B, S, H, num_labels = 2, 20, 64, 15
    head = EnhancedTokenClassificationHead(
        hidden_size=H,
        num_labels=num_labels,
        num_layers_to_fuse=4,
        dropout_rates=(0.05, 0.10, 0.15, 0.20, 0.25),
        use_conv_refinement=True,
        conv_kernel_size=7
    )
    layers = [torch.randn(B, S, H) for _ in range(4)]
    targets = torch.randint(0, num_labels, (B, S))
    loss_fct = FocalLoss(gamma=2.0)

    # 1. Training mode with targets -> Per-branch focal loss
    head.train()
    logits, loss = head(layers, targets=targets, loss_fct=loss_fct)
    assert logits.shape == (B, S, num_labels)
    assert loss is not None
    assert loss.item() > 0

    # 2. Eval mode
    head.eval()
    eval_logits = head(layers)
    assert isinstance(eval_logits, torch.Tensor)
    assert eval_logits.shape == (B, S, num_labels)


def test_enhanced_bert_for_token_classification_forward():
    class DummyConfig:
        hidden_size = 64
        num_labels = 15
        output_hidden_states = True

    class DummyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(100, 64)

        def forward(self, input_ids, **kwargs):
            h = self.emb(input_ids)
            # return hidden states as list
            return type("Output", (), {"hidden_states": [h, h, h, h], "attentions": None})()

    config = DummyConfig()
    backbone = DummyBackbone()
    model = EnhancedBertForTokenClassification(
        config=config,
        base_model=backbone,
        num_layers_to_fuse=4,
        focal_gamma=2.0,
        per_branch_focal_loss=True
    )

    input_ids = torch.randint(0, 100, (2, 16))
    labels = torch.randint(0, 15, (2, 16))

    # Train mode
    model.train()
    out = model(input_ids=input_ids, labels=labels)
    assert out.loss is not None
    assert out.logits.shape == (2, 16, 15)

    # Eval mode
    model.eval()
    out_eval = model(input_ids=input_ids)
    assert out_eval.loss is None
    assert out_eval.logits.shape == (2, 16, 15)
