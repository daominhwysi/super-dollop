#!/usr/bin/env python3
"""
Enhanced Token Classification Head with:
1. Weighted Layer Pooling (Fusing representations from top-K transformer layers)
2. Dense Projection with GELU non-linearity and LayerNorm
3. Multi-Sample Dropout (MSD) for accelerated convergence and regularization
4. Focal Loss with optional Label Smoothing for mitigating class imbalance
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Dict, Any, List

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from transformers.modeling_outputs import TokenClassifierOutput
from transformers import AutoModel, AutoConfig, PreTrainedModel
from transformers.utils import logging as transformers_logging

transformers_logging.disable_progress_bar()
try:
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
except ImportError:
    pass


class FocalLoss(nn.Module):
    """
    Multiclass Focal Loss for addressing class imbalance by dynamically down-weighting
    easy, well-classified examples (e.g. background 'O' tokens) and focusing on hard boundary tokens.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [batch_size * seq_len, num_classes] or [batch_size, seq_len, num_classes]
        targets: [batch_size * seq_len] or [batch_size, seq_len]
        """
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = targets.view(-1)

        # Mask ignored tokens (e.g. padding, subwords, special tokens with -100)
        valid_mask = flat_targets != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        valid_logits = flat_logits[valid_mask]
        valid_targets = flat_targets[valid_mask]

        # Compute log probabilities and probabilities directly for numerical stability
        log_probs = F.log_softmax(valid_logits, dim=-1)
        probs = torch.exp(log_probs)

        # True class log_prob and prob
        log_p_t = log_probs.gather(-1, valid_targets.unsqueeze(-1)).squeeze(-1)
        p_t = probs.gather(-1, valid_targets.unsqueeze(-1)).squeeze(-1)

        # Focal modulation factor: (1 - p_t)^gamma
        focal_weight = ((1.0 - p_t) ** self.gamma)

        if self.label_smoothing > 0.0:
            ce_loss = F.cross_entropy(
                valid_logits,
                valid_targets,
                weight=None,
                reduction="none",
                label_smoothing=self.label_smoothing
            )
            base_loss = ce_loss
        else:
            base_loss = -log_p_t

        focal_loss = focal_weight * base_loss

        # Apply class weights if provided (without corrupting p_t exponentiation)
        if self.weight is not None:
            class_weights = self.weight[valid_targets]
            focal_loss = class_weights * focal_loss

        return focal_loss.mean()


class SigmoidWeightedLayerPooling(nn.Module):
    """
    Non-competing layer pooling. Uses independent sigmoid gates normalized after activation,
    preventing early layer collapse caused by softmax zero-sum competition.
    """
    def __init__(self, num_layers: int = 4):
        super().__init__()
        self.num_layers = max(1, num_layers)
        # Initialize weights to 0.0 -> sigmoid(0.0) = 0.5 (equal weighting)
        self.weights = nn.Parameter(torch.zeros(self.num_layers))

    def forward(
        self,
        hidden_states: Union[Tuple[torch.Tensor, ...], List[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(hidden_states, (tuple, list)):
            k = min(self.num_layers, len(hidden_states))
            selected_layers = hidden_states[-k:]
            gates = torch.sigmoid(self.weights[:k])
            norm_gates = gates / (torch.sum(gates) + 1e-8)
            stacked = torch.stack(selected_layers, dim=0)  # Shape: [k, B, T, H]
            fused = torch.einsum('l,lbth->bth', norm_gates, stacked)
            return fused
        return hidden_states


# Backward-compatible alias
WeightedLayerPooling = SigmoidWeightedLayerPooling


class LocalConvRefinement(nn.Module):
    """
    1D Depthwise Separable Convolution for local boundary refinement.
    - Kernel size: 7 tokens (~3-4 Vietnamese words receptive field)
    - Depthwise (groups = hidden_size) + Pointwise (1x1)
    - Dirac initialization ensures identity mapping at step 0.
    """
    def __init__(self, hidden_size: int = 768, kernel_size: int = 7, dropout_prob: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd for symmetric padding"
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # Depthwise Conv (per-channel spatial pattern)
        self.depthwise = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=padding,
            groups=hidden_size,
            bias=False
        )
        # Pointwise Conv (cross-channel projection)
        self.pointwise = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=1,
            bias=True
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_prob)

        self._init_dirac_identity(hidden_size, kernel_size)

    def _init_dirac_identity(self, hidden_size: int, kernel_size: int):
        # Center impulse for depthwise kernel
        nn.init.zeros_(self.depthwise.weight)
        center_idx = kernel_size // 2
        for i in range(hidden_size):
            self.depthwise.weight.data[i, 0, center_idx] = 1.0

        # Identity for pointwise projection
        nn.init.eye_(self.pointwise.weight.data.view(hidden_size, hidden_size))
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, S, H]
        residual = x
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).float()

        x_trans = x.transpose(1, 2)  # [B, H, S]
        x_conv = self.depthwise(x_trans)
        x_conv = self.pointwise(x_conv)
        x_conv = x_conv.transpose(1, 2)  # [B, S, H]

        out = self.act(x_conv)
        out = self.dropout(out)
        out = self.norm(residual + out)
        return out


class TightenedMultiSampleDropout(nn.Module):
    """
    Multi-sample dropout with tightened schedule [0.05, 0.10, 0.15, 0.20, 0.25]
    to preserve fine-grained token classification features.
    """
    def __init__(self, dropout_rates: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25)):
        super().__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropout_rates])

    def forward(self, features: torch.Tensor, classifier: nn.Module) -> List[torch.Tensor]:
        if self.training and len(self.dropouts) > 1:
            return [classifier(drop(features)) for drop in self.dropouts]
        return [classifier(features)]


# Backward-compatible alias
MultiSampleDropout = TightenedMultiSampleDropout


class LayoutFeatureEmbedding(nn.Module):
    """
    2D Spatial & Indentation-Aware Layout Feature Embedding.
    Injects column bucket, line-start, and blank-preceded indicators directly
    into token representations.
    Initialized with zeros to guarantee exact identity pass-through at step 0.
    """
    def __init__(self, hidden_size: int = 768, num_col_buckets: int = 5):
        super().__init__()
        self.col_embed = nn.Embedding(num_col_buckets, hidden_size)
        self.line_start_embed = nn.Embedding(2, hidden_size)
        self.blank_preceded_embed = nn.Embedding(2, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

        # Zero init ensures zero disruption to pretrained representations at step 0
        nn.init.zeros_(self.col_embed.weight)
        nn.init.zeros_(self.line_start_embed.weight)
        nn.init.zeros_(self.blank_preceded_embed.weight)

    def forward(
        self,
        col_buckets: torch.Tensor,
        is_line_start: torch.Tensor,
        is_blank_preceded: torch.Tensor
    ) -> torch.Tensor:
        emb = (
            self.col_embed(col_buckets.long())
            + self.line_start_embed(is_line_start.long())
            + self.blank_preceded_embed(is_blank_preceded.long())
        )
        return self.norm(emb)


class EnhancedTokenClassificationHead(nn.Module):
    """
    Enhanced dense token classification head.
    
    Architecture:
    1. Sigmoid-Gated Layer Pooling across top-K transformer layers.
    2. Optional 2D Layout Feature Embedding injection.
    3. 1D Depthwise Separable Convolutional boundary refinement.
    4. Dense GELU + LayerNorm projection.
    5. Tightened Multi-Sample Dropout (MSD) with Per-Branch Focal Loss.
    6. Classification Projection: Linear(H, num_labels).
    """
    def __init__(
        self,
        hidden_size: int = 768,
        num_labels: int = 15,
        num_layers_to_fuse: int = 4,
        dropout_rates: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25),
        intermediate_dim: Optional[int] = None,
        use_conv_refinement: bool = True,
        conv_kernel_size: int = 7,
        use_layout_embeddings: bool = True,
        pooling_type: str = "sigmoid"
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.num_layers_to_fuse = max(1, num_layers_to_fuse)
        inter_dim = intermediate_dim or hidden_size

        # 1. Non-competing Layer Pooling (Sigmoid gating)
        self.pooling_type = pooling_type
        if pooling_type == "sigmoid":
            self.layer_pooling = SigmoidWeightedLayerPooling(num_layers=self.num_layers_to_fuse)
        else:
            self.layer_pooling = WeightedLayerPooling(num_layers=self.num_layers_to_fuse)

        # 2. Optional Layout Feature Embedding
        self.use_layout_embeddings = use_layout_embeddings
        if use_layout_embeddings:
            self.layout_embed = LayoutFeatureEmbedding(hidden_size=hidden_size)
        else:
            self.layout_embed = None

        # 3. Local 1D Depthwise Separable Convolution Refinement
        self.use_conv_refinement = use_conv_refinement
        if use_conv_refinement:
            self.conv_refinement = LocalConvRefinement(
                hidden_size=hidden_size,
                kernel_size=conv_kernel_size,
                dropout_prob=dropout_rates[0] if dropout_rates else 0.1
            )
        else:
            self.conv_refinement = None

        # 4. Dense feature projection with LayerNorm & GELU
        self.dense = nn.Linear(hidden_size, inter_dim)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(inter_dim)

        # 5. Tightened Multi-Sample Dropout (MSD)
        self.multi_sample_dropout = TightenedMultiSampleDropout(dropout_rates=dropout_rates)
        self.dropouts = self.multi_sample_dropout.dropouts  # Backward compatibility

        # 6. Final classification projection
        self.classifier = nn.Linear(inter_dim, num_labels)

        self._init_weights()

    @property
    def layer_weights(self):
        if hasattr(self.layer_pooling, "weights"):
            return self.layer_pooling.weights
        elif hasattr(self.layer_pooling, "layer_weights"):
            return self.layer_pooling.layer_weights
        return None

    def _init_weights(self):
        nn.init.xavier_uniform_(self.dense.weight)
        if self.dense.bias is not None:
            nn.init.zeros_(self.dense.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        hidden_states: Union[Tuple[torch.Tensor, ...], List[torch.Tensor], torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        loss_fct: Optional[nn.Module] = None,
        layout_features: Optional[Dict[str, torch.Tensor]] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        # 1. Pool top-K layers
        fused = self.layer_pooling(hidden_states)

        # 2. Inject 2D Layout Features if provided
        if self.layout_embed is not None and layout_features is not None:
            col_buckets = layout_features.get("col_buckets")
            is_line_start = layout_features.get("is_line_start")
            is_blank_preceded = layout_features.get("is_blank_preceded")
            if col_buckets is not None and is_line_start is not None and is_blank_preceded is not None:
                fused = fused + self.layout_embed(col_buckets, is_line_start, is_blank_preceded)

        # 3. Local boundary refinement with Depthwise Conv1D
        if self.conv_refinement is not None:
            fused = self.conv_refinement(fused, attention_mask=attention_mask)

        # 4. Dense projection
        features = self.norm(self.act(self.dense(fused)))

        # 5. Multi-Sample Dropout & Loss computation
        if self.training and targets is not None and loss_fct is not None:
            branch_logits = self.multi_sample_dropout(features, self.classifier)
            # Compute focal loss per branch BEFORE averaging to prevent ensemble smoothing
            branch_losses = [loss_fct(logits, targets) for logits in branch_logits]
            loss = torch.mean(torch.stack(branch_losses))
            logits = torch.mean(torch.stack(branch_logits, dim=0), dim=0)
            return logits, loss
        elif self.training and len(self.multi_sample_dropout.dropouts) > 1:
            branch_logits = self.multi_sample_dropout(features, self.classifier)
            logits = torch.mean(torch.stack(branch_logits, dim=0), dim=0)
            if targets is not None and loss_fct is not None:
                return logits, loss_fct(logits, targets)
            return logits
        else:
            logits = self.classifier(features)
            if targets is not None and loss_fct is not None:
                return logits, loss_fct(logits, targets)
            return logits


SequenceLabellingHead = EnhancedTokenClassificationHead


class EnhancedBertForTokenClassification(nn.Module):
    """
    Full Token Classification Model packaging a base Transformer backbone with
    the EnhancedTokenClassificationHead, LocalConvRefinement, and FocalLoss.
    
    Fully compatible with Hugging Face Trainer, save_pretrained, and ONNX export.
    """
    supports_gradient_checkpointing = True
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Any,
        base_model: nn.Module,
        num_layers_to_fuse: int = 4,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.05,
        class_weights: Optional[torch.Tensor] = None,
        dropout_rates: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25),
        use_conv_refinement: bool = True,
        conv_kernel_size: int = 7,
        use_layout_embeddings: bool = True,
        per_branch_focal_loss: bool = True
    ):
        super().__init__()
        self.config = config
        self.config.output_hidden_states = True
        self.num_labels = config.num_labels
        self.per_branch_focal_loss = per_branch_focal_loss
        
        # Underlying transformer backbone (e.g. mmBERT-base)
        self.base_model = base_model
        
        hidden_size = getattr(config, "hidden_size", 768)
        self.head = EnhancedTokenClassificationHead(
            hidden_size=hidden_size,
            num_labels=self.num_labels,
            num_layers_to_fuse=num_layers_to_fuse,
            dropout_rates=dropout_rates,
            use_conv_refinement=use_conv_refinement,
            conv_kernel_size=conv_kernel_size,
            use_layout_embeddings=use_layout_embeddings
        )

        self.loss_fct = FocalLoss(
            gamma=focal_gamma,
            weight=class_weights,
            label_smoothing=label_smoothing
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            try:
                self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            except TypeError:
                self.base_model.gradient_checkpointing_enable()
        elif hasattr(self.base_model, "encoder") and hasattr(self.base_model.encoder, "gradient_checkpointing"):
            self.base_model.encoder.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            self.base_model.gradient_checkpointing_disable()
        elif hasattr(self.base_model, "encoder") and hasattr(self.base_model.encoder, "gradient_checkpointing"):
            self.base_model.encoder.gradient_checkpointing = False

    @property
    def is_gradient_checkpointing(self) -> bool:
        return getattr(self.base_model, "is_gradient_checkpointing", False)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(self.base_model, "_set_gradient_checkpointing"):
            self.base_model._set_gradient_checkpointing(module, value=value)

    def can_generate(self) -> bool:
        return False

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:
        if hasattr(self.base_model, "resize_token_embeddings"):
            return self.base_model.resize_token_embeddings(new_num_tokens)
        return None

    def get_input_embeddings(self) -> nn.Module:
        if hasattr(self.base_model, "get_input_embeddings"):
            return self.base_model.get_input_embeddings()
        return None

    def set_input_embeddings(self, value: nn.Module):
        if hasattr(self.base_model, "set_input_embeddings"):
            self.base_model.set_input_embeddings(value)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        col_buckets: Optional[torch.Tensor] = None,
        is_line_start: Optional[torch.Tensor] = None,
        is_blank_preceded: Optional[torch.Tensor] = None,
        **kwargs
    ) -> TokenClassifierOutput:
        
        # Ensure base model outputs hidden states from all layers
        kwargs.pop("output_hidden_states", None)
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )

        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            hidden_states = outputs[0]

        layout_features = None
        if col_buckets is not None and is_line_start is not None and is_blank_preceded is not None:
            layout_features = {
                "col_buckets": col_buckets,
                "is_line_start": is_line_start,
                "is_blank_preceded": is_blank_preceded
            }

        if self.training and labels is not None and self.per_branch_focal_loss:
            head_out = self.head(
                hidden_states,
                attention_mask=attention_mask,
                targets=labels,
                loss_fct=self.loss_fct,
                layout_features=layout_features
            )
            logits, loss = head_out
        else:
            head_out = self.head(
                hidden_states,
                attention_mask=attention_mask,
                layout_features=layout_features
            )
            if isinstance(head_out, tuple):
                logits = head_out[0]
            else:
                logits = head_out
            loss = self.loss_fct(logits, labels) if labels is not None else None

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        
        if hasattr(self.config, "save_pretrained"):
            self.config.save_pretrained(save_directory)
        elif isinstance(self.config, dict):
            with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2)

        saved_safetensors = False
        try:
            from safetensors.torch import save_file
            tensors = {k: v.contiguous() for k, v in self.state_dict().items()}
            save_file(tensors, os.path.join(save_directory, "model.safetensors"))
            saved_safetensors = True
        except Exception:
            saved_safetensors = False

        # Only fall back to pytorch_model.bin if safetensors failed or is unavailable
        if not saved_safetensors:
            model_save_path = os.path.join(save_directory, "pytorch_model.bin")
            torch.save(self.state_dict(), model_save_path)

        meta = {
            "model_type": "enhanced_token_classifier",
            "hidden_size": getattr(self.config, "hidden_size", 768),
            "num_labels": self.num_labels,
            "num_layers_to_fuse": self.head.num_layers_to_fuse,
            "use_conv_refinement": getattr(self.head, "use_conv_refinement", False),
            "use_layout_embeddings": getattr(self.head, "use_layout_embeddings", False),
            "pooling_type": getattr(self.head, "pooling_type", "sigmoid"),
        }
        with open(os.path.join(save_directory, "enhanced_head_config.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, os.PathLike],
        num_labels: Optional[int] = None,
        id2label: Optional[Dict[int, str]] = None,
        label2id: Optional[Dict[str, int]] = None,
        token: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs
    ) -> "EnhancedBertForTokenClassification":
        revision = kwargs.get("revision", None)
        config_kwargs = {}
        if revision is not None:
            config_kwargs["revision"] = revision
        if num_labels is not None:
            config_kwargs["num_labels"] = num_labels
        if id2label is not None:
            config_kwargs["id2label"] = id2label
        if label2id is not None:
            config_kwargs["label2id"] = label2id
            
        config = AutoConfig.from_pretrained(model_name_or_path, token=token, **config_kwargs)
        config.output_hidden_states = True
        
        meta_file = None
        safetensors_file = None
        model_bin_file = None

        if os.path.isdir(str(model_name_or_path)):
            p_meta = os.path.join(str(model_name_or_path), "enhanced_head_config.json")
            if os.path.exists(p_meta):
                meta_file = p_meta
            p_safe = os.path.join(str(model_name_or_path), "model.safetensors")
            if os.path.exists(p_safe):
                safetensors_file = p_safe
            p_bin = os.path.join(str(model_name_or_path), "pytorch_model.bin")
            if os.path.exists(p_bin):
                model_bin_file = p_bin
        else:
            try:
                from transformers.utils.hub import cached_file
                meta_file = cached_file(str(model_name_or_path), "enhanced_head_config.json", token=token, revision=revision)
            except Exception:
                meta_file = None
            try:
                from transformers.utils.hub import cached_file
                safetensors_file = cached_file(str(model_name_or_path), "model.safetensors", token=token, revision=revision)
            except Exception:
                safetensors_file = None
            if not safetensors_file:
                try:
                    from transformers.utils.hub import cached_file
                    model_bin_file = cached_file(str(model_name_or_path), "pytorch_model.bin", token=token, revision=revision)
                except Exception:
                    model_bin_file = None

        num_layers_to_fuse = 4
        use_conv_refinement = True
        use_layout_embeddings = True
        if meta_file and os.path.exists(meta_file):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    num_layers_to_fuse = meta.get("num_layers_to_fuse", 4)
                    use_conv_refinement = meta.get("use_conv_refinement", True)
                    use_layout_embeddings = meta.get("use_layout_embeddings", True)
            except Exception:
                pass

        loaded_state_dict = None
        if safetensors_file and os.path.exists(safetensors_file):
            try:
                from safetensors.torch import load_file
                loaded_state_dict = load_file(safetensors_file, device="cpu")
            except Exception:
                pass
        elif model_bin_file and os.path.exists(model_bin_file):
            try:
                loaded_state_dict = torch.load(model_bin_file, map_location="cpu")
            except Exception:
                pass

        # If this checkpoint already contains full model weights (base_model.* + head.*),
        # initialize base_model from config directly to avoid misleading missing/unexpected warnings from AutoModel
        has_full_weights = loaded_state_dict is not None and any(k.startswith("base_model.") for k in loaded_state_dict.keys())
        if has_full_weights:
            base_model = AutoModel.from_config(config)
        else:
            base_model = AutoModel.from_pretrained(
                model_name_or_path,
                config=config,
                token=token,
                torch_dtype=torch_dtype,
                **kwargs
            )

        # Check and align vocabulary size if token embeddings were expanded during training
        if loaded_state_dict is not None:
            emb_weight = loaded_state_dict.get("base_model.embeddings.tok_embeddings.weight", None)
            if emb_weight is not None and base_model.get_input_embeddings().weight.shape[0] != emb_weight.shape[0]:
                base_model.resize_token_embeddings(emb_weight.shape[0])

        model = cls(
            config=config,
            base_model=base_model,
            num_layers_to_fuse=num_layers_to_fuse,
            use_conv_refinement=use_conv_refinement,
            use_layout_embeddings=use_layout_embeddings
        )

        if loaded_state_dict is not None:
            model.load_state_dict(loaded_state_dict, strict=False)

        return model
