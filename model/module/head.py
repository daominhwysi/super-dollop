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
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import AutoModel, AutoConfig, PreTrainedModel


class FocalLoss(nn.Module):
    """
    Multiclass Focal Loss for addressing class imbalance by dynamically down-weighting
    easy, well-classified examples (e.g. background 'O' tokens) and focusing on hard boundary tokens.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(
        self,
        gamma: float = 1.5,
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

        # Standard cross entropy with optional label smoothing
        ce_loss = F.cross_entropy(
            valid_logits,
            valid_targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing
        )

        # Probabilities of true class
        p_t = torch.exp(-ce_loss)
        
        # Focal modulation factor: (1 - p_t)^gamma
        focal_loss = ((1.0 - p_t) ** self.gamma) * ce_loss

        return focal_loss.mean()


class WeightedLayerPooling(nn.Module):
    """
    Weighted Layer Pooling computes a softmax-weighted linear combination of representations
    from the last K transformer layers.
    """
    def __init__(self, num_layers_to_fuse: int = 4):
        super().__init__()
        self.num_layers_to_fuse = max(1, num_layers_to_fuse)
        self.layer_weights = nn.Parameter(torch.ones(self.num_layers_to_fuse))

    def forward(
        self,
        hidden_states: Union[Tuple[torch.Tensor, ...], List[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(hidden_states, (tuple, list)):
            k = min(self.num_layers_to_fuse, len(hidden_states))
            selected_layers = hidden_states[-k:]
            stacked = torch.stack(selected_layers, dim=0)  # Shape: [K, B, T, H]
            weights = self.layer_weights[:k]
            norm_weights = F.softmax(weights, dim=0).view(-1, 1, 1, 1)
            return torch.sum(stacked * norm_weights, dim=0)
        return hidden_states


class MultiSampleDropout(nn.Module):
    """
    Multi-Sample Dropout applies multiple parallel dropout masks during training
    and averages their projected logits, accelerating convergence and improving generalization.
    """
    def __init__(self, dropout_rates: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)):
        super().__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropout_rates])

    def forward(self, features: torch.Tensor, classifier: nn.Module) -> torch.Tensor:
        if self.training and len(self.dropouts) > 1:
            logits_list = [classifier(drop(features)) for drop in self.dropouts]
            return torch.mean(torch.stack(logits_list, dim=0), dim=0)
        return classifier(features)


class EnhancedTokenClassificationHead(nn.Module):
    """
    Enhanced dense token classification head.
    
    Architecture:
    1. Weighted Layer Pooling: Softmax-weighted linear combination of the last K hidden states.
    2. Dense Layer: Linear(H, H) -> GELU() -> LayerNorm(H).
    3. Multi-Sample Dropout: K parallel dropout masks averaged during training.
    4. Classification Projection: Linear(H, num_labels).
    """
    def __init__(
        self,
        hidden_size: int = 768,
        num_labels: int = 15,
        num_layers_to_fuse: int = 4,
        dropout_rates: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5),
        intermediate_dim: Optional[int] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.num_layers_to_fuse = max(1, num_layers_to_fuse)
        inter_dim = intermediate_dim or hidden_size

        # 1. Learnable layer pooling weights across the last K layers
        self.layer_weights = nn.Parameter(torch.ones(self.num_layers_to_fuse))

        # 2. Dense feature projection with LayerNorm & GELU
        self.dense = nn.Linear(hidden_size, inter_dim)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(inter_dim)

        # 3. Multi-Sample Dropout (MSD)
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropout_rates])

        # 4. Final classification layer
        self.classifier = nn.Linear(inter_dim, num_labels)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.dense.weight)
        if self.dense.bias is not None:
            nn.init.zeros_(self.dense.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        hidden_states: Union[Tuple[torch.Tensor, ...], List[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(hidden_states, (tuple, list)):
            k = min(self.num_layers_to_fuse, len(hidden_states))
            selected_layers = hidden_states[-k:]
            stacked = torch.stack(selected_layers, dim=0)  # Shape: [K, B, T, H]
            
            # Slice or pad layer_weights to match actual K
            weights = self.layer_weights[:k]
            norm_weights = F.softmax(weights, dim=0).view(-1, 1, 1, 1)
            
            # Weighted average across layers
            fused = torch.sum(stacked * norm_weights, dim=0)  # Shape: [B, T, H]
        else:
            fused = hidden_states

        # Intermediate non-linear projection
        features = self.norm(self.act(self.dense(fused)))

        # Multi-Sample Dropout averaging during training
        if self.training and len(self.dropouts) > 1:
            logits_list = [self.classifier(drop(features)) for drop in self.dropouts]
            logits = torch.mean(torch.stack(logits_list, dim=0), dim=0)
        else:
            logits = self.classifier(features)

        return logits


class EnhancedBertForTokenClassification(nn.Module):
    """
    Full Token Classification Model packaging a base Transformer backbone with
    the EnhancedTokenClassificationHead and FocalLoss.
    
    Fully compatible with Hugging Face Trainer, save_pretrained, and ONNX export.
    """
    supports_gradient_checkpointing = True
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Any,
        base_model: nn.Module,
        num_layers_to_fuse: int = 4,
        focal_gamma: float = 1.5,
        label_smoothing: float = 0.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.config = config
        self.config.output_hidden_states = True
        self.num_labels = config.num_labels
        
        # Underlying transformer backbone (e.g. mmBERT-base)
        self.base_model = base_model
        
        hidden_size = getattr(config, "hidden_size", 768)
        self.head = EnhancedTokenClassificationHead(
            hidden_size=hidden_size,
            num_labels=self.num_labels,
            num_layers_to_fuse=num_layers_to_fuse
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

        logits = self.head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

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

        try:
            from safetensors.torch import save_file
            tensors = {k: v.contiguous() for k, v in self.state_dict().items()}
            save_file(tensors, os.path.join(save_directory, "model.safetensors"))
        except Exception:
            pass

        model_save_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_save_path)

        meta = {
            "model_type": "enhanced_token_classifier",
            "hidden_size": getattr(self.config, "hidden_size", 768),
            "num_labels": self.num_labels,
            "num_layers_to_fuse": self.head.num_layers_to_fuse,
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
        config_kwargs = {}
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
                meta_file = cached_file(str(model_name_or_path), "enhanced_head_config.json", token=token)
            except Exception:
                meta_file = None

        num_layers_to_fuse = 4
        if meta_file and os.path.exists(meta_file):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    num_layers_to_fuse = meta.get("num_layers_to_fuse", 4)
            except Exception:
                pass

        base_model = AutoModel.from_pretrained(
            model_name_or_path,
            config=config,
            token=token,
            torch_dtype=torch_dtype,
            **kwargs
        )
        
        model = cls(
            config=config,
            base_model=base_model,
            num_layers_to_fuse=num_layers_to_fuse
        )

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

        if loaded_state_dict is not None:
            model.load_state_dict(loaded_state_dict, strict=False)

        return model
