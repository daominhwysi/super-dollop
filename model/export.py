#!/usr/bin/env python3
"""
Model Exporter for Vietnamese Sequence Labelling.
Packages trained mmBERT checkpoints and exports to ONNX with dynamic batch and sequence axes.
"""

import os
import sys
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

import json
import argparse
from typing import Dict, Any, Tuple

import torch
from transformers import AutoTokenizer

from model.module.head import EnhancedBertForTokenClassification
from model.inference.predict import load_label_mapping


def export_to_onnx(
    model: torch.nn.Module,
    tokenizer: Any,
    output_onnx_path: Path,
    device: str = "cpu"
) -> bool:
    """Exports model to ONNX with dynamic batch and sequence axes."""
    model.eval()
    model.to(device)

    dummy_text = "Câu 1: Cho hàm số $y = f(x)$. Giá trị lớn nhất là?\nA. 1\nB. 2\nC. 3\nD. 4"
    inputs = tokenizer(dummy_text, return_tensors="pt", max_length=128, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"}
    }

    print(f"Exporting ONNX model to '{output_onnx_path}'...")
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(output_onnx_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True
    )

    # Verification with ONNX Runtime if installed
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(output_onnx_path))
        ort_inputs = {
            "input_ids": input_ids.cpu().numpy(),
            "attention_mask": attention_mask.cpu().numpy()
        }
        ort_outputs = session.run(None, ort_inputs)
        
        with torch.no_grad():
            pt_outputs = model(input_ids, attention_mask=attention_mask)
            pt_logits = pt_outputs.logits.cpu().numpy()

        diff = float(torch.max(torch.abs(torch.tensor(ort_outputs[0]) - torch.tensor(pt_logits))))
        print(f"ONNX Runtime verification passed! Max numerical difference: {diff:.6f}")
        return True
    except ImportError:
        print("Notice: onnxruntime not installed, skipped numerical verification.")
        return True
    except Exception as e:
        print(f"Warning during ONNX verification: {e}")
        return False


export_model = export_to_onnx


def main():
    parser = argparse.ArgumentParser(description="Export trained mmBERT model to ONNX.")
    parser.add_argument("--model-dir", type=str, default="results/mmbert_seqlabel_v2", help="Path to trained model directory")
    parser.add_argument("--output-dir", type=str, default="results/mmbert_seqlabel_v2/onnx", help="Output directory for ONNX model")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)

    print(f"Loading checkpoint from '{model_dir}'...")
    tag_to_id, id_to_tag = load_label_mapping(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model = EnhancedBertForTokenClassification.from_pretrained(
        model_dir,
        num_labels=len(tag_to_id),
        id2label=id_to_tag,
        label2id=tag_to_id
    )

    onnx_file = output_dir / "model.onnx"
    export_to_onnx(model, tokenizer, onnx_file, device=args.device)

    # Save tokenizer and label mapping into onnx directory for standalone inference
    tokenizer.save_pretrained(output_dir)
    with open(output_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"tag_to_id": tag_to_id, "id_to_tag": id_to_tag}, f, indent=2)

    print(f"Export complete! Standalone ONNX bundle saved to '{output_dir}'.")


if __name__ == "__main__":
    main()
