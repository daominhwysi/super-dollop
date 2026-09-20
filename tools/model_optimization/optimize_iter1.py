#!/usr/bin/env python3
"""Build the accuracy-preserving INT8 ONNX variant selected for this host.

The RF-DETR decoder and detection heads remain FP32. Only MatMul nodes with a
constant weight input are dynamically quantized, using per-output-channel
signed INT8 weights. This is intentionally narrower than quantizing Conv nodes
or activations: both alternatives were slower and less accurate on Haswell.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


DEFAULT_SOURCE = Path("export_onnx/rfdetr-small.onnx")
DEFAULT_OUTPUT = Path("export_onnx/iter1-haswell-int8.onnx")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def constant_weight_matmuls(model: onnx.ModelProto) -> list[str]:
    """Return named MatMuls whose A or B input is a serialized initializer."""
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    return [
        node.name
        for node in model.graph.node
        if node.op_type == "MatMul"
        and node.name
        and any(input_name in initializer_names for input_name in node.input)
    ]


def set_metadata(model: onnx.ModelProto, values: dict[str, str]) -> None:
    existing = {entry.key: entry for entry in model.metadata_props}
    for key, value in values.items():
        entry = existing.get(key)
        if entry is None:
            entry = model.metadata_props.add()
            entry.key = key
        entry.value = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("iter1.pth"),
        help="Optional source checkpoint recorded in ONNX metadata.",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing output file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    checkpoint = args.checkpoint.resolve()

    if not source.is_file():
        raise FileNotFoundError(f"FP32 ONNX source does not exist: {source}")
    if output.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output}. Pass --force to replace it.")
    if source == output:
        raise ValueError("Source and output paths must be different.")

    source_model = onnx.load(source, load_external_data=False)
    nodes = constant_weight_matmuls(source_model)
    if not nodes:
        raise ValueError("No constant-weight MatMul nodes were found in the source graph.")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.tmp.onnx")
    if temporary_output.exists():
        temporary_output.unlink()

    print(f"Quantizing {len(nodes)} constant-weight MatMul nodes with per-channel QInt8 weights...")
    try:
        quantize_dynamic(
            model_input=source,
            model_output=temporary_output,
            weight_type=QuantType.QInt8,
            per_channel=True,
            nodes_to_quantize=nodes,
        )

        optimized_model = onnx.load(temporary_output, load_external_data=False)
        metadata = {
            "azozo.optimization": "dynamic-per-channel-qint8-constant-matmul",
            "azozo.quantized_node_count": str(len(nodes)),
            "azozo.source_onnx_sha256": sha256_file(source),
            "azozo.target_cpu": "Intel Haswell AVX2; 2 physical cores / 4 SMT threads",
            "azozo.runtime": "ONNX Runtime CPUExecutionProvider; intra_op=2; inter_op=1; sequential",
            "azozo.build_host": platform.platform(),
        }
        if checkpoint.is_file():
            metadata["azozo.source_checkpoint"] = checkpoint.name
            metadata["azozo.source_checkpoint_sha256"] = sha256_file(checkpoint)
        set_metadata(optimized_model, metadata)
        onnx.checker.check_model(optimized_model)
        onnx.save(optimized_model, temporary_output)
        os.replace(temporary_output, output)
    finally:
        if temporary_output.exists():
            temporary_output.unlink()

    before_mib = source.stat().st_size / (1024 * 1024)
    after_mib = output.stat().st_size / (1024 * 1024)
    reduction = 100 * (1 - after_mib / before_mib)
    print(f"Wrote {output}")
    print(f"Size: {before_mib:.2f} MiB -> {after_mib:.2f} MiB ({reduction:.1f}% smaller)")
    print(f"SHA-256: {sha256_file(output)}")


if __name__ == "__main__":
    main()
