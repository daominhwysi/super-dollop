#!/usr/bin/env python3
"""Compare an optimized RF-DETR ONNX model with its FP32 ONNX reference."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms.functional as vision_f
from PIL import Image


@dataclass
class DetectionMetrics:
    reference_count: int
    candidate_count: int
    matched_count: int
    precision: float
    recall: float
    mean_matched_iou: float
    mean_matched_score_error: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("export_onnx/rfdetr-small.onnx"))
    parser.add_argument("--candidate", type=Path, default=Path("export_onnx/iter1-haswell-int8.onnx"))
    parser.add_argument("--images", type=Path, default=Path("data/benchmark_sample_images"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.7],
        help="Confidence thresholds at which decoded detections are compared.",
    )
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--latency-runs", type=int, default=7)
    parser.add_argument("--latency-images", type=int, default=8)
    return parser.parse_args()


def create_session(path: Path, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])


def preprocess(path: Path, height: int, width: int) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Match RFDETR.predict(): tensor bilinear resize with antialias disabled."""
    with Image.open(path) as image:
        tensor = vision_f.to_tensor(image.convert("RGB"))
    tensor = vision_f.resize(tensor, [height, width], antialias=False)
    tensor = vision_f.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return tensor.unsqueeze(0).numpy().astype(np.float32)


def decode(
    outputs: list[np.ndarray[Any, Any]], threshold: float
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    boxes = outputs[0][0]
    logits = outputs[1][0, :, :-1]
    scores_by_class = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
    classes = scores_by_class.argmax(axis=1)
    scores = scores_by_class.max(axis=1)
    keep = scores > threshold
    boxes = boxes[keep]
    xyxy = np.stack(
        [
            boxes[:, 0] - boxes[:, 2] / 2,
            boxes[:, 1] - boxes[:, 3] / 2,
            boxes[:, 0] + boxes[:, 2] / 2,
            boxes[:, 1] + boxes[:, 3] / 2,
        ],
        axis=1,
    )
    return xyxy, classes[keep], scores[keep]


def box_iou(first: np.ndarray[Any, Any], second: np.ndarray[Any, Any]) -> float:
    lower = np.maximum(first[:2], second[:2])
    upper = np.minimum(first[2:], second[2:])
    intersection = float(np.maximum(upper - lower, 0).prod())
    first_area = float(np.maximum(first[2:] - first[:2], 0).prod())
    second_area = float(np.maximum(second[2:] - second[:2], 0).prod())
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def match_detections(
    reference: list[np.ndarray[Any, Any]],
    candidate: list[np.ndarray[Any, Any]],
    threshold: float,
    minimum_iou: float,
) -> tuple[int, int, list[tuple[float, float]]]:
    ref_boxes, ref_classes, ref_scores = decode(reference, threshold)
    cand_boxes, cand_classes, cand_scores = decode(candidate, threshold)
    possible_matches: list[tuple[float, int, int]] = []
    for ref_index, ref_box in enumerate(ref_boxes):
        for cand_index, cand_box in enumerate(cand_boxes):
            if ref_classes[ref_index] == cand_classes[cand_index]:
                possible_matches.append((box_iou(ref_box, cand_box), ref_index, cand_index))

    used_reference: set[int] = set()
    used_candidate: set[int] = set()
    matches: list[tuple[float, float]] = []
    for iou, ref_index, cand_index in sorted(possible_matches, reverse=True):
        if iou < minimum_iou:
            break
        if ref_index in used_reference or cand_index in used_candidate:
            continue
        used_reference.add(ref_index)
        used_candidate.add(cand_index)
        matches.append((iou, abs(float(ref_scores[ref_index] - cand_scores[cand_index]))))
    return len(ref_boxes), len(cand_boxes), matches


def summarize_detections(
    reference_outputs: list[list[np.ndarray[Any, Any]]],
    candidate_outputs: list[list[np.ndarray[Any, Any]]],
    threshold: float,
    minimum_iou: float,
) -> DetectionMetrics:
    reference_count = 0
    candidate_count = 0
    matches: list[tuple[float, float]] = []
    for reference, candidate in zip(reference_outputs, candidate_outputs, strict=True):
        ref_count, cand_count, image_matches = match_detections(reference, candidate, threshold, minimum_iou)
        reference_count += ref_count
        candidate_count += cand_count
        matches.extend(image_matches)
    matched_count = len(matches)
    return DetectionMetrics(
        reference_count=reference_count,
        candidate_count=candidate_count,
        matched_count=matched_count,
        precision=matched_count / candidate_count if candidate_count else 1.0,
        recall=matched_count / reference_count if reference_count else 1.0,
        mean_matched_iou=statistics.fmean(match[0] for match in matches) if matches else 0.0,
        mean_matched_score_error=statistics.fmean(match[1] for match in matches) if matches else 0.0,
    )


def benchmark_latency(
    session: ort.InferenceSession,
    inputs: list[np.ndarray[Any, Any]],
    warmup: int,
    runs: int,
) -> dict[str, Any]:
    input_name = session.get_inputs()[0].name
    for index in range(warmup):
        session.run(None, {input_name: inputs[index % len(inputs)]})
    timings = []
    for index in range(runs):
        started = time.perf_counter()
        session.run(None, {input_name: inputs[index % len(inputs)]})
        timings.append((time.perf_counter() - started) * 1000)
    return {
        "runs": runs,
        "median_ms": statistics.median(timings),
        "mean_ms": statistics.fmean(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
        "samples_ms": timings,
    }


def main() -> None:
    args = parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")
    image_paths = sorted(path for path in args.images.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not image_paths:
        raise FileNotFoundError(f"No benchmark images found in {args.images}")

    reference_session = create_session(args.reference, args.threads)
    candidate_session = create_session(args.candidate, args.threads)
    reference_shape = reference_session.get_inputs()[0].shape
    candidate_shape = candidate_session.get_inputs()[0].shape
    if reference_shape != candidate_shape:
        raise ValueError(f"Model input mismatch: reference={reference_shape}, candidate={candidate_shape}")
    _, _, height, width = reference_shape
    inputs = [preprocess(path, height, width) for path in image_paths]

    reference_input = reference_session.get_inputs()[0].name
    candidate_input = candidate_session.get_inputs()[0].name
    print(f"Comparing detections on {len(inputs)} images...")
    reference_outputs = [reference_session.run(None, {reference_input: value}) for value in inputs]
    candidate_outputs = [candidate_session.run(None, {candidate_input: value}) for value in inputs]
    agreements = {
        str(threshold): summarize_detections(
            reference_outputs,
            candidate_outputs,
            threshold,
            args.match_iou,
        )
        for threshold in args.thresholds
    }

    latency_inputs = inputs[: args.latency_images]
    reference_latency = benchmark_latency(reference_session, latency_inputs, args.warmup, args.latency_runs)
    candidate_latency = benchmark_latency(candidate_session, latency_inputs, args.warmup, args.latency_runs)
    reference_size = args.reference.stat().st_size
    candidate_size = args.candidate.stat().st_size
    result = {
        "device": {
            "provider": "CPUExecutionProvider",
            "intra_op_threads": args.threads,
            "inter_op_threads": 1,
            "execution_mode": "sequential",
        },
        "inputs": {
            "image_directory": str(args.images),
            "correctness_images": len(inputs),
            "latency_images": len(latency_inputs),
            "shape": reference_shape,
            "preprocessing": "RF-DETR tensor bilinear resize, antialias=False, ImageNet normalization",
        },
        "models": {
            "reference": {"path": str(args.reference), "size_bytes": reference_size},
            "candidate": {"path": str(args.candidate), "size_bytes": candidate_size},
            "size_reduction_percent": 100 * (1 - candidate_size / reference_size),
        },
        "detection_agreement": {
            "minimum_match_iou": args.match_iou,
            "by_confidence_threshold": {
                threshold: asdict(metrics) for threshold, metrics in agreements.items()
            },
        },
        "latency": {
            "reference": reference_latency,
            "candidate": candidate_latency,
            "speedup": reference_latency["median_ms"] / candidate_latency["median_ms"],
        },
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
