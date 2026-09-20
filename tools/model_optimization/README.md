# iter1 Haswell optimization

The selected deployment artifact is
[`export_onnx/iter1-haswell-int8.onnx`](../../export_onnx/iter1-haswell-int8.onnx).
It is derived from `iter1.pth` through the existing FP32 RF-DETR export and is
tuned for this machine's Intel Core i3-4130T (Haswell, AVX2, 2 physical cores,
4 SMT threads).

Artifact SHA-256: `a4ab8df72a88fcce3878ad0dd73b351d44b52dfafe75b2d5976497ac9fc576e1`.

## Selected optimization

The model uses ONNX Runtime dynamic, per-channel signed INT8 quantization for
the 119 constant-weight `MatMul` nodes. Activations, convolutions, and all
remaining operators stay FP32. This mixed graph was selected from measured
candidates; static activation quantization was both slower and less faithful
on this CPU.

Use two intra-op threads, one inter-op thread, sequential execution, and full
graph optimization:

```python
import onnxruntime as ort

options = ort.SessionOptions()
options.intra_op_num_threads = 2
options.inter_op_num_threads = 1
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    "export_onnx/iter1-haswell-int8.onnx",
    options,
    providers=["CPUExecutionProvider"],
)
```

Inference inputs must match RF-DETR preprocessing: RGB, 704 x 704 tensor
bilinear resize with antialiasing disabled, values scaled to `[0, 1]`, then
ImageNet normalization. The input layout is NCHW. Outputs are normalized
`cxcywh` boxes (`dets`) and logits (`labels`); the final logit slot is the
no-object slot and must be excluded before applying sigmoid.

## Device results

The reproducible result is stored in
[`data/benchmark_results/iter1_haswell_benchmark.json`](../../data/benchmark_results/iter1_haswell_benchmark.json).
The benchmark used all 36 representative exam-page images for agreement and
eight images for latency, with two warmups and seven timed runs per model.

| Measurement | FP32 ONNX | Selected INT8 | Change |
| --- | ---: | ---: | ---: |
| File size | 123.50 MiB | 45.17 MiB | 63.4% smaller |
| Median latency | 1284.2 ms/page | 1197.4 ms/page | 1.072x faster |
| Session RSS after load | 205.1 MiB | 132.8 MiB | 72.3 MiB lower |
| RSS after first inference | 390.2 MiB | 320.9 MiB | 69.4 MiB lower |

Detection agreement treats FP32 output as the reference and greedily matches
same-class boxes at IoU >= 0.5:

| Confidence | FP32 / INT8 detections | Matched | Precision | Recall | Mean matched IoU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.3 | 36 / 38 | 36 | 0.947 | 1.000 | 0.976 |
| 0.5 | 26 / 26 | 26 | 1.000 | 1.000 | 0.976 |
| 0.7 | 17 / 16 | 16 | 1.000 | 0.941 | 0.978 |

These are equivalence measurements, not ground-truth mAP: the repository does
not contain labeled boxes for the benchmark pages. At the default 0.3 cutoff,
two candidate scores cross above the threshold; at 0.7, one crosses below it.
Do not silently change the production threshold based only on this benchmark.
Validate mAP on the training/validation dataset if it becomes available.

## OCR pipeline integration

The production converter loads this artifact lazily through
`backend/app/domains/ocr/annotator/figure_detector.py`. Detection runs in
absolute PDF page order before vision OCR batches are submitted, so figure IDs
remain continuous (`fig_1`, `fig_2`, ...) even when OCR batches run
concurrently.

The checkpoint's foreground classes have different handling:

- Class `0` (`bangbienthien`) is a mathematical variation table. It is ignored
  by figure detection and remains available to the normal OCR/table pipeline.
- Class `1` (`class_1`) is a general figure. It receives a border and an ID
  badge, and the vision OCR model emits
  `<figure id="fig_N" description="One concise sentence." />` at its logical
  reading position. After the LLM returns, the deterministic projector appends
  `bbox="x1,y1,x2,y2"` using coordinates projected back from the stretched
  704 x 704 detector input to the original rendered page.

The confidence threshold, model path, included class IDs, and class names are
configured under `models.ocr.figure_detection` in `backend/config.yaml`.

The FP32 ONNX source was also checked against `iter1_clean.pth` on two images.
Mean absolute error was approximately `6e-6` to `9e-6` for boxes and `2e-5` to
`3e-5` for logits, confirming that the ONNX reference represents the source
checkpoint.

## Reproduce

From the repository root:

```bash
.venv/bin/python tools/model_optimization/optimize_iter1.py --force
.venv/bin/python tools/model_optimization/benchmark_iter1.py \
  --output data/benchmark_results/iter1_haswell_benchmark.json
```

The optimizer records the source checkpoint and ONNX SHA-256 hashes, target CPU,
quantization recipe, and recommended runtime settings in ONNX metadata.

## Rejected candidates

- Gemini's full dynamic INT8 artifact was smaller (39.14 MiB) but matched only
  34 of 36 FP32 detections at confidence 0.3 and was slower than the selected
  per-channel model.
- Static QDQ S8/S8 quantization was about 1.58 s/page in the probe and lost half
  of the FP32 detections at confidence 0.5. Haswell lacks the newer INT8 vector
  instructions that make this path attractive on recent Intel CPUs.
- OpenVINO 2026.3 FP32 reached about 1.19 s/page in a one-image thread sweep,
  essentially tying the selected model, but did not reduce artifact size and
  required a new runtime. OpenVINO execution of the quantized graph was slower
  at about 1.69 s/page.
