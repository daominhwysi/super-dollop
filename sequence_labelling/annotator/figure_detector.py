"""RF-DETR figure detection and page annotation for the vision OCR pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class FigureDetection:
    """One figure detection in original page pixel coordinates."""

    box: tuple[int, int, int, int]
    score: float
    class_id: int
    class_name: str
    figure_id: str = ""
    page_number: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.figure_id,
            "page": self.page_number,
            "label": self.class_name,
            "class_id": self.class_id,
            "score": round(self.score, 6),
            "box": list(self.box),
        }


class RFDETRFigureDetector:
    """Run the optimized RF-DETR ONNX model with its Haswell settings."""

    IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    COLORS = ((30, 113, 230), (31, 150, 90), (174, 93, 35), (145, 71, 161))

    def __init__(
        self,
        model_path: Path | str,
        confidence_threshold: float = 0.3,
        class_names: Optional[Sequence[str]] = None,
        included_class_ids: Optional[Sequence[int]] = (1,),
        intra_op_threads: int = 2,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.class_names = tuple(class_names or ("bangbienthien", "class_1"))
        self.included_class_ids = (
            frozenset(included_class_ids) if included_class_ids is not None else None
        )
        self.intra_op_threads = intra_op_threads
        self._session: Any = None

    def _get_session(self) -> Any:
        if self._session is not None:
            return self._session
        if not self.model_path.is_file():
            raise FileNotFoundError(f"RF-DETR model not found: {self.model_path}")

        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = self.intra_op_threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(self.model_path),
            options,
            providers=["CPUExecutionProvider"],
        )
        return self._session

    @classmethod
    def preprocess(cls, image_bgr: np.ndarray, height: int, width: int) -> np.ndarray:
        """Stretch directly to model size, then return normalized RGB NCHW input.

        The resize intentionally does not preserve the source aspect ratio or add
        letterbox padding. RF-DETR boxes remain normalized to this stretched input,
        so decoding can project them directly back to the source page dimensions.
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        normalized = (resized.astype(np.float32) / 255.0 - cls.IMAGENET_MEAN) / cls.IMAGENET_STD
        return np.transpose(normalized, (2, 0, 1))[None, ...].astype(np.float32)

    def decode(
        self,
        outputs: Sequence[np.ndarray],
        image_height: int,
        image_width: int,
    ) -> list[FigureDetection]:
        """Decode normalized cxcywh boxes and foreground logits into page pixels."""
        boxes = np.asarray(outputs[0])[0]
        logits = np.asarray(outputs[1])[0, :, :-1]
        scores_by_class = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
        labels = scores_by_class.argmax(axis=1)
        scores = scores_by_class.max(axis=1)
        keep = scores > self.confidence_threshold

        detections: list[FigureDetection] = []
        for box, class_id, score in zip(boxes[keep], labels[keep], scores[keep]):
            cx, cy, box_width, box_height = (float(value) for value in box)
            x1 = int(round((cx - box_width / 2) * image_width))
            y1 = int(round((cy - box_height / 2) * image_height))
            x2 = int(round((cx + box_width / 2) * image_width))
            y2 = int(round((cy + box_height / 2) * image_height))
            x1, x2 = sorted((max(0, min(image_width - 1, x1)), max(0, min(image_width, x2))))
            y1, y2 = sorted((max(0, min(image_height - 1, y1)), max(0, min(image_height, y2))))
            if x2 <= x1 or y2 <= y1:
                continue
            class_index = int(class_id)
            if (
                self.included_class_ids is not None
                and class_index not in self.included_class_ids
            ):
                continue
            class_name = (
                self.class_names[class_index]
                if class_index < len(self.class_names)
                else f"class_{class_index}"
            )
            detections.append(
                FigureDetection(
                    box=(x1, y1, x2, y2),
                    score=float(score),
                    class_id=class_index,
                    class_name=class_name,
                )
            )
        return self.sort_reading_order(detections)

    @staticmethod
    def sort_reading_order(
        detections: Iterable[FigureDetection], row_tolerance: int = 12
    ) -> list[FigureDetection]:
        """Sort top-to-bottom, then left-to-right for figures sharing a row."""
        ordered = sorted(detections, key=lambda item: (item.box[1], item.box[0]))
        rows: list[list[FigureDetection]] = []
        row_bottoms: list[int] = []
        for detection in ordered:
            y1, y2 = detection.box[1], detection.box[3]
            for index, bottom in enumerate(row_bottoms):
                row_top = min(item.box[1] for item in rows[index])
                if y1 <= bottom + row_tolerance and y2 >= row_top - row_tolerance:
                    rows[index].append(detection)
                    row_bottoms[index] = max(bottom, y2)
                    break
            else:
                rows.append([detection])
                row_bottoms.append(y2)
        rows.sort(key=lambda row: min(item.box[1] for item in row))
        return [item for row in rows for item in sorted(row, key=lambda value: value.box[0])]

    def detect(self, image_bgr: np.ndarray) -> list[FigureDetection]:
        session = self._get_session()
        input_spec = session.get_inputs()[0]
        _, _, input_height, input_width = input_spec.shape
        tensor = self.preprocess(image_bgr, int(input_height), int(input_width))
        outputs = session.run(None, {input_spec.name: tensor})
        height, width = image_bgr.shape[:2]
        return self.decode(outputs, height, width)

    def number_page(
        self,
        detections: Sequence[FigureDetection],
        page_number: int,
        first_figure_number: int,
    ) -> list[FigureDetection]:
        return [
            replace(
                detection,
                figure_id=f"fig_{first_figure_number + index}",
                page_number=page_number,
            )
            for index, detection in enumerate(detections)
        ]

    def draw(self, image_bgr: np.ndarray, detections: Sequence[FigureDetection]) -> np.ndarray:
        """Draw unobtrusive borders and readable ID badges without covering figures."""
        output = image_bgr.copy()
        height, width = output.shape[:2]
        thickness = max(2, round(min(height, width) / 700))
        font_scale = max(0.5, min(1.0, min(height, width) / 1100))
        font_thickness = max(1, thickness - 1)

        for detection in detections:
            x1, y1, x2, y2 = detection.box
            color = self.COLORS[detection.class_id % len(self.COLORS)]
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

            badge = f"FIGURE {detection.figure_id}"
            (text_width, text_height), baseline = cv2.getTextSize(
                badge, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            badge_height = text_height + baseline + 8
            badge_width = text_width + 12
            badge_x = min(max(0, x1), max(0, width - badge_width))
            badge_y1 = y1 - badge_height if y1 >= badge_height else y1
            badge_y2 = min(height, badge_y1 + badge_height)
            cv2.rectangle(
                output,
                (badge_x, badge_y1),
                (min(width, badge_x + badge_width), badge_y2),
                color,
                -1,
            )
            cv2.putText(
                output,
                badge,
                (badge_x + 6, badge_y2 - baseline - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA,
            )
        return output

    @staticmethod
    def crops(
        image_bgr: np.ndarray, detections: Sequence[FigureDetection]
    ) -> dict[str, np.ndarray]:
        """Return independent ROI copies keyed by stable figure ID."""
        return {
            detection.figure_id: image_bgr[
                detection.box[1] : detection.box[3], detection.box[0] : detection.box[2]
            ].copy()
            for detection in detections
        }
