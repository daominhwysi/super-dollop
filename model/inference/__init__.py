"""
Inference subsystem for sliding-window sequence labelling and zero-drift XML reconstruction.
"""

from model.inference.predict import (
    predict_text,
    build_tagged_xml,
    reconstruct_xml_from_predictions,
    SequenceLabelPredictor,
)

__all__ = [
    "predict_text",
    "build_tagged_xml",
    "reconstruct_xml_from_predictions",
    "SequenceLabelPredictor",
]
