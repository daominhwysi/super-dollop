from __future__ import annotations

import numpy as np

from sequence_labelling.annotator.figure_detector import (
    FigureDetection,
    RFDETRFigureDetector,
)
from sequence_labelling.annotator.pdf_converter import (
    format_figure_inventory,
    project_figures_to_llm_output,
)
from sequence_labelling.annotator.annotate_ocr import parse_xml_annotations


def _detection(
    box: tuple[int, int, int, int], figure_id: str = "", page: int = 0
) -> FigureDetection:
    return FigureDetection(
        box=box,
        score=0.91,
        class_id=0,
        class_name="bangbienthien",
        figure_id=figure_id,
        page_number=page,
    )


def test_decode_scales_boxes_and_excludes_no_object_slot() -> None:
    detector = RFDETRFigureDetector("unused.onnx", confidence_threshold=0.5)
    boxes = np.asarray([[[0.5, 0.5, 0.4, 0.2], [0.1, 0.1, 0.1, 0.1]]], dtype=np.float32)
    logits = np.asarray([[[-4.0, 4.0, 20.0], [-4.0, -4.0, 20.0]]], dtype=np.float32)

    detections = detector.decode([boxes, logits], image_height=1000, image_width=500)

    assert len(detections) == 1
    assert detections[0].box == (150, 400, 350, 600)
    assert detections[0].class_name == "class_1"


def test_preprocess_stretch_resizes_to_model_dimensions() -> None:
    source = np.zeros((100, 300, 3), dtype=np.uint8)

    tensor = RFDETRFigureDetector.preprocess(source, height=704, width=704)

    assert tensor.shape == (1, 3, 704, 704)
    assert tensor.dtype == np.float32


def test_decode_skips_variation_table_class() -> None:
    detector = RFDETRFigureDetector("unused.onnx", confidence_threshold=0.5)
    boxes = np.asarray([[[0.5, 0.5, 0.4, 0.2]]], dtype=np.float32)
    variation_table_logits = np.asarray([[[4.0, -4.0, 20.0]]], dtype=np.float32)

    assert detector.decode([boxes, variation_table_logits], 1000, 500) == []


def test_numbering_is_continuous_across_pages() -> None:
    detector = RFDETRFigureDetector("unused.onnx")
    page_one = detector.number_page([_detection((0, 0, 20, 20))], 1, 1)
    page_two = detector.number_page(
        [_detection((0, 0, 20, 20)), _detection((30, 0, 50, 20))], 2, 2
    )

    assert [item.figure_id for item in page_one + page_two] == ["fig_1", "fig_2", "fig_3"]
    assert [item.page_number for item in page_one + page_two] == [1, 2, 2]


def test_draw_preserves_input_and_crops_use_stable_ids() -> None:
    detector = RFDETRFigureDetector("unused.onnx")
    source = np.full((120, 160, 3), 255, dtype=np.uint8)
    detection = _detection((20, 30, 100, 90), "fig_7", 3)

    annotated = detector.draw(source, [detection])
    crops = detector.crops(source, [detection])

    assert np.all(source == 255)
    assert not np.array_equal(annotated, source)
    assert crops["fig_7"].shape == (60, 80, 3)


def test_inventory_and_missing_tag_reconciliation_are_page_scoped() -> None:
    page_one = [_detection((1, 2, 3, 4), "fig_1", 1)]
    page_two = [_detection((5, 6, 7, 8), "fig_2", 2)]
    response = """<pages>
<page>Text <figure id="fig_1" description="A chart." label="legacy" />
<figure id="fig_1" description="Duplicate." />
<page_metadata>{"p": 1}</page_metadata></page>
<page>More text
<page_metadata>{"p": 2}</page_metadata></page>
</pages>"""

    reconciled = project_figures_to_llm_output(response, [page_one, page_two])

    assert reconciled.count('id="fig_1"') == 1
    assert reconciled.count('id="fig_2"') == 1
    assert "label=" not in reconciled
    assert 'bbox="1,2,3,4"' in reconciled
    assert 'bbox="5,6,7,8"' in reconciled
    assert reconciled.index('id="fig_2"') < reconciled.rindex("<page_metadata>")
    assert "fig_1" in format_figure_inventory(page_one)
    assert "Do not emit" in format_figure_inventory([])


def test_sequence_annotation_parser_preserves_figure_placeholder() -> None:
    tagged = (
        '<stem>Refer to <figure id="fig_4" description="A cube diagram." '
        'bbox="20,30,100,90" /> '
        "to answer.</stem>"
    )

    raw_text, spans = parse_xml_annotations(tagged)

    assert (
        '<figure id="fig_4" description="A cube diagram." bbox="20,30,100,90" />'
        in raw_text
    )
    assert spans == [
        {
            "start": 0,
            "end": len(raw_text),
            "label": "stem",
            "text": raw_text,
        }
    ]
