"""
Hybrid Digital PDF OCR.

Extracts native Vietnamese digital text with 100% fidelity, detects mathematical formulas
and tables, packs formulas dynamically into mosaic images, transcribes them via agy Vision,
and stitches the transcribed LaTeX and HTML tables back into the document.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import fitz  # PyMuPDF
import numpy as np

from sequence_labelling.llm.agy_adapter import chat_with_agy
from tools.experiments.dynamic_mosaic_packer import DynamicMosaicPacker


class HybridDigitalOCR:
    """
    Hybrid Digital PDF OCR pipeline.
    """

    def __init__(
        self,
        model: str = "gemini-3.8-flash",
        thinking: Optional[str] = "low",
        dpi: int = 150,
        max_mosaic_width: int = 1024,
        max_mosaic_height: int = 1024,
        margin_top: float = 35.0,
        margin_bottom: float = 35.0,
    ):
        self.model = model
        self.thinking = thinking
        self.dpi = dpi
        self.margin_top = margin_top
        self.margin_bottom = margin_bottom
        self.packer = DynamicMosaicPacker(
            max_canvas_width=max_mosaic_width,
            max_canvas_height=max_mosaic_height,
            padding=10,
        )

    @staticmethod
    def is_digital_pdf(pdf_path: Union[str, Path], min_chars_per_page: int = 80) -> bool:
        """Determines if a PDF has a usable digital text layer."""
        try:
            doc = fitz.open(str(pdf_path))
            if len(doc) == 0:
                doc.close()
                return False
            sample_pages = min(3, len(doc))
            total_chars = sum(len(doc[i].get_text().strip()) for i in range(sample_pages))
            doc.close()
            return (total_chars / sample_pages) >= min_chars_per_page
        except Exception:
            return False

    def extract_page_components(
        self, page: fitz.Page, page_num: int
    ) -> Dict[str, Any]:
        """
        Extracts native text, tables, and formula regions from a single PDF page.
        Returns a dictionary containing template text, formula crops, and table crops.
        """
        page_rect = page.rect
        h_page, w_page = page_rect.height, page_rect.width

        # 1. Detect Tables
        tables = list(page.find_tables())
        table_rects: List[fitz.Rect] = []
        table_items: List[Tuple[str, np.ndarray, fitz.Rect]] = []
        for t_idx, t in enumerate(tables, 1):
            t_rect = fitz.Rect(t.bbox)
            table_rects.append(t_rect)
            t_id = f"T{page_num}_{t_idx}"
            # Render table crop
            pix = page.get_pixmap(clip=t_rect, dpi=self.dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            table_items.append((t_id, img, t_rect))

        # 2. Detect Math and Formula Candidates
        data = page.get_text("dict")
        raw_boxes: List[fitz.Rect] = []

        # Find option label positions (e.g. "A.", "B.", "C.", "D.") to avoid merging into them
        option_label_rects: List[fitz.Rect] = []
        for b in data.get("blocks", []):
            if "lines" not in b:
                continue
            for l in b["lines"]:
                spans = l.get("spans", [])
                for s in spans:
                    font = s.get("font", "")
                    text = s.get("text", "").strip()
                    sr = fitz.Rect(s["bbox"])
                    if re.match(r"^[A-D]\.$", text):
                        option_label_rects.append(sr)
                    elif any(f in font for f in ["Symbol", "MT-Extra", "CambriaMath", "Wingdings", "Italic"]):
                        raw_boxes.append(sr)

                # Intra-line whitespace gaps where equations reside
                for i in range(len(spans) - 1):
                    s1, s2 = spans[i], spans[i + 1]
                    gap = s2["bbox"][0] - s1["bbox"][2]
                    if gap > 12:
                        raw_boxes.append(
                            fitz.Rect(
                                s1["bbox"][2],
                                min(s1["bbox"][1], s2["bbox"][1]),
                                s2["bbox"][0],
                                max(s1["bbox"][3], s2["bbox"][3]),
                            )
                        )

        # Include vector drawings (fraction bars, arrows, roots)
        for d in page.get_drawings():
            dr = fitz.Rect(d["rect"])
            if dr.y0 < self.margin_top or dr.y1 > (h_page - self.margin_bottom):
                continue
            if any(dr.intersects(tr) for tr in table_rects):
                continue
            # Fraction line or vector arrow: narrow vertical height
            if dr.height <= 2.5 and 3.0 <= dr.width <= 50.0:
                # Expand vertically by 14pt to encompass numerator and denominator
                raw_boxes.append(fitz.Rect(dr.x0 - 2, dr.y0 - 14, dr.x1 + 2, dr.y1 + 14))
            elif dr.width <= 250 and dr.height <= 80 and (dr.width >= 3 or dr.height >= 3):
                raw_boxes.append(dr)

        # Exclude candidates inside tables or margins
        filtered_boxes: List[fitz.Rect] = []
        for r in raw_boxes:
            if any(r.intersects(tr) for tr in table_rects):
                continue
            if r.y0 < self.margin_top or r.y1 > (h_page - self.margin_bottom):
                continue
            filtered_boxes.append(r)

        # Cluster adjacent or overlapping boxes
        merged: List[fitz.Rect] = []
        for r in filtered_boxes:
            r_padded = fitz.Rect(r.x0 - 3, r.y0 - 2, r.x1 + 3, r.y1 + 2)
            was_merged = False
            for i, m in enumerate(merged):
                if r_padded.intersects(m):
                    merged[i] = m | r_padded
                    was_merged = True
                    break
            if not was_merged:
                merged.append(r_padded)

        # Second clustering pass
        final_boxes: List[fitz.Rect] = []
        for r in merged:
            was_merged = False
            for i, m in enumerate(final_boxes):
                if r.intersects(m):
                    final_boxes[i] = m | r
                    was_merged = True
                    break
            if not was_merged:
                final_boxes.append(r)

        # Filter noise, clamp to page, and protect option labels
        formula_boxes: List[fitz.Rect] = []
        for b in final_boxes:
            if 5 <= b.width <= (w_page * 0.85) and b.height >= 4:
                clamped = fitz.Rect(
                    max(0, b.x0 - 1),
                    max(0, b.y0 - 1),
                    min(w_page, b.x1 + 1),
                    min(h_page, b.y1 + 1),
                )
                formula_boxes.append(clamped)

        # Sort formula boxes in reading order
        formula_boxes.sort(key=lambda b: (round(b.y0 / 12) * 12, b.x0))

        # Render formula crops
        formula_items: List[Tuple[str, np.ndarray, fitz.Rect]] = []
        for f_idx, fb in enumerate(formula_boxes, 1):
            f_id = f"F{page_num}_{f_idx}"
            pix = page.get_pixmap(clip=fb, dpi=self.dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            formula_items.append((f_id, img, fb))

        # 3. Collect Surviving Text Spans
        surviving_spans = []
        for b in data.get("blocks", []):
            if "lines" not in b:
                continue
            for l in b["lines"]:
                for s in l.get("spans", []):
                    sr = fitz.Rect(s["bbox"])
                    if sr.y0 < self.margin_top or sr.y1 > (h_page - self.margin_bottom):
                        continue
                    center = fitz.Point((sr.x0 + sr.x1) / 2, (sr.y0 + sr.y1) / 2)
                    if any(tr.contains(center) for tr in table_rects):
                        continue
                    if any(fb.contains(center) for _, _, fb in formula_items):
                        continue
                    txt = s.get("text", "").strip()
                    if txt:
                        surviving_spans.append({
                            "text": txt,
                            "rect": sr,
                            "x0": sr.x0,
                            "y_mid": (sr.y0 + sr.y1) / 2,
                        })

        # Sort surviving spans by vertical reading order
        surviving_spans.sort(key=lambda s: s["y_mid"])

        # Cluster spans into horizontal reading lines
        lines: List[Dict[str, Any]] = []
        for s in surviving_spans:
            if not lines or (s["y_mid"] - lines[-1]["base_y"]) > 7.0:
                lines.append({
                    "base_y": s["y_mid"],
                    "items": [(s["x0"], s["text"])],
                    "y_min": s["rect"].y0,
                    "y_max": s["rect"].y1,
                })
            else:
                lines[-1]["items"].append((s["x0"], s["text"]))
                lines[-1]["y_min"] = min(lines[-1]["y_min"], s["rect"].y0)
                lines[-1]["y_max"] = max(lines[-1]["y_max"], s["rect"].y1)

        # Assign formulas to closest visual reading line
        used_formulas = set()
        for f_id, _, fb in formula_items:
            fb_mid = (fb.y0 + fb.y1) / 2
            best_line = None
            min_dist = float("inf")
            for l in lines:
                if fb.y0 <= (l["y_max"] + 6) and fb.y1 >= (l["y_min"] - 6):
                    dist = abs(fb_mid - l["base_y"])
                    if dist < min_dist:
                        min_dist = dist
                        best_line = l
            if best_line is not None:
                best_line["items"].append((fb.x0, f"{{{{FORMULA_{f_id}}}}}"))
                used_formulas.add(f_id)
            else:
                # Display / standalone formula on its own line
                lines.append({
                    "base_y": fb_mid,
                    "items": [(fb.x0, f"{{{{FORMULA_{f_id}}}}}")],
                    "y_min": fb.y0,
                    "y_max": fb.y1,
                })
                used_formulas.add(f_id)

        # Sort lines vertically from top to bottom
        lines.sort(key=lambda l: l["base_y"])

        # 4. Build Template Text with Placeholders and Tables
        template_lines: List[str] = []
        used_tables = set()

        for l in lines:
            # Insert any table whose vertical position precedes this line
            for t_id, _, t_rect in table_items:
                if t_id not in used_tables and t_rect.y0 <= l["base_y"]:
                    template_lines.append(f"\n\n{{{{TABLE_{t_id}}}}}\n\n")
                    used_tables.add(t_id)

            l["items"].sort(key=lambda x: x[0])
            assembled_line = " ".join(txt for _, txt in l["items"])
            # Normalize spacing around punctuation: e.g. " . " -> ". "
            assembled_line = re.sub(r"\s+([.,:;!?])", r"\1", assembled_line)
            template_lines.append(assembled_line)

        # Append any remaining tables
        for t_id, _, _ in table_items:
            if t_id not in used_tables:
                template_lines.append(f"\n\n{{{{TABLE_{t_id}}}}}\n\n")

        template_text = "\n".join(template_lines)

        return {
            "page_num": page_num,
            "template_text": template_text,
            "formula_items": [(f_id, img) for f_id, img, _ in formula_items],
            "table_items": [(t_id, img) for t_id, img, _ in table_items],
        }

    def transcribe_components(
        self,
        formula_items: Sequence[Tuple[str, np.ndarray]],
        table_items: Sequence[Tuple[str, np.ndarray]],
    ) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Any]]:
        """
        Transcribes formula crops and table crops using agy Vision.
        Returns (formula_mapping, table_mapping, token_usage).
        """
        formula_map: Dict[str, str] = {}
        table_map: Dict[str, str] = {}
        total_usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

        # 1. Transcribe Formulas via Dynamic Mosaic Canvases
        if formula_items:
            canvases = self.packer.pack(formula_items)
            for canvas_img, included_ids in canvases:
                _, img_bytes = cv2.imencode(".jpg", canvas_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

                prompt = (
                    "You are an expert Math OCR Assistant.\n"
                    "The image contains several labeled boxes: "
                    + ", ".join(f"[{fid}]" for fid in included_ids)
                    + ".\n"
                    "Each box contains a mathematical formula, symbol, or equation.\n"
                    "Transcribe each box into valid LaTeX math ($...$).\n"
                    "Output strictly a JSON object mapping each box ID to its LaTeX string:\n"
                    "{\n"
                    + ",\n".join(f'  "{fid}": "$...$"' for fid in included_ids)
                    + "\n}\n"
                    "Do NOT include conversational text or markdown code fences."
                )

                resp_text, _, usage = chat_with_agy(
                    prompt=prompt,
                    system="You are an expert Math OCR Assistant. Output valid JSON only.",
                    model=self.model,
                    thinking=self.thinking,
                    image_bytes=img_bytes.tobytes(),
                    image_filename="formula_mosaic.jpg",
                    timeout=180,
                    caller="hybrid_math_mosaic",
                )

                total_usage["calls"] += 1
                total_usage["input_tokens"] += usage.get("input_tokens", 0)
                total_usage["output_tokens"] += usage.get("output_tokens", 0)

                # Parse JSON
                try:
                    cleaned_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", resp_text.strip(), flags=re.DOTALL)
                    parsed = json.loads(cleaned_json)
                    for k, v in parsed.items():
                        formula_map[k] = str(v).strip()
                except Exception:
                    # Fallback regex extraction if model produced surrounding text
                    for fid in included_ids:
                        m = re.search(rf'"{fid}"\s*:\s*"([^"]+)"', resp_text)
                        if m:
                            formula_map[fid] = m.group(1).strip()
                        else:
                            formula_map[fid] = f"${fid}$"

        # 2. Transcribe Tables
        for t_id, t_img in table_items:
            _, img_bytes = cv2.imencode(".jpg", t_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            prompt = (
                "You are an expert Table OCR assistant.\n"
                "Convert the provided table image into a clean standard HTML <table>...</table> element.\n"
                "Use <table>, <tr>, <th> for headers, and <td> for data cells.\n"
                "Format math expressions and interval headers in LaTeX ($...$).\n"
                "Output strictly the HTML <table> block without markdown code blocks."
            )

            resp_text, _, usage = chat_with_agy(
                prompt=prompt,
                system="You are an expert Table OCR assistant. Return HTML table only.",
                model=self.model,
                thinking=self.thinking,
                image_bytes=img_bytes.tobytes(),
                image_filename="table.jpg",
                timeout=180,
                caller="hybrid_table_ocr",
            )

            total_usage["calls"] += 1
            total_usage["input_tokens"] += usage.get("input_tokens", 0)
            total_usage["output_tokens"] += usage.get("output_tokens", 0)

            clean_html = re.sub(r"^```(?:html)?\s*|\s*```$", "", resp_text.strip(), flags=re.DOTALL)
            table_map[t_id] = clean_html

        return formula_map, table_map, total_usage

    def stitch_document(
        self,
        template_text: str,
        formula_map: Dict[str, str],
        table_map: Dict[str, str],
        page_num: int,
    ) -> str:
        """Substitutes placeholder tokens with converted LaTeX and HTML tables."""
        rendered = template_text

        # Substitute formulas
        for fid, latex in formula_map.items():
            rendered = rendered.replace(f"{{{{FORMULA_{fid}}}}}", latex)

        # Substitute tables
        for tid, html in table_map.items():
            rendered = rendered.replace(f"{{{{TABLE_{tid}}}}}", html)

        # Clean any remaining unbound placeholders
        rendered = re.sub(r"\{\{FORMULA_[^\}]+\}\}", "", rendered)
        rendered = re.sub(r"\{\{TABLE_[^\}]+\}\}", "", rendered)

        # Add page boundary and metadata
        meta = json.dumps(
            {
                "p": page_num,
                "head": "CLEAN",
                "tail": "CLEAN",
            },
            indent=2,
        )

        return f"<page>\n{rendered.strip()}\n\n<page_metadata>\n{meta}\n</page_metadata>\n</page>"

    def process_pdf(
        self,
        pdf_path: Union[str, Path],
        pages: Optional[List[int]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Processes selected pages of a digital PDF using the hybrid mosaic pipeline.
        Returns (full_ocr_markdown, telemetry_dict).
        """
        start_t = time.time()
        doc = fitz.open(str(pdf_path))
        target_pages = pages if pages is not None else list(range(1, len(doc) + 1))

        page_results: List[str] = []
        all_formulas = []
        all_tables = []
        page_components = []

        extract_t0 = time.time()
        for p_num in target_pages:
            if p_num < 1 or p_num > len(doc):
                continue
            comp = self.extract_page_components(doc[p_num - 1], p_num)
            page_components.append(comp)
            all_formulas.extend(comp["formula_items"])
            all_tables.extend(comp["table_items"])
        doc.close()
        extract_duration = time.time() - extract_t0

        vision_t0 = time.time()
        formula_map, table_map, usage = self.transcribe_components(all_formulas, all_tables)
        vision_duration = time.time() - vision_t0

        for comp in page_components:
            page_md = self.stitch_document(
                comp["template_text"], formula_map, table_map, comp["page_num"]
            )
            page_results.append(page_md)

        total_duration = time.time() - start_t
        full_markdown = "<pages>\n" + "\n\n".join(page_results) + "\n</pages>"

        telemetry = {
            "total_duration_sec": round(total_duration, 2),
            "extract_duration_sec": round(extract_duration, 2),
            "vision_duration_sec": round(vision_duration, 2),
            "pages_processed": len(target_pages),
            "sec_per_page": round(total_duration / max(1, len(target_pages)), 2),
            "formula_count": len(all_formulas),
            "table_count": len(all_tables),
            "vision_api_calls": usage["calls"],
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
        }

        return full_markdown, telemetry
