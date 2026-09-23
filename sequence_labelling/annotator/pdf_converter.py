import argparse
import base64
import html
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Dict, Any, Sequence, Union
from dotenv import load_dotenv
import cv2
import fitz  # PyMuPDF
import numpy as np
from openai import OpenAI

# Reconfigure stdout to use UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Load environment variables
workspace_dir = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=workspace_dir / ".env")

from sequence_labelling.config import (
    FIGURE_CLASS_NAMES,
    FIGURE_CONFIDENCE_THRESHOLD,
    FIGURE_DETECTION_ENABLED,
    FIGURE_INCLUDED_CLASS_IDS,
    FIGURE_MODEL_PATH,
    OCR_MODEL,
    OCR_PROVIDER,
    OCR_BATCH_SIZE,
    OCR_CONCURRENCY,
    get_provider_base_url,
    get_provider_api_key,
)
from sequence_labelling.annotator.figure_detector import (
    FigureDetection,
    RFDETRFigureDetector,
)


SYSTEM_PROMPT_LONG_CONTEXT = (
    "You are an expert Document OCR and Structural Layout Mining Assistant. "
    "Perform precise OCR on the provided document page images into clean, structured Markdown AND emit a compact page boundary JSON header.\n\n"
    "CRITICAL USABILITY RULES (Usability over Visual Reproduction):\n"
    "1. PAGES WRAPPER: Wrap the entire response in <pages> ... </pages>. Inside <pages>, wrap each individual page in <page> ... </page>.\n"
    "2. NO ARTIFICIAL SPACING OR VISUAL REPRODUCTION: Do NOT attempt to visually replicate physical layout using spaces, tabs, or multiple consecutive empty spaces. Do NOT use whitespace to simulate multi-column layouts, align numbers under gaps, or pad text to match physical margins. Prioritize clean, usable text over visual reproduction.\n"
    "3. SINGLE-COLUMN MERGING: If the input page has multiple columns, convert and merge them into a clean, single-column linear flow following logical reading order.\n"
    "4. STRICT HTML TABLES ONLY: Convert ALL tables (including data tables, option choice grids, matrices, and side-by-side structures) strictly to standard HTML <table>...</table> elements (e.g. <table><tr><th>...</th></tr><tr><td>...</td></tr></table>). NEVER use Markdown pipe tables (| col | col |).\n"
    "5. MARKDOWN & LATEX: Extract text, headings, and lists in standard Markdown. Convert all math formulas and equations to standard LaTeX ($...$ inline, $$...$$ block).\n"
    "6. PAGE METADATA HEADER: At the bottom of the page content (just before </page>), output a strict JSON block enclosed in <page_metadata> ... </page_metadata>.\n\n"
    "7. FIGURE ANNOTATIONS: RF-DETR has outlined general figures and placed a visible badge such as `FIGURE fig_3` on each one. At the figure's exact logical position in the page reading order, emit exactly one self-closing tag: <figure id=\"fig_3\" description=\"One concise sentence describing the figure and its educational meaning.\" />. Copy the supplied figure ID exactly. Describe graphs, diagrams, maps, and illustrations sufficiently for a question to reference them. Escape XML attribute characters. Do not add any other attributes, output the badge as ordinary OCR text, invent figure IDs, or omit a supplied ID.\n"
    "8. FIGURE TEXT: Preserve important visible labels, values, axes, legends, and captions in the figure description. Do not duplicate all internal figure text as unrelated body paragraphs. Variation tables are not figure-annotated; process them using the normal OCR/table rules.\n\n"
    "JSON Schema:\n"
    "{\n"
    "  \"p\": page_num,\n"
    "  \"head\": \"CLEAN\"|\"CONT_GROUP\"|\"CONT_THEORY\",\n"
    "  \"tail\": \"CLEAN\"|\"OPEN_GROUP\"|\"OPEN_THEORY\"|\"OPEN_STEM\"|\"OPEN_OPT\",\n"
    "  \"seq\": [[\"THEORY_START\", title], [\"STIM_START\", id], [\"Q_START\", num], [\"Q_END\", num]]\n"
    "}"
)






def prune_think_tags(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def normalize_batch_metadata(batch_text: str, start_page_num: int) -> str:
    """
    Ensures each page's <page_metadata> block in a batch response has the correct absolute PDF page number 'p'.
    start_page_num is 1-indexed (e.g. 7 for batch pages 7..12).
    """
    page_idx = 0

    def replace_meta(match):
        nonlocal page_idx
        meta_content = match.group(1)
        actual_p = start_page_num + page_idx
        page_idx += 1
        fixed_meta = re.sub(r'("p"\s*:\s*)\d+', r"\g<1>" + str(actual_p), meta_content)
        return f"<page_metadata>\n{fixed_meta}\n</page_metadata>"

    pattern = r"<page_metadata>\s*(.*?)\s*</page_metadata>"
    return re.sub(pattern, replace_meta, batch_text, flags=re.DOTALL)


def format_figure_inventory(detections: Sequence[FigureDetection]) -> str:
    """Give the vision model a machine-readable source of truth for badge IDs."""
    if not detections:
        return "RF-DETR figure inventory: none. Do not emit any <figure> tags for this page."
    items = ", ".join(
        f'{item.figure_id} (confidence={item.score:.3f})'
        for item in detections
    )
    return (
        "RF-DETR figure inventory for this page: "
        f"{items}. Emit each ID exactly once as a <figure ... /> tag."
    )


def _bbox_attribute(detection: FigureDetection) -> str:
    """Serialize an original rendered-page xyxy box for the OCR XML contract."""
    return ",".join(str(value) for value in detection.box)


def _fallback_figure_tag(detection: FigureDetection) -> str:
    return (
        f'<figure id="{detection.figure_id}" '
        'description="Detected figure; a vision description was not returned." '
        f'bbox="{_bbox_attribute(detection)}" />'
    )


def project_figures_to_llm_output(
    batch_text: str, page_figures: Sequence[Sequence[FigureDetection]]
) -> str:
    """Project trusted detector boxes into figure tags after vision OCR completes."""
    expected_detections = {
        item.figure_id: item for figures in page_figures for item in figures
    }
    expected_ids = set(expected_detections)
    seen_ids: set[str] = set()

    def canonicalize(match: re.Match[str]) -> str:
        attributes = match.group(1)
        id_match = re.search(r'\bid\s*=\s*(["\'])(.*?)\1', attributes, re.DOTALL)
        description_match = re.search(
            r'\bdescription\s*=\s*(["\'])(.*?)\1', attributes, re.DOTALL
        )
        if id_match is None:
            return ""
        figure_id = html.unescape(id_match.group(2)).strip()
        if figure_id not in expected_ids or figure_id in seen_ids:
            return ""
        seen_ids.add(figure_id)
        description = (
            html.unescape(description_match.group(2)).strip()
            if description_match is not None
            else "Detected figure; a vision description was not returned."
        )
        detection = expected_detections[figure_id]
        return (
            f'<figure id="{html.escape(figure_id, quote=True)}" '
            f'description="{html.escape(description, quote=True)}" '
            f'bbox="{_bbox_attribute(detection)}" />'
        )

    batch_text = re.sub(
        r"<figure\b([^>]*)/>", canonicalize, batch_text, flags=re.DOTALL
    )
    existing_ids = set(
        re.findall(r'<figure\b[^>]*\bid=["\']([^"\']+)["\'][^>]*/>', batch_text)
    )
    page_matches = list(re.finditer(r"<page\b[^>]*>(.*?)</page>", batch_text, re.DOTALL))
    if len(page_matches) != len(page_figures):
        missing = [
            _fallback_figure_tag(item)
            for figures in page_figures
            for item in figures
            if item.figure_id not in existing_ids
        ]
        if not missing:
            return batch_text
        insertion = "\n" + "\n".join(missing) + "\n"
        pages_end = re.search(r"</pages>\s*$", batch_text, re.IGNORECASE)
        offset = pages_end.start() if pages_end else len(batch_text)
        return batch_text[:offset].rstrip() + insertion + batch_text[offset:]

    updated = batch_text
    for page_index in range(len(page_matches) - 1, -1, -1):
        match = page_matches[page_index]
        missing = [
            item
            for item in page_figures[page_index]
            if item.figure_id not in existing_ids
        ]
        if not missing:
            continue
        page_content = match.group(1)
        insertion = "\n".join(_fallback_figure_tag(item) for item in missing) + "\n"
        metadata_match = re.search(r"<page_metadata>", page_content)
        offset = metadata_match.start() if metadata_match else len(page_content)
        page_content = page_content[:offset] + insertion + page_content[offset:]
        updated = updated[: match.start(1)] + page_content + updated[match.end(1) :]
    return updated


# Compatibility name for callers that used the first implementation.
ensure_batch_figure_tags = project_figures_to_llm_output


def load_few_shot_messages(example_dir: Path) -> List[Dict[str, Any]]:
    messages = []
    if not example_dir.exists():
        return messages

    # Check for numbered subdirectories first (e.g. 1/, 2/, ...)
    subdirs = sorted([d for d in example_dir.iterdir() if d.is_dir() and d.name.isdigit()], key=lambda x: int(x.name))
    if subdirs:
        for sdir in subdirs:
            out_file = sdir / "out.md"
            if not out_file.exists():
                out_file = sdir / "out_1.md"
            if not out_file.exists():
                continue

            in_files = sorted([f for f in sdir.glob("in_*.png")], key=lambda x: int(re.search(r"\d+", x.stem).group() if re.search(r"\d+", x.stem) else 0))
            if not in_files:
                continue

            try:
                with open(out_file, "r", encoding="utf-8") as f_out:
                    out_content = f_out.read()

                content_parts = [{"type": "text", "text": SYSTEM_PROMPT_LONG_CONTEXT}]
                for idx, in_file in enumerate(in_files):
                    page_num = idx + 1
                    with open(in_file, "rb") as f_in:
                        img_base64 = base64.b64encode(f_in.read()).decode("utf-8")
                    content_parts.append({"type": "text", "text": f"--- Document Page {page_num} ---"})
                    content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}})

                messages.append({"role": "user", "content": content_parts})
                messages.append({"role": "assistant", "content": out_content})
            except Exception as e:
                print(f"  [Warning] Failed to load multi-image OCR few-shot pair from {sdir}: {e}")

        if messages:
            print(f"  Loaded {len(messages) // 2} multi-image few-shot OCR example pair(s).")
            return messages

    # Legacy flat directory structure fallback (in_1.png, out_1.md ...)
    i = 1
    while True:
        in_file = example_dir / f"in_{i}.png"
        out_file = example_dir / f"out_{i}.md"
        if not (in_file.exists() and out_file.exists()):
            break
        try:
            with open(in_file, "rb") as f_in:
                img_base64 = base64.b64encode(f_in.read()).decode("utf-8")
            with open(out_file, "r", encoding="utf-8") as f_out:
                out_content = f_out.read()
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all text from this exam page image."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                ],
            })
            messages.append({"role": "assistant", "content": out_content})
        except Exception as e:
            print(f"  [Warning] Failed to load OCR few-shot pair {i}: {e}")
        i += 1

    if messages:
        print(f"  Loaded {len(messages) // 2} few-shot OCR example pair(s).")
    return messages


class PDFOCRConverter:
    """
    Renders PDF pages to high-resolution PNG images using PyMuPDF (fitz)
    and passes them to a Vision LLM API (MiniMax / DeepSeek Vision / Qwen-VL) to produce Markdown OCR.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        batch_size: Optional[int] = None,
        concurrency: Optional[int] = None,
        examples_dir: Optional[Union[str, Path]] = None,
        enable_figure_detection: Optional[bool] = None,
        figure_detector: Optional[RFDETRFigureDetector] = None,
    ):
        self.model = model or OCR_MODEL
        self.provider = provider or OCR_PROVIDER or "xah"
        self.batch_size = batch_size if batch_size is not None else OCR_BATCH_SIZE
        self.concurrency = concurrency if concurrency is not None else OCR_CONCURRENCY

        self.base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("LLM_BASE_URL")
            or get_provider_base_url(self.provider)
        )
        self.api_key = (
            api_key
            or get_provider_api_key(self.provider)
            or os.environ.get("OPENAI_API_KEY")
        )

        figure_detection_enabled = (
            FIGURE_DETECTION_ENABLED
            if enable_figure_detection is None
            else enable_figure_detection
        )
        self.figure_detector = figure_detector
        if self.figure_detector is None and figure_detection_enabled:
            self.figure_detector = RFDETRFigureDetector(
                model_path=FIGURE_MODEL_PATH,
                confidence_threshold=FIGURE_CONFIDENCE_THRESHOLD,
                class_names=FIGURE_CLASS_NAMES,
                included_class_ids=FIGURE_INCLUDED_CLASS_IDS,
            )
        self.last_figures: list[dict[str, Any]] = []
        self.client = None
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        # Load few-shot examples
        script_dir = Path(__file__).resolve().parent
        ocr_examples_dir = Path(examples_dir) if examples_dir else script_dir / "examples" / "ocr"
        self.few_shot_messages = load_few_shot_messages(ocr_examples_dir)

    def convert_pdf(
        self,
        pdf_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        dpi: int = 150,
        batch_size: Optional[int] = None,
        concurrency: Optional[int] = None,
        use_fallback: bool = False,
        progress_callback: Optional[Any] = None,
    ) -> str:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        effective_batch_size = batch_size if batch_size is not None else self.batch_size
        effective_concurrency = concurrency if concurrency is not None else self.concurrency

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF file '{pdf_path}': {e}")

        total_pages = len(doc)
        print(f"Opening PDF '{pdf_path.name}' ({total_pages} page(s))...")
        print(f"  [OCR Settings] Batch Size: {effective_batch_size} image(s)/batch, Concurrency: {effective_concurrency} parallel worker(s)")

        def notify_progress(stage: str, current: int, total: int, msg: str):
            if not progress_callback:
                return
            try:
                progress_callback(stage, current, total, msg)
            except TypeError:
                progress_callback(current, total, f"[{stage}] {msg}")

        notify_progress("Render", 0, total_pages, f"Rendering {total_pages} page(s)...")

        # Render and detect in absolute page order before concurrent OCR starts. This
        # makes figure IDs stable regardless of which OCR batch finishes first.
        page_images: List[str] = []
        page_figures: list[list[FigureDetection]] = []
        next_figure_number = 1
        active_detector = self.figure_detector
        for idx in range(total_pages):
            notify_progress("RF-DETR", idx + 1, total_pages, f"Detecting figures page {idx + 1}/{total_pages}")
            page = doc[idx]
            pix = page.get_pixmap(dpi=dpi)
            img_bytes = pix.tobytes("jpeg")
            numbered_detections: list[FigureDetection] = []
            if active_detector is not None:
                try:
                    image_array = cv2.imdecode(
                        np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if image_array is None:
                        raise ValueError("OpenCV could not decode the rendered PDF page")
                    detections = active_detector.detect(image_array)
                    numbered_detections = active_detector.number_page(
                        detections,
                        page_number=idx + 1,
                        first_figure_number=next_figure_number,
                    )
                    next_figure_number += len(numbered_detections)
                    if numbered_detections:
                        annotated = active_detector.draw(image_array, numbered_detections)
                        encoded, encoded_image = cv2.imencode(
                            ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 95]
                        )
                        if not encoded:
                            raise ValueError("OpenCV could not encode an annotated page")
                        img_bytes = encoded_image.tobytes()
                except Exception as error:
                    print(f"  [Figure Detection Warning] Disabled after page {idx + 1}: {error}")
                    active_detector = None
            page_figures.append(numbered_detections)
            page_images.append(base64.b64encode(img_bytes).decode("utf-8"))
        doc.close()
        self.last_figures = [
            detection.as_dict()
            for detections in page_figures
            for detection in detections
        ]
        if self.last_figures:
            print(f"  [Figure Detection] Annotated {len(self.last_figures)} figure(s) across {total_pages} page(s).")

        # Prepare balanced batches so page ranges are distributed more evenly across OCR calls.
        batches = []
        batch_sizes = []
        if total_pages > 0:
            batch_count = max(1, (total_pages + effective_batch_size - 1) // effective_batch_size)
            base_size, remainder = divmod(total_pages, batch_count)
            start_idx = 0
            for batch_id in range(batch_count):
                batch_len = base_size + (1 if batch_id < remainder else 0)
                end_idx = start_idx + batch_len
                batches.append((batch_id, start_idx, end_idx))
                batch_sizes.append(batch_len)
                start_idx = end_idx

        def process_single_batch(batch_info: tuple) -> tuple[int, str]:
            b_id, s_idx, e_idx = batch_info
            print(f"  [OCR Batch {b_id + 1}/{len(batches)}] Processing pages {s_idx + 1} to {e_idx}...")

            if self.client is None:
                raise RuntimeError(
                    f"Vision LLM API client is not configured for OCR model '{self.model}'. "
                    f"Please verify API key and provider configuration."
                )

            try:
                system_prompt = SYSTEM_PROMPT_LONG_CONTEXT
                content_parts = [{"type": "text", "text": system_prompt}]

                for idx in range(s_idx, e_idx):
                    page_num = idx + 1
                    content_parts.append(
                        {
                            "type": "text",
                            "text": (
                                f"--- Document Page {page_num} ---\n"
                                f"{format_figure_inventory(page_figures[idx])}"
                            ),
                        }
                    )
                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{page_images[idx]}"},
                        }
                    )

                start_ocr_t = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
                duration_ocr_sec = time.time() - start_ocr_t
                try:
                    from sequence_labelling.llm.token_tracker import log_response
                    log_response(
                        response,
                        model=self.model,
                        provider=self.provider,
                        caller="pdf_vision_ocr",
                        duration_sec=duration_ocr_sec,
                    )
                except Exception:
                    pass
                raw_result = response.choices[0].message.content
                batch_result = prune_think_tags(raw_result)
                batch_result = normalize_batch_metadata(batch_result, s_idx + 1)
                batch_result = project_figures_to_llm_output(
                    batch_result, page_figures[s_idx:e_idx]
                )
                return (b_id, batch_result)

            except Exception as e:
                raise RuntimeError(
                    f"Vision LLM API failed for batch {b_id + 1} (pages {s_idx + 1}-{e_idx}): {e}"
                ) from e

        # Run batches concurrently and print one progress line per completed batch.
        results = [None] * len(batches)
        if batches:
            max_workers = min(effective_concurrency, len(batches))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_batch = {executor.submit(process_single_batch, b): b for b in batches}
                completed_batches = 0
                completed_pages = 0
                for future in as_completed(future_to_batch):
                    b_id, batch_text = future.result()
                    results[b_id] = batch_text
                    completed_batches += 1
                    completed_pages += batch_sizes[b_id]
                    print(f"[OCR Batches] {completed_batches}/{len(batches)} completed ({completed_pages}/{total_pages} pages)")
                    notify_progress(
                        "OCR LLM",
                        completed_pages,
                        total_pages,
                        f"Batch {b_id + 1}/{len(batches)} done ({completed_pages}/{total_pages} pages)"
                    )

        clean_batches = []
        for batch_res in filter(None, results):
            cleaned_b = re.sub(r"^\s*<pages>\s*", "", batch_res, flags=re.IGNORECASE)
            cleaned_b = re.sub(r"\s*</pages>\s*$", "", cleaned_b, flags=re.IGNORECASE)
            clean_batches.append(cleaned_b.strip())

        combined_pages = "\n\n".join(clean_batches).strip()
        final_text = f"<pages>\n{combined_pages}\n</pages>"

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_text)
            print(f"OCR output saved to: {output_path}")

        return final_text

    def convert_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        limit: Optional[int] = None,
        force: bool = False,
    ) -> List[Path]:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

        pdf_files = list(input_dir.rglob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found recursively in '{input_dir}'.")
            return []

        if limit is not None:
            pdf_files = pdf_files[:limit]

        output_paths = []
        for file_idx, pdf_path in enumerate(pdf_files, start=1):
            print(f"[OCR Files] {file_idx}/{len(pdf_files)}: {pdf_path.name}")
            rel_path = pdf_path.relative_to(input_dir)
            out_rel_path = rel_path.with_suffix(".md")
            output_file_path = output_dir / out_rel_path

            if output_file_path.exists() and not force:
                print(f"Skipping existing output: {output_file_path}")
                output_paths.append(output_file_path)
                continue

            print(f"\nProcessing PDF: {pdf_path}")
            self.convert_pdf(pdf_path, output_path=output_file_path)
            output_paths.append(output_file_path)

        return output_paths



def main():
    parser = argparse.ArgumentParser(
        description="Recursively perform OCR on PDF exam files using Vision LLM / PyMuPDF fallback."
    )
    parser.add_argument(
        "--input", "-i", default="input", help="Input directory or single PDF file"
    )
    parser.add_argument(
        "--output", "-o", default="out", help="Output directory to save markdown OCR files"
    )
    parser.add_argument(
        "--limit", "-l", type=int, default=None, help="Limit number of PDF files to process"
    )
    parser.add_argument(
        "--test", "-t", action="store_true", help="Test mode (processes 1 PDF file)"
    )
    parser.add_argument(
        "--force", "-f", action="store_true", help="Force reprocessing even if output exists"
    )
    parser.add_argument(
        "--model", default=None, help="Vision LLM model identifier"
    )
    parser.add_argument(
        "--batch-size", type=int, default=OCR_BATCH_SIZE, help=f"Number of images per OCR batch request (default: {OCR_BATCH_SIZE})"
    )
    parser.add_argument(
        "--concurrency", type=int, default=OCR_CONCURRENCY, help=f"Number of parallel OCR batch requests (default: {OCR_CONCURRENCY})"
    )

    args = parser.parse_args()
    limit = 1 if args.test else args.limit

    converter = PDFOCRConverter(
        model=args.model,
        batch_size=args.batch_size,
        concurrency=args.concurrency
    )

    inp = Path(args.input)
    if inp.is_file() and inp.suffix.lower() == ".pdf":
        out = Path(args.output)
        if out.is_dir() or not out.suffix:
            out = out / f"{inp.stem}.md"
        converter.convert_pdf(inp, output_path=out)
    else:
        converter.convert_directory(args.input, args.output, limit=limit, force=args.force)


if __name__ == "__main__":
    main()
