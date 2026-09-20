import time
import uuid
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from collections import OrderedDict
from sequence_labelling.config import (
    OCR_MODEL,
    PARSER_MODEL,
    LINKER_MODEL,
    PARSER_PROVIDER,
    LINKER_PROVIDER,
    CHUNKER_TARGET_TOKENS,
    CHUNKER_MAX_TOKENS,
    CHUNKER_OVERLAP_PAGES,
)
from sequence_labelling.annotator.pdf_converter import PDFOCRConverter
from sequence_labelling.parser.long_parser.sequence_reconciler import (
    DocumentStateStack,
    extract_metadata_headers_from_markdown,
    reconcile_parser_chunk_results,
)
from sequence_labelling.parser.long_parser.greedy_chunker import (
    PAGE_SEPARATOR,
    build_chunk_plan,
    greedy_oversize_chunker,
)
from sequence_labelling.parser.long_parser.parser_agent_worker import ParserAgentWorker
from sequence_labelling.parser.long_parser.linking_agent import CompactGraphResolverAgent

class LongContextParserPipeline:
    """
    Complete Pipeline Orchestrator for Long-Context Document Parsing (Azozo Engine v2.0).
    Orchestrates OCR -> State Machine Reconstruction -> Passage-Locked Chunking -> Parser Swarm -> Patch Graph Linker.
    """
    def __init__(
        self,
        ocr_model: Optional[str] = None,
        parser_model: Optional[str] = None,
        linker_model: Optional[str] = None,
        parser_provider: Optional[str] = None,
        linker_provider: Optional[str] = None,
        batch_size: int = 3,
        concurrency: int = 5,
        target_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        overlap_pages: Optional[int] = None,
        use_det_anchor: bool = False,
        parser_version: str = "legacy",
    ):
        self.ocr_model = ocr_model or OCR_MODEL
        self.parser_model = parser_model or PARSER_MODEL
        self.linker_model = linker_model or LINKER_MODEL
        self.parser_provider = parser_provider or PARSER_PROVIDER
        self.linker_provider = linker_provider or LINKER_PROVIDER
        self.batch_size = batch_size
        self.concurrency = concurrency
        self.target_tokens = target_tokens if target_tokens is not None else CHUNKER_TARGET_TOKENS
        self.max_tokens = max_tokens if max_tokens is not None else CHUNKER_MAX_TOKENS
        self.overlap_pages = max(0, overlap_pages if overlap_pages is not None else CHUNKER_OVERLAP_PAGES)

    @staticmethod
    def _normalize_text(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip()).lower()

    @staticmethod
    def _question_signature(question: Dict[str, Any]) -> str:
        q_num = str(question.get("question_number") or "").strip()
        q_num_match = re.search(r"\d+", q_num)
        if q_num_match:
            return f"num:{q_num_match.group(0)}"

        stem = question.get("stem") or ""
        stem_norm = LongContextParserPipeline._normalize_text(stem)
        options = question.get("options") or []
        option_sig = "|".join(
            f"{opt.get('label','')}:{LongContextParserPipeline._normalize_text(opt.get('text',''))}"
            for opt in options[:4]
        )
        return f"fallback:{stem_norm[:140]}|{option_sig[:140]}"

    @staticmethod
    def _question_quality(question: Dict[str, Any]) -> int:
        stem_len = len((question.get("stem") or ""))
        option_text_len = sum(len(opt.get("text") or "") for opt in question.get("options") or [])
        explanation_bonus = 20 if (question.get("explanation") or "").strip() else 0
        option_bonus = 5 * min(len(question.get("options") or []), 10)
        answer_bonus = 3 if question.get("correct_answer") else 0
        return stem_len + option_text_len + explanation_bonus + option_bonus + answer_bonus

    def _merge_duplicate_questions(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Resolve duplicated questions caused by overlap chunks (same question index/content).
        Keeps the most complete representation and preserves first-seen ordering.
        """
        buckets: Dict[str, Dict[str, Any]] = OrderedDict()

        for q in questions:
            key = self._question_signature(q)
            existing = buckets.get(key)
            if not existing:
                buckets[key] = dict(q)
                continue

            existing_quality = self._question_quality(existing)
            incoming_quality = self._question_quality(q)
            if incoming_quality <= existing_quality:
                continue

            # Keep better-quality question content for the same semantic key
            merged = existing
            merged["stem"] = q.get("stem") or existing.get("stem") or ""
            merged["options"] = q.get("options") or existing.get("options") or []
            if not merged.get("stimulus_id") and q.get("stimulus_id"):
                merged["stimulus_id"] = q.get("stimulus_id")
                merged["stimulus_text"] = q.get("stimulus_text", merged.get("stimulus_text", ""))
            if not merged.get("explanation") and q.get("explanation"):
                merged["explanation"] = q.get("explanation")
            if not merged.get("correct_answer") and q.get("correct_answer"):
                merged["correct_answer"] = q.get("correct_answer")
            if "chunk_index" not in merged and "chunk_index" in q:
                merged["chunk_index"] = q.get("chunk_index")

            buckets[key] = merged

        return list(buckets.values())

    @staticmethod
    def _merge_stimuli(stimuli: Dict[str, str]) -> Dict[str, str]:
        canonical_stimuli: Dict[str, str] = {}
        text_to_id: Dict[str, str] = {}
        id_alias: Dict[str, str] = {}

        for stim_id, stim_text in stimuli.items():
            norm = re.sub(r"\s+", "", (stim_text or "").strip())
            if not norm:
                continue
            existing_id = text_to_id.get(norm)
            if existing_id:
                id_alias[stim_id] = existing_id
                continue

            canonical_stimuli[stim_id] = stim_text
            text_to_id[norm] = stim_id

        if not id_alias:
            return canonical_stimuli

        remapped: Dict[str, str] = {}
        for stim_id, stim_text in canonical_stimuli.items():
            remapped[id_alias.get(stim_id, stim_id)] = stim_text

        # merge any aliases that may point to a different canonical id
        for alias_id, target_id in id_alias.items():
            if target_id in remapped and target_id != alias_id and alias_id in remapped:
                del remapped[alias_id]

        return remapped

    def _extract_pages_from_ocr(self, full_ocr_markdown: str) -> List[str]:
        """
        Extracts page blocks from OCR output using `<page>...</page>` sections.
        """
        if not full_ocr_markdown:
            return []

        # strip outer <pages> wrapper if present
        cleaned = re.sub(
            r"^\s*<\s*/?pages\s*>\s*",
            "",
            full_ocr_markdown.strip(),
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\s*<\s*/?\s*pages\s*>\s*$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

        page_blocks = re.findall(r"<page>(.*?)</page>", cleaned, flags=re.DOTALL | re.IGNORECASE)
        if page_blocks:
            return [block.strip() for block in page_blocks if block.strip()]

        # Fallback for unexpected OCR formats: treat as a single page
        return [cleaned] if cleaned else []

    def parse_pdf_long_context(
        self,
        pdf_path: str,
        doc_id: Optional[str] = None,
        progress_callback: Optional[Any] = None
    ) -> Dict[str, Any]:
        start_time = time.time()
        doc_id = doc_id or f"doc_{uuid.uuid4().hex[:8]}"

        # --- Phase 1: Parallel Vision LLM OCR & Metadata Mining ---
        if progress_callback:
            progress_callback(5, 100, "Đang chạy Vision OCR & khai thác Metadata trang...")

        converter = PDFOCRConverter(
            model=self.ocr_model,
            batch_size=self.batch_size,
            concurrency=self.concurrency
        )

        def ocr_callback(completed, total, msg):
            if progress_callback:
                prog = 5 + int((completed / max(1, total)) * 35)
                progress_callback(prog, 100, f"OCR (Minimax-M3): {msg}")

        full_ocr_markdown = converter.convert_pdf(pdf_path, progress_callback=ocr_callback)

        # --- Phase 2: Sequence Reconstruction & State Stack Machine ---
        if progress_callback:
            progress_callback(42, 100, "Xây dựng sơ đồ cấu trúc trang (DocumentStateStack)...")

        metadata_headers = extract_metadata_headers_from_markdown(full_ocr_markdown)
        state_stack = DocumentStateStack(doc_id=doc_id)
        completed_groups = state_stack.process_metadata_stream(metadata_headers)

        # Split full markdown into page dictionaries
        raw_pages = self._extract_pages_from_ocr(full_ocr_markdown)
        page_list = []
        global_offset = 0
        for idx, p_text in enumerate(raw_pages):
            if not p_text.strip():
                continue
            meta = metadata_headers[idx] if idx < len(metadata_headers) else {}
            text_body = re.sub(
                r"<page_metadata>.*?(?:</page_metadata>|(?=</page>)|$)",
                "",
                p_text,
                flags=re.DOTALL | re.IGNORECASE,
            ).strip()
            if page_list:
                global_offset += len(PAGE_SEPARATOR)
            page_start = global_offset
            global_offset += len(text_body)
            page_list.append({
                "p": meta.get("p", idx + 1),
                "text": text_body,
                "estimated_tokens": max(50, len(p_text.split())),
                "head": meta.get("head", "CLEAN"),
                "tail": meta.get("tail", "CLEAN"),
                "global_start": page_start,
                "global_end": global_offset,
            })

        # --- Phase 3: Passage-Locked Chunk Partitioning ---
        if progress_callback:
            progress_callback(48, 100, "Phân chia Chunk không cắt đoạn văn (Greedy Chunker)...")

        chunks = greedy_oversize_chunker(
            page_list,
            target_tokens=self.target_tokens,
            max_tokens=self.max_tokens,
            overlap_pages=self.overlap_pages,
        )
        chunk_plans = [
            build_chunk_plan(chunk_pages, chunk_index)
            for chunk_index, chunk_pages in enumerate(chunks)
        ]

        # --- Phase 4: Parallel Sequence Extraction (Parser Swarm) ---
        if progress_callback:
            progress_callback(52, 100, f"Đang chạy Parser Agent Swarm ({len(chunks)} Chunks)...")

        worker = ParserAgentWorker(model=self.parser_model, provider=self.parser_provider)
        chunk_results = [None] * len(chunks)
        max_workers = max(1, min(self.concurrency, len(chunks)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for plan in chunk_plans:
                chunk_idx = plan["chunk_index"]
                future = executor.submit(
                    worker.process_chunk,
                    plan["raw_chunk_text"],
                    chunk_index=chunk_idx,
                    page_ids=plan["page_ids"],
                    overlap_page_ids=plan["overlap_page_ids"],
                    page_offset_ranges=plan["page_offset_ranges"],
                )
                futures[future] = plan

            completed = 0
            for future in as_completed(futures):
                plan = futures[future]
                chunk_idx = plan["chunk_index"]
                try:
                    res = future.result()
                except Exception as e:
                    print(f"[Long Parser Warning] Parser worker failed for chunk {chunk_idx}: {e}")
                    res = {
                        **plan,
                        "raw_xml": "",
                        "spans": [],
                        "parse_status": "failed",
                        "parse_diagnostics": {
                            "attempts": worker.max_attempts,
                            "validation_errors": [str(e)],
                        },
                        "questions": [],
                        "stimuli": {},
                    }

                chunk_results[chunk_idx] = res
                completed += 1

                if progress_callback:
                    prog = 52 + int((completed / max(1, len(chunks))) * 33)
                    progress_callback(
                        prog,
                        100,
                        f"Parser Worker: Đã bóc tách Chunk {completed}/{len(chunks)}..."
                    )

        merge_result = reconcile_parser_chunk_results(
            [result for result in chunk_results if result is not None]
        )
        merged_questions = merge_result["structured_questions"]
        merged_stimuli = {
            stimulus_id: value["text"]
            for stimulus_id, value in merge_result["structured_stimuli"].items()
        }

        # --- Phase 5: Patch-Based Graph Resolver (Linker Agent) ---
        if progress_callback:
            progress_callback(86, 100, "Đang liên kết đồ thị ngữ cảnh & đáp án (Graph Resolver Agent)...")
        linker = CompactGraphResolverAgent(model=self.linker_model, provider=self.linker_provider)
        final_questions, final_stimuli = linker.resolve_and_apply_patches(
            questions=merged_questions,
            stimuli=merged_stimuli
        )

        duration = time.time() - start_time
        if progress_callback:
            progress_callback(100, 100, f"Hoàn tất xử lý Long-Context ({len(final_questions)} câu hỏi, {duration:.1f}s)!")

        return {
            "success": True,
            "doc_id": doc_id,
            "total_chunks": len(chunks),
            "questions_count": len(final_questions),
            "stimuli_count": len(final_stimuli),
            "document_groups": completed_groups,
            "metadata_blocks": len(metadata_headers),
            "questions": final_questions,
            "stimuli": final_stimuli,
            "canonical_text": merge_result["original_text"],
            "merged_xml": merge_result["merged_xml"],
            "merge_diagnostics": merge_result["diagnostics"],
            "source_fidelity_ok": merge_result["diagnostics"]["validation"].get(
                "source_fidelity_ok",
                False,
            ),
            "duration": duration,
            "raw_text": full_ocr_markdown
        }
