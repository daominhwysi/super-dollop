"""
adversarial_theory_generator.py — Modular Annotated Hard-Negative Generator.

Generates self-contained, modular pedagogical theory and administrative TOC/outlines
that can be:
1. Used as standalone background 'O' hard-negative documents in sequence labelling training.
2. Randomly sampled and spliced/pasted into exam XML documents or sliding-window chunks.

Subjects:
- Math (Toán học: Đại số, Giải tích, Hình học)
- Physics (Vật lí)
- English (Tiếng Anh: TOEIC / Ngữ pháp / Từ vựng)

Output Invariants:
- Modular blocks wrapped in <!-- MODULE: <id> | type: <type> | subject: <subject> --> ... <!-- /MODULE -->
- All titles, headings, and unit markers wrapped in <section>...</section>
- 100% expository content (0 exam questions, 0 option labels A/B/C/D)
- Saved as annotated XML ({doc_id}.xml) and raw text ({doc_id}.md) in data/synthetic_hard_negative/
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import random
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

WORKSPACE_DIR = Path(__file__).resolve().parents[1]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from sequence_labelling.llm.agy_adapter import chat_with_agy


def strip_xml_to_raw_markdown(xml_text: str) -> str:
    """Strips sequence labelling XML tags and module comment markers to produce pure Markdown."""
    text = re.sub(r"<!--\s*/?MODULE\b.*?-->", "", xml_text)
    text = re.sub(r"</?(?:section|question_label|stem|option_label|option_text|explanation|stimulus|figure)\b[^>]*>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("artifacts/hard_negative_generation.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("adversarial_modular_generator")

SUBJECT_ALIAS_MAP: Dict[str, List[str]] = {
    "math": ["math_algebra", "math_geometry"],
    "physics": ["physics"],
    "english": ["toeic"],
    "toeic": ["toeic"],
}


# ==============================================================================
# 1. Curriculum Bank Loader
# ==============================================================================

class CurriculumBank:
    """Loads and samples directly from data/curriculum/*.json."""

    def __init__(self, curriculum_dir: Optional[Path] = None):
        self.curriculum_dir = curriculum_dir or (WORKSPACE_DIR / "data/curriculum")
        self.curricula: Dict[str, Dict[str, Any]] = {}
        self._load_all()

    def _load_all(self) -> None:
        if not self.curriculum_dir.exists():
            logger.warning(f"Curriculum directory '{self.curriculum_dir}' not found.")
            return

        for f in sorted(self.curriculum_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                self.curricula[f.stem] = data
            except Exception as e:
                logger.warning(f"Failed to load {f.name}: {e}")

        logger.info(f"Loaded {len(self.curricula)} curriculum files from {self.curriculum_dir}.")

    def _resolve_keys(self, target_subject: Optional[str] = None) -> List[str]:
        keys = list(self.curricula.keys())
        if not target_subject:
            return keys

        prefixes = SUBJECT_ALIAS_MAP.get(target_subject.lower(), [target_subject.lower()])
        filtered = [k for k in keys if any(k.startswith(p) for p in prefixes)]
        return filtered or keys

    def sample_unit_for_theory(self, target_subject: Optional[str] = None) -> Dict[str, Any]:
        """Samples a specific curriculum chapter and unit for modular theory generation."""
        keys = self._resolve_keys(target_subject)
        key = random.choice(keys)
        curr = self.curricula[key]
        subject = curr.get("subject", key)
        grade = curr.get("grade", 12)
        chapters = curr.get("chapters", [])

        if not chapters:
            return {"subject": subject, "grade": grade, "chapter": "Tổng quan", "unit": "Khái niệm chung", "details": ""}

        chapter = random.choice(chapters)
        units = chapter.get("units", [])
        if not units:
            return {
                "subject": subject,
                "grade": grade,
                "chapter": chapter.get("name", ""),
                "unit": "Lý thuyết trọng tâm",
                "details": "",
            }

        unit = random.choice(units)
        pts = unit.get("problem_types", [])
        details_list = [pt.get("details", "") for pt in pts if pt.get("details")]
        combined_details = "\n\n".join(details_list[:3])

        return {
            "curriculum_key": key,
            "subject": subject,
            "grade": grade,
            "chapter": chapter.get("name", ""),
            "unit": unit.get("name", ""),
            "details": combined_details,
            "problem_type_names": [pt.get("name", "") for pt in pts],
        }

    def sample_curriculum_for_toc(self, target_subject: Optional[str] = None) -> Dict[str, Any]:
        """Samples an entire curriculum subject tree for modular Table of Contents or Syllabus generation."""
        keys = self._resolve_keys(target_subject)
        key = random.choice(keys)
        curr = self.curricula[key]
        return {
            "curriculum_key": key,
            "subject": curr.get("subject", key),
            "grade": curr.get("grade", 12),
            "chapters": curr.get("chapters", []),
        }


# ==============================================================================
# 2. System Prompts (Annotated XML Output)
# ==============================================================================

MODULAR_THEORY_ANNOTATED_SYSTEM_PROMPT = """You are an elite academic textbook author and educational sequence-labelling expert.
Your goal is to generate modular educational theory material outputted DIRECTLY in XML-annotated format compatible with sequence labelling ground truth.

CRITICAL INVARIANTS:
1. MODULAR XML STRUCTURE:
   - The document MUST consist of 3 to 4 distinct, self-contained modules.
   - Each module MUST be clearly wrapped in XML comment boundary markers:
     <!-- MODULE: id="<unique_id>" type="<definition|theorem|formula_card|method_steps|pitfall_trap>" subject="<subject>" -->
     ... module content ...
     <!-- /MODULE -->
   - This ensures any module can be randomly extracted and pasted anywhere into other exam documents.

2. SEQUENCE LABELLING XML ANNOTATION TAGS:
   - ALL section headers, subsection titles, definitions/theorem titles, and method headings MUST be wrapped in:
     <section>### Tiêu đề...</section> or <section>## 1. Định nghĩa...</section>
   - All body text, formulas, condition lists, and explanations remain UNTAGGED (background 'O' class).
   - You MUST NEVER generate exam questions, quizzes, or multiple-choice options (NO <question_label>, NO <stem>, NO <option_label>, NO 'A.', 'B.', 'C.', 'D.').
   - For Math and Physics: Use standard LaTeX math ($...$, $$...$$) accurately.
   - For English: Provide comprehensive grammar rules, sentence structures, tenses, collocations, or vocabulary tables.

3. CLEAN XML FORMAT:
   - Output valid XML. Close every <section> tag properly.
   - Output the XML directly without markdown code fences (no ```xml or ```).
"""

MODULAR_TOC_ANNOTATED_SYSTEM_PROMPT = """You are a senior curriculum director and textbook editor for the Vietnamese Ministry of Education.
Your goal is to generate modular Tables of Contents (Mục lục), Syllabi (Kế hoạch dạy học / Phân phối chương trình), and Competency Outlines outputted DIRECTLY in XML-annotated format.

CRITICAL INVARIANTS:
1. MODULAR XML STRUCTURE:
   - The document MUST consist of 2 to 3 distinct, self-contained administrative modules.
   - Each module MUST be clearly wrapped in XML comment boundary markers:
     <!-- MODULE: id="<unique_id>" type="<table_of_contents|syllabus_table|competency_matrix>" subject="<subject>" -->
     ... module content ...
     <!-- /MODULE -->
   - This ensures any administrative block can be randomly extracted and pasted anywhere into other exam documents.

2. SEQUENCE LABELLING XML ANNOTATION TAGS:
   - ALL administrative headers, chapter titles, and table captions MUST be wrapped in:
     <section># MỤC LỤC CHI TIẾT</section> or <section>### Khung phân phối chương trình: Học kỳ I</section>
   - All table contents, lesson hour allocations, dot leaders (........ Trang XX), and competency criteria remain UNTAGGED (background 'O' class).
   - You MUST NEVER generate exam questions or multiple-choice questions.

3. CLEAN XML FORMAT:
   - Output valid XML. Close every <section> tag properly.
   - Output the XML directly without markdown code fences (no ```xml or ```).
"""


# ==============================================================================
# 3. Generator Class
# ==============================================================================

@dataclass
class GeneratedDocMetadata:
    doc_id: str
    category: str
    subject: str
    grade: int
    title: str
    modules_count: int
    words_count: int
    tokens_est: int
    char_count: int
    created_at: float = field(default_factory=time.time)


class ModularAdversarialGenerator:
    """Generates modular annotated XML for Category 1 (Theory) and Category 4 (TOC & Outlines)."""

    def __init__(self, model: str = "gemini-3.8-flash-low"):
        self.model = model
        self.curriculum_bank = CurriculumBank()

    def generate_theory_document(
        self,
        target_subject: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> Tuple[str, GeneratedDocMetadata]:
        sample = self.curriculum_bank.sample_unit_for_theory(target_subject=target_subject)
        subj_raw = sample["subject"]
        grade = sample["grade"]
        doc_id = doc_id or f"theory_{subj_raw}_{grade}_{int(time.time()*1000)%1000000:06d}"

        is_english = (subj_raw == "toeic" or "english" in str(target_subject).lower())
        if is_english:
            subj_title = "TIẾNG ANH (TOEIC & CHƯƠNG TRÌNH THPT)"
            domain_guide = (
                "- Đối với môn Tiếng Anh, tạo 3-4 module độc lập:\n"
                "  + Module 1: <section>### Grammar Focus: Quy tắc ngữ pháp trọng tâm</section> (cấu trúc câu, cách dùng, ví dụ minh họa)\n"
                "  + Module 2: <section>### Bảng thì / Động từ / Cấu trúc câu</section> (bảng Markdown so sánh các trường hợp)\n"
                "  + Module 3: <section>### Common Traps & Confusing Words</section> (các bẫy ngữ pháp thường gặp, cặp từ dễ nhầm lẫn)\n"
                "  + Module 4: <section>### Collocations & Phrasal Verbs</section> (cụm từ cố định và ngữ cảnh sử dụng)\n"
                "- Tuyệt đối KHÔNG tạo bài tập yêu cầu chọn đáp án A, B, C, D."
            )
        elif "math" in subj_raw:
            subj_title = "TOÁN HỌC (ĐẠI SỐ / HÌNH HỌC)"
            domain_guide = (
                "- Đối với môn Toán, tạo 3-4 module độc lập:\n"
                "  + Module 1: <section>### Định nghĩa và Khái niệm nền tảng</section> (phát biểu chính xác, điều kiện tồn tại)\n"
                "  + Module 2: <section>### Hệ thống Định lý & Công thức then chốt</section> (công thức LaTeX chuẩn xác $...$, $$...$$)\n"
                "  + Module 3: <section>### Chú ý & Bẫy kinh điển cần tránh</section> (các sai lầm về điều kiện biên, đổi dấu, miền xác định)\n"
                "  + Module 4: <section>### Phương pháp & Thuật toán giải quyết dạng bài</section> (quy trình giải theo từng bước Bước 1, Bước 2)\n"
                "- Tuyệt đối KHÔNG tạo bài tập yêu cầu học sinh giải."
            )
        else:
            subj_title = "VẬT LÍ"
            domain_guide = (
                "- Đối với môn Vật lí, tạo 3-4 module độc lập:\n"
                "  + Module 1: <section>### Định nghĩa hiện tượng và Bản chất vật lý</section> (khái niệm, đại lượng đặc trưng, đơn vị đo SI)\n"
                "  + Module 2: <section>### Định luật và Hệ thống công thức liên hệ</section> (công thức LaTeX chuẩn xác)\n"
                "  + Module 3: <section>### Cảnh báo sai lầm & Bẫy nhận thức</section> (nhầm lẫn giữa các đại lượng, sai lầm về hệ quy chiếu/đơn vị)\n"
                "  + Module 4: <section>### Quy trình phân tích và Phương pháp giải</section> (các bước lập luận vật lý)\n"
                "- Tuyệt đối KHÔNG tạo bài tập yêu cầu học sinh giải."
            )

        prompt_lines = [
            f"=== YÊU CẦU BIÊN SOẠN CÁC MODULE LÝ THUYẾT ĐỘC LẬP (ANNOTATED XML) ===",
            f"Môn học: {subj_title} (Lớp {grade})",
            f"Chương: {sample['chapter']}",
            f"Bài học / Chủ đề: {sample['unit']}",
            f"Chi tiết nội dung từ chương trình:",
            f"{sample['details'] or 'Định nghĩa, định lý, công thức, quy tắc ngữ pháp và phương pháp.'}",
            "",
            f"HƯỚNG DẪN CẤU TRÚC MODULAR:",
            domain_guide,
            "",
            f"YÊU CẦU ĐỊNH DẠNG XML BẮT BUỘC:",
            f"1. Mỗi module phải được bao quanh bởi: <!-- MODULE: id=\"{doc_id}_m<N>\" type=\"<type>\" subject=\"{subj_raw}\" --> ... <!-- /MODULE -->.",
            f"2. Mỗi tiêu đề module và tiểu mục bên trong BẮT BUỘC dùng thẻ <section>...</section>.",
            f"3. Nội dung diễn giải, công thức, bảng Markdown để tự do (không bọc tag, thuộc lớp 'O').",
            f"4. Tổng dung lượng: 3-4 module, khoảng 800 - 1400 từ.",
            f"5. Bắt đầu ngay bằng nội dung XML, không kèm lời mở đầu hoặc ```xml.",
        ]
        user_prompt = "\n".join(prompt_lines)

        logger.info(f"Generating Modular Theory '{doc_id}' for {subj_title} (Grade {grade})...")
        t0 = time.time()
        content, conv_id, usage = chat_with_agy(
            prompt=user_prompt,
            system=MODULAR_THEORY_ANNOTATED_SYSTEM_PROMPT,
            model=self.model,
            timeout=180.0,
            caller="modular_adversarial_generator",
        )
        dur = time.time() - t0

        content = content.strip()
        # Clean any accidental code block wrappers
        if content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z]*\n", "", content)
            content = re.sub(r"\n```$", "", content)
            content = content.strip()

        modules_found = len(re.findall(r"<!--\s*MODULE:", content))
        words = len(content.split())
        chars = len(content)
        tokens_est = usage.get("total_tokens", 0) if usage else int(words * 1.5)

        logger.info(f"✅ Generated Theory '{doc_id}' in {dur:.1f}s | {modules_found} modules | {words:,} words | ~{tokens_est:,} tokens")

        meta = GeneratedDocMetadata(
            doc_id=doc_id,
            category="theory",
            subject=subj_raw,
            grade=grade,
            title=f"{sample['chapter']} - {sample['unit']}",
            modules_count=modules_found,
            words_count=words,
            tokens_est=tokens_est,
            char_count=chars,
        )
        return content, meta

    def generate_toc_outline_document(
        self,
        target_subject: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> Tuple[str, GeneratedDocMetadata]:
        sample = self.curriculum_bank.sample_curriculum_for_toc(target_subject=target_subject)
        subj_raw = sample["subject"]
        grade = sample["grade"]
        doc_id = doc_id or f"toc_{subj_raw}_{grade}_{int(time.time()*1000)%1000000:06d}"

        chapter_summaries = []
        for ch in sample["chapters"]:
            ch_name = ch.get("name", "")
            unit_names = [u.get("name", "") for u in ch.get("units", [])]
            chapter_summaries.append(f"- {ch_name}: {', '.join(unit_names[:4])}")
        curriculum_text = "\n".join(chapter_summaries)

        is_english = (subj_raw == "toeic" or "english" in str(target_subject).lower())
        subj_name = "TIẾNG ANH (TOEIC)" if is_english else subj_raw.upper()

        prompt_lines = [
            f"=== YÊU CẦU BIÊN SOẠN CÁC MODULE MỤC LỤC & PHÂN PHỐI CHƯƠNG TRÌNH (ANNOTATED XML) ===",
            f"Môn học: {subj_name} (Lớp {grade})",
            f"Danh mục chương trình:",
            f"{curriculum_text}",
            "",
            f"TẠO 2-3 MODULE HÀNH CHÍNH ĐỘC LẬP:",
            f"1. Module 1: <section>## MỤC LỤC SÁCH GIÁO KHOA / CHUYÊN ĐỀ</section> (Số chương, tên bài, dấu chấm chấm và số trang Trang XX).",
            f"2. Module 2: <section>### KHUNG KẾ HOẠCH DẠY HỌC & PHÂN PHỐI CHƯƠNG TRÌNH</section> (Bảng Markdown: Tuần, Tiết PPCT, Tên bài học, Số tiết, Yêu cầu cần đạt).",
            f"3. Module 3: <section>### MA TRẬN CHUẨN KIẾN THỨC VÀ TIÊU CHÍ NĂNG LỰC</section> (Bảng phân bổ nhận thức và bảng kiểm).",
            "",
            f"YÊU CẦU ĐỊNH DẠNG XML BẮT BUỘC:",
            f"1. Mỗi module phải được bao quanh bởi: <!-- MODULE: id=\"{doc_id}_m<N>\" type=\"<type>\" subject=\"{subj_raw}\" --> ... <!-- /MODULE -->.",
            f"2. Tất cả tiêu đề BẮT BUỘC dùng thẻ <section>...</section>.",
            f"3. Tuyệt đối KHÔNG có câu hỏi trắc nghiệm hay bài tập giải.",
            f"4. Bắt đầu ngay bằng nội dung XML, không kèm lời mở đầu hoặc ```xml.",
        ]
        user_prompt = "\n".join(prompt_lines)

        logger.info(f"Generating Modular TOC/Outline '{doc_id}' for {subj_name} (Grade {grade})...")
        t0 = time.time()
        content, conv_id, usage = chat_with_agy(
            prompt=user_prompt,
            system=MODULAR_TOC_ANNOTATED_SYSTEM_PROMPT,
            model=self.model,
            timeout=180.0,
            caller="modular_adversarial_generator",
        )
        dur = time.time() - t0

        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z]*\n", "", content)
            content = re.sub(r"\n```$", "", content)
            content = content.strip()

        modules_found = len(re.findall(r"<!--\s*MODULE:", content))
        words = len(content.split())
        chars = len(content)
        tokens_est = usage.get("total_tokens", 0) if usage else int(words * 1.5)

        logger.info(f"✅ Generated TOC/Outline '{doc_id}' in {dur:.1f}s | {modules_found} modules | {words:,} words | ~{tokens_est:,} tokens")

        meta = GeneratedDocMetadata(
            doc_id=doc_id,
            category="toc_outline",
            subject=subj_raw,
            grade=grade,
            title=f"Mục lục & Đề cương - {subj_name} {grade}",
            modules_count=modules_found,
            words_count=words,
            tokens_est=tokens_est,
            char_count=chars,
        )
        return content, meta


# ==============================================================================
# 4. Batch Runner
# ==============================================================================

class ModularBatchRunner:
    """Coordinates resilient multi-worker generation across Math, Physics, English."""

    def __init__(
        self,
        output_dir: Path,
        model: str = "gemini-3.8-flash-low",
        concurrency: int = 2,
    ):
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generator = ModularAdversarialGenerator(model=model)
        self.concurrency = max(1, concurrency)
        self.manifest_path = self.output_dir / "manifest.json"
        self._lock = threading.Lock()
        self.manifest: List[Dict[str, Any]] = self._load_manifest()

    def _load_manifest(self) -> List[Dict[str, Any]]:
        if self.manifest_path.exists():
            try:
                return json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Error loading manifest: {e}")
        return []

    def _save_manifest_entry(self, entry: Dict[str, Any]) -> None:
        with self._lock:
            for i, item in enumerate(self.manifest):
                if item["doc_id"] == entry["doc_id"]:
                    self.manifest[i] = entry
                    break
            else:
                self.manifest.append(entry)

            tmp_path = self.manifest_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_path.replace(self.manifest_path)

    def is_completed(self, doc_id: str) -> bool:
        xml_file = self.output_dir / f"{doc_id}.xml"
        return xml_file.exists() and xml_file.stat().st_size > 500

    def generate_single_theory_task(self, idx: int, subject: str) -> None:
        doc_id = f"theory_{idx:03d}_{subject}"
        if self.is_completed(doc_id):
            logger.info(f"⏩ [Skip Existing] Theory {doc_id} already exists.")
            return

        try:
            xml_content, meta = self.generator.generate_theory_document(
                target_subject=subject,
                doc_id=doc_id,
            )
            xml_file = self.output_dir / f"{meta.doc_id}.xml"
            xml_file.write_text(xml_content, encoding="utf-8")

            # Also save raw text version with tags stripped
            md_file = self.output_dir / f"{meta.doc_id}.md"
            raw_text = strip_xml_to_raw_markdown(xml_content)
            md_file.write_text(raw_text, encoding="utf-8")

            entry = {
                "doc_id": meta.doc_id,
                "category": meta.category,
                "subject": meta.subject,
                "grade": meta.grade,
                "title": meta.title,
                "modules_count": meta.modules_count,
                "words": meta.words_count,
                "tokens_est": meta.tokens_est,
                "xml_file": str(xml_file.relative_to(WORKSPACE_DIR)),
                "md_file": str(md_file.relative_to(WORKSPACE_DIR)),
            }
            self._save_manifest_entry(entry)
        except Exception as e:
            logger.error(f"❌ Failed to generate modular theory {doc_id}: {e}", exc_info=True)

    def generate_single_toc_task(self, idx: int, subject: str) -> None:
        doc_id = f"toc_{idx:03d}_{subject}"
        if self.is_completed(doc_id):
            logger.info(f"⏩ [Skip Existing] TOC/Outline {doc_id} already exists.")
            return

        try:
            xml_content, meta = self.generator.generate_toc_outline_document(
                target_subject=subject,
                doc_id=doc_id,
            )
            xml_file = self.output_dir / f"{meta.doc_id}.xml"
            xml_file.write_text(xml_content, encoding="utf-8")

            md_file = self.output_dir / f"{meta.doc_id}.md"
            raw_text = strip_xml_to_raw_markdown(xml_content)
            md_file.write_text(raw_text, encoding="utf-8")

            entry = {
                "doc_id": meta.doc_id,
                "category": meta.category,
                "subject": meta.subject,
                "grade": meta.grade,
                "title": meta.title,
                "modules_count": meta.modules_count,
                "words": meta.words_count,
                "tokens_est": meta.tokens_est,
                "xml_file": str(xml_file.relative_to(WORKSPACE_DIR)),
                "md_file": str(md_file.relative_to(WORKSPACE_DIR)),
            }
            self._save_manifest_entry(entry)
        except Exception as e:
            logger.error(f"❌ Failed to generate modular TOC {doc_id}: {e}", exc_info=True)

    def run(
        self,
        num_theory: int = 200,
        num_toc: int = 50,
        subjects: Optional[List[str]] = None,
    ) -> None:
        subjects = subjects or ["math", "physics", "english"]
        logger.info(
            f"Starting Modular Batch Hard-Negative Generation: {num_theory} Theory + {num_toc} TOC/Outlines | "
            f"Subjects: {subjects} | Concurrency: {self.concurrency} | Target Dir: {self.output_dir}"
        )

        tasks = []
        for i in range(1, num_theory + 1):
            subj = subjects[(i - 1) % len(subjects)]
            tasks.append(("theory", i, subj))

        for i in range(1, num_toc + 1):
            subj = subjects[(i - 1) % len(subjects)]
            tasks.append(("toc", i, subj))

        total_tasks = len(tasks)
        logger.info(f"Total modular tasks scheduled: {total_tasks} ({num_theory} theory, {num_toc} TOC)")

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = []
            for kind, idx, subj in tasks:
                if kind == "theory":
                    f = executor.submit(self.generate_single_theory_task, idx, subj)
                else:
                    f = executor.submit(self.generate_single_toc_task, idx, subj)
                futures.append(f)

            done_count = 0
            for fut in concurrent.futures.as_completed(futures):
                done_count += 1
                try:
                    fut.result()
                except Exception as e:
                    logger.error(f"Worker task error: {e}")

                if done_count % 5 == 0 or done_count == total_tasks:
                    logger.info(f"📈 Overall Progress: {done_count}/{total_tasks} ({done_count/total_tasks*100:.1f}%)")

        logger.info(f"🎉 Modular Batch Generation Completed! Manifest saved at {self.manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Modular Annotated Adversarial Hard-Negative Generator")
    parser.add_argument("--output-dir", type=str, default="data/synthetic_hard_negative", help="Output directory")
    parser.add_argument("--num-theory", type=int, default=200, help="Number of theory documents")
    parser.add_argument("--num-toc", type=int, default=50, help="Number of TOC & Outline documents")
    parser.add_argument("--allowed-subjects", nargs="+", default=["math", "physics", "english"], help="Subjects")
    parser.add_argument("--concurrency", type=int, default=2, help="Concurrency level")
    parser.add_argument("--model", type=str, default="gemini-3.8-flash-low", help="Model name")
    args = parser.parse_args()

    out_path = Path(args.output_dir).resolve()
    runner = ModularBatchRunner(
        output_dir=out_path,
        model=args.model,
        concurrency=args.concurrency,
    )
    runner.run(
        num_theory=args.num_theory,
        num_toc=args.num_toc,
        subjects=args.allowed_subjects,
    )


if __name__ == "__main__":
    main()
