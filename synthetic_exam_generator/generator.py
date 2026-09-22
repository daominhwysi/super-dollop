import json
import os
import re
import random
from enum import Enum
from typing import Dict, Any, List, Optional

try:
    from synthetic_exam_generator.deepseek_client import chat
    from synthetic_exam_generator.curriculum import (
        load_curriculum,
        select_curriculum_path,
        map_cognitive_level_to_difficulty,
    )
    from synthetic_exam_generator.parser import parse_question_xml
except ImportError:
    from .deepseek_client import chat
    from .curriculum import (
        load_curriculum,
        select_curriculum_path,
        map_cognitive_level_to_difficulty,
    )
    from .parser import parse_question_xml


class Subject(Enum):
    ECONOMICS_LAW = "economics_law"
    GEOGRAPHY = "geography"
    HISTORY = "history"
    MATH_ALGEBRA = "math_algebra"
    MATH_GEOMETRY = "math_geometry"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    ENGLISH = "english"
    LITERATURE = "literature"
    TOEIC = "toeic"


class QuestionType(Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"
    ORDERING = "ordering"
    GROUP_MULTIPLE_CHOICE = "group_multiple_choice"
    GROUP_SHORT_ANSWER = "group_short_answer"


class Difficulty(Enum):
    RECOGNIZE = "recognize"
    COMPREHEND = "comprehend"
    LOW_APPLICATION = "low_application"
    APPLICATION = "application"
    HIGH_APPLICATION = "high_application"


SUBJECT_DISPLAY: Dict[str, str] = {
    "economics_law": "Kinh tế & Pháp luật",
    "geography": "Địa lý",
    "history": "Lịch sử",
    "math_algebra": "Toán Đại số",
    "math_geometry": "Toán Hình học",
    "physics": "Vật lý",
    "chemistry": "Hóa học",
    "biology": "Sinh học",
    "english": "Tiếng Anh",
    "literature": "Ngữ văn",
    "toeic": "Tiếng Anh TOEIC",
}

QUESTION_TYPE_DISPLAY: Dict[str, str] = {
    "multiple_choice": "Trắc nghiệm",
    "true_false": "Đúng / Sai",
    "short_answer": "Trả lời ngắn",
    "ordering": "Sắp xếp",
    "group_multiple_choice": "Nhóm trắc nghiệm",
    "group_short_answer": "Nhóm trả lời ngắn",
}

DIFFICULTY_DISPLAY: Dict[str, str] = {
    "recognize": "Nhận biết",
    "comprehend": "Thông hiểu",
    "low_application": "Vận dụng thấp",
    "application": "Vận dụng",
    "high_application": "Vận dụng cao",
}


def generate_single_question(
    subject: Subject,
    grade: int,
    model: Optional[str] = None,
    thinking: Optional[bool] = None,
    question_type: Optional[QuestionType] = None,
    difficulty: Optional[Difficulty] = None,
    chapter_filter: Optional[str] = None,
    unit_filter: Optional[str] = None,
    problem_type_filter: Optional[str] = None,
    provider: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Tạo một câu hỏi đơn lẻ hoặc nhóm câu hỏi dựa trên cấu trúc chương trình học (Curriculum).
    Bảo đảm hướng dẫn làm bài ở mức đề thi không bao giờ lọt vào nội dung chi tiết của stem.
    """
    curr = None
    if subject not in [Subject.ENGLISH, Subject.LITERATURE]:
        curr = load_curriculum(
            subject.value, grade, autogenerate=True, model=model, thinking=thinking, provider=provider
        )

    chapter_name = "Tổng hợp"
    unit_name = "Tổng hợp"
    pt_id = problem_type_filter or "general"
    pt_name = problem_type_filter or "general"
    pt_level = "NB_TH"
    pt_details = ""
    pt_examples = []

    if curr:
        path = select_curriculum_path(
            curr, chapter_filter, unit_filter, problem_type_filter
        )
        if path:
            chapter, unit, pt = path
            chapter_name = chapter.get("name", "Tổng hợp")
            unit_name = unit.get("name", "Tổng hợp")
            pt_id = pt.get("id", pt_id)
            pt_name = pt.get("name", pt_name)
            pt_level = pt.get("cognitive_level", "NB_TH")
            pt_details = pt.get("details", "")
            pt_examples = pt.get("examples", [])
            if difficulty is None:
                diff_str = map_cognitive_level_to_difficulty(pt_level)
                difficulty = Difficulty(diff_str)

    if difficulty is None:
        difficulty = Difficulty.COMPREHEND

    if question_type is None:
        question_type = QuestionType.MULTIPLE_CHOICE

    is_group = question_type in [
        QuestionType.GROUP_MULTIPLE_CHOICE,
        QuestionType.GROUP_SHORT_ANSWER,
    ]

    # Chỉ thị ngăn chặn lặp lại hướng dẫn chung và quy định về nguồn trích dẫn
    strict_no_instructions_guideline = """
YÊU CẦU QUAN TRỌNG - KHÔNG ĐƯỢC CHÈN HƯỚNG DẪN CHUNG CỦA ĐỀ THI VÀO STEM HOẶC STIMULUS:
Tuyệt đối không viết các câu chỉ dẫn đề thi như:
- "Mark the letter A, B, C, or D on your answer sheet to indicate..."
- "Read the following passage and mark..."
- "Mark the letter A, B, C, or D to indicate the word that differs..."
- "Read the following piece of news..."
Các câu chỉ dẫn này đã được xử lý ở cấp độ phân mục (section level) của đề thi.
Thẻ <stem> của bạn CHỈ ĐƯỢC chứa nội dung câu hỏi/câu văn cụ thể. Thẻ <stimulus> chỉ chứa văn bản đọc hiểu/ngữ cảnh gốc.

QUY ĐỊNH BẮT BUỘC VỀ NGUỒN TRÍCH DẪN (STIMULUS CITATION):
- Đối với bài đọc/ngữ cảnh chung (<stimulus>): Mọi thông tin nguồn trích dẫn, tác giả, hoặc xuất xứ (như "(Adapted from CNN)", "(Adapted from The Guardian)", "(Nguồn: Báo Lao Động)", "(Theo britannica.com)", "(Trích: Chiếc thuyền ngoài xa, Nguyễn Minh Châu)") BẮT BUỘC PHẢI được đặt ở dòng cuối cùng BÊN TRONG thẻ <stimulus>...</stimulus>. Tuyệt đối không để nguồn trích dẫn ngoài thẻ <stimulus> hoặc đưa vào <stem>.
"""

    english_guidelines = ""
    if subject == Subject.ENGLISH:
        english_guidelines = f"""
Yêu cầu định dạng riêng cho bài thi tiếng Anh dạng '{pt_id}':
- Nếu là dạng phát âm (pronunciation): Thẻ <stem> nên để trống hoàn toàn hoặc chứa khoảng trắng " ". Các thẻ <option> chứa 4 từ tiếng Anh với phần cần gạch chân được bao bởi thẻ <u>...</u> (Ví dụ: f<u>oo</u>d).
- Nếu là dạng trọng âm (stress): Thẻ <stem> nên để trống hoàn toàn hoặc chứa khoảng trắng " ". Các thẻ <option> chứa 4 từ tiếng Anh để tìm từ có trọng âm khác biệt.
- Nếu là dạng tìm từ đồng nghĩa/trái nghĩa (closest_meaning, opposite_meaning): Thẻ <stem> chứa câu văn hoàn chỉnh có phần cần tìm từ đồng nghĩa/trái nghĩa được bao bởi thẻ <u>...</u>.
- Nếu là dạng sửa lỗi sai (error_correction): Thẻ <stem> chứa một câu văn tiếng Anh có lỗi sai. Các phương án lựa chọn là các phần trong câu đó.
- Nếu là dạng điền từ/điền câu (cloze_word, cloze_sentence, cloze_word_news, cloze_word_leaflet): Ngữ cảnh văn bản (stimulus) phải chứa các chỗ trống được đánh số cụ thể dạng '(26) <blank />', '(27) <blank />'. Cuối bài đọc bắt buộc có dòng nguồn trích dẫn dạng '(Adapted from ...)' BÊN TRONG thẻ <stimulus>.
- Nếu là dạng đọc hiểu (reading_comprehension, reading_comprehension_short, reading_comprehension_long): Thẻ <stimulus> chứa đoạn văn bản hoàn chỉnh và dòng cuối cùng kết thúc bằng nguồn trích dẫn '(Adapted from ...)' BÊN TRONG thẻ <stimulus>.
- Nếu là dạng ngữ pháp/từ vựng (grammar_vocabulary): Thẻ <stem> chứa câu văn tiếng Anh có một chỗ trống cần điền từ/cụm từ. Chỗ trống PHẢI được biểu diễn bằng thẻ XML <blank /> (TUYỆT ĐỐI KHÔNG dùng dấu gạch dưới ______ hay dấu chấm ...). Ví dụ: "She wishes she <blank /> harder for the final exam."
- Nếu là dạng hội thoại/giao tiếp (exchange): Thẻ <stem> chứa lượt nói của người A. Lượt trả lời của người B có chỗ trống PHẢI dùng thẻ <blank /> (TUYỆT ĐỐI KHÔNG dùng ______). Ví dụ: 'John: "I am thinking of taking a gap year." - Mary: "<blank />"'
- Nếu là dạng sắp xếp hội thoại (reordering_dialogue): Thẻ <stem> chứa lượt nói của cuộc đối thoại, mỗi dòng bắt đầu bằng ký tự thường kèm dấu chấm và tên người nói (ví dụ: "a. Tom: ...\nb. Mary: ..."). Các phương án <option> là các tổ hợp thứ tự các câu thoại (ví dụ: "a-b-c", "b-c-a").
- Nếu là dạng sắp xếp thư (reordering_letter): Thẻ <stem> chứa một bức thư hoàn chỉnh có các phần nội dung chính được chia thành các câu bắt đầu bằng ký tự thường kèm dấu chấm (ví dụ: "a. ...\nb. ..."), đồng thời PHẢI chứa câu chào mở đầu (ví dụ: "Dear Ms Smith,") và chữ ký kết thúc bức thư (ví dụ: "Yours sincerely,\nABC Bank"). Các phương án <option> là các tổ hợp thứ tự các câu thư (ví dụ: "d-a-c-b-e").
- Nếu là dạng sắp xếp đoạn văn (reordering_text): Thẻ <stem> chứa một đoạn văn học thuật/nghị luận ngắn có các câu được chia thành các phần bắt đầu bằng ký tự thường kèm dấu chấm (ví dụ: "a. ...\nb. ..."). Mỗi câu có độ dài hợp lý, mang văn phong học thuật (IELTS/THPT). Các phương án <option> là các tổ hợp thứ tự các câu văn (ví dụ: "e-c-d-b-a").
"""

    toeic_guidelines = ""
    if subject == Subject.TOEIC:
        toeic_guidelines = f"""
QUY CHUẨN ĐẶC BIỆT CHO BÀI THI TIẾNG ANH TOEIC (TOEIC PARTS 1 TO 7):
1. YÊU CẦU ĐỊNH DẠNG THEO TỪNG PART:
- Part 1 (Photographs): Thẻ <stem> để trống "" hoặc "Look at the photograph and choose the best statement:". Thẻ <option> gồm 4 câu mô tả bằng tiếng Anh (A), (B), (C), (D) về hành động của con người hoặc vị trí đồ vật/công trình.
- Part 2 (Question-Response): Thẻ <stem> chứa câu hỏi/phát biểu ngắn bằng tiếng Anh (ví dụ: "Where is the annual meeting being held?"). Thẻ <option> gồm ĐÚNG 3 PHƯƠNG ÁN (A), (B), (C).
- Part 3 (Short Conversations): Thẻ <stimulus> chứa đoạn hội thoại 2-3 người với tên người nói (ví dụ: "Man: ...\nWoman: ...\nMan: ..."). Nhóm câu hỏi gồm đúng 3 câu hỏi trắc nghiệm <question>.
- Part 4 (Short Talks): Thẻ <stimulus> chứa đoạn thông báo/tin nhắn thoại/thuyết trình một người nói. Nhóm câu hỏi gồm đúng 3 câu hỏi trắc nghiệm <question>.
- Part 5 (Incomplete Sentences): Thẻ <stem> chứa một câu văn doanh nghiệp có duy nhất một chỗ trống được biểu diễn bằng 5 dấu gạch dưới '_____'. 4 thẻ <option> là các phương án lựa chọn A, B, C, D.
- Part 6 (Text Completion): Thẻ <stimulus> chứa một văn bản doanh nghiệp hoàn chỉnh (thư từ, memo, thông báo) có 4 vị trí điền được đánh số rõ ràng dạng '_____ [131] _____', '_____ [132] _____', '_____ [133] _____', '_____ [134] _____'. Nhóm câu hỏi gồm 4 <question> tương ứng.
- Part 7 Single Passage (Chat Chains / Invoices / Notices / Memos):
  + Đối với Chat Chain: Thẻ <stimulus> chứa tin nhắn hội thoại có dấu mốc thời gian (ví dụ: "[9:02 AM] Marcus: ...\n[9:05 AM] Clara: ...").
  + Đối với Invoices/Tables: Thẻ <stimulus> chứa cấu trúc hóa đơn / bảng giá / biểu mẫu có tiêu đề và các dòng chi tiết.
  + Đối với Sentence Placement: Thẻ <stimulus> chứa văn bản có các đánh dấu vị trí [1], [2], [3], [4] và câu hỏi cuối là "In which of the positions marked [1], [2], [3], and [4] does the following sentence best belong?".
- Part 7 Double Passages: Thẻ <stimulus> chứa 2 văn bản liên kết (ví dụ: "[Document 1: Job Advertisement]\n...\n\n[Document 2: Email Application]\n..."). Nhóm câu hỏi gồm 5 <question>.
- Part 7 Triple Passages: Thẻ <stimulus> chứa 3 văn bản liên kết (ví dụ: "[Passage 1: Conference Program]\n...\n\n[Passage 2: Email Inquiry]\n...\n\n[Passage 3: Confirmation Notice]\n..."). Nhóm câu hỏi gồm 5 <question>.
"""

    literature_guidelines = ""
    if subject == Subject.LITERATURE:
        literature_guidelines = f"""
QUY CHUẨN ĐẶC BIỆT CHO ĐỀ THI MÔN NGỮ VĂN FORMAT 2025 - 2026 (CHƯƠNG TRÌNH GDPT 2018 & HSGQG):
1. QUY ĐỊNH BẮT BUỘC VỀ NGỮ LIỆU:
   - TUYỆT ĐỐI KHÔNG SỬ DỤNG các tác phẩm quen thuộc trong SGK cũ (như Truyện Kiều, Vợ nhặt, Vợ chồng A Phủ, Chiếc thuyền ngoài xa, Tây Tiến, Sóng, Người lái đò Sông Đà, Rừng xà nu, Việt Bắc, Đất Nước, Đồng chí...).
   - BẮT BUỘC 100% sử dụng ngữ liệu MỞ HOÀN TOÀN NẰM NGOÀI SÁCH GIÁO KHOA.
   - ĐA DẠNG HÓA THỂ LOẠI (Thơ hiện đại nhiều khổ ngắt dòng, Văn xuôi/Tản văn/Ký, Kịch bản sân khấu có lời thoại và chỉ dẫn sân khấu in nghiêng, Báo chí khoa học có số liệu %, Đối thoại triết học nhân sinh kèm cước chú học thuật ¹, ²).
   - CHÚ THÍCH TÁC GIẢ & CƯỚC CHÚ: Mọi thông tin chú thích tiểu sử tác giả (`* Chú thích: ...`), hoàn cảnh sáng tác, hoặc cước chú giải nghĩa (`¹ ...`, `² ...`) BẮT BUỘC nằm trọn vẹn BÊN TRONG thẻ <stimulus>...</stimulus>.

2. NẾU LÀ DẠNG ĐỌC HIỂU VĂN BẢN (reading_comprehension_literature...):
   - Thẻ <stimulus> chứa văn bản hoàn chỉnh kèm thông tin xuất xứ và chú thích tác giả.
   - Các thẻ <question> bên trong gồm ĐÚNG 5 CÂU HỎI phân hóa theo ma trận chuẩn 2025-2026:
     * Câu 1 (Nhận biết - 0.5đ): Xác định thể loại, ngôi kể, nhân vật, thể thơ, hoặc phương thức biểu đạt chính.
     * Câu 2 (Nhận biết/Thông hiểu - 0.75đ): Tìm các chi tiết, hình ảnh, từ ngữ trong văn bản hoặc gọi tên biện pháp tu từ.
     * Câu 3 (Thông hiểu - 1.0đ): Giải thích ý nghĩa của một hình ảnh/câu thơ hoặc phân tích tác dụng của biện pháp nghệ thuật.
     * Câu 4 (Thông hiểu/Vận dụng - 1.0đ): Phân tích tình cảm, thái độ của tác giả hoặc thông điệp/tư tưởng cốt lõi.
     * Câu 5 (Vận dụng - 0.75đ): Rút ra bài học cuộc sống, liên hệ bản thân hoặc bày tỏ quan điểm đồng tình/không đồng tình.
   - Mỗi câu hỏi: <stem> chứa câu hỏi, <answer> chứa gợi ý đáp án tự luận ngắn, <explanation> chứa biểu điểm chi tiết.

3. NẾU LÀ DẠNG ĐOẠN VĂN NGHỊ LUẬN XÃ HỘI HOẶC VIẾT ỨNG DỤNG (200 chữ):
   - Thẻ <stem> yêu cầu viết một đoạn văn (khoảng 200 chữ) bàn về một tư tưởng đạo lý/kỹ năng sống, hoặc viết Thư ngỏ, Bản kiến nghị, Bài giới thiệu sách, Bài phát biểu.
   - Thẻ <answer> chứa dàn ý đoạn văn (Mở đoạn, Thân đoạn: luận điểm & dẫn chứng, Đánh giá, Kết đoạn).
   - Thẻ <explanation> chứa biểu điểm (0.25đ hình thức, 1.25đ nội dung, 0.25đ sáng tạo, 0.25đ chính tả).

4. NẾU LÀ DẠNG BÀI VĂN NGHỊ LUẬN VĂN HỌC (600 chữ):
   - Nếu là phân tích tác phẩm: Thẻ <stem> yêu cầu viết bài văn (khoảng 600 chữ) phân tích một nét đặc sắc nội dung/nghệ thuật của văn bản Đọc hiểu.
   - Nếu là so sánh 2 tác phẩm (literary_comparative_essay_600): Thẻ <stem> cung cấp 2 đoạn trích thơ/văn xuôi của 2 tác giả kèm `* Chú thích:` về 2 tác giả và yêu cầu viết bài văn 600 chữ so sánh điểm tương đồng và nét độc đáo riêng.
   - Thẻ <answer> chứa dàn ý bài văn hoàn chỉnh 6 bước.
   - Thẻ <explanation> chứa biểu điểm chuẩn 4.0 điểm.

5. NẾU LÀ DẠNG HSGQG & CHUYÊN VĂN:
   - Thẻ <stimulus> chứa đoạn đối thoại triết học (*Dám hạnh phúc, Khắc kỷ...*) kèm cước chú học thuật `¹`, `²`.
   - Câu 1 (8.0đ NLXH): Bàn về mối quan hệ giữa các cặp phạm trù triết học/nhân sinh.
   - Câu 2 (12.0đ NLVH): Bàn luận về nhận định lý luận văn học chuyên sâu (C.S. Lewis, Roland Barthes, Milan Kundera...) bằng trải nghiệm văn học.
"""

    system_prompt = (
        "Bạn là chuyên gia xây dựng ngân hàng câu hỏi khảo thí chuyên nghiệp tại Việt Nam. "
        "Bạn luôn cung cấp câu hỏi sạch, chuẩn hóa dưới dạng cấu trúc XML và không bao giờ chèn thêm chỉ dẫn chung của đề thi vào nội dung câu hỏi."
    )

    xml_format = ""
    if is_group:
        xml_format = """
<group_question>
  <stimulus>Văn bản đọc hiểu hoặc đoạn thông tin chung...
(Adapted from Source Name)</stimulus>
  <question>
    <stem>Nội dung câu hỏi phụ thứ nhất...</stem>
    <option>Phương án A</option>
    <option>Phương án B</option>
    <option>Phương án C</option>
    <option>Phương án D</option>
    <answer>Đáp án (ví dụ: A hoặc B... hoặc ý trả lời chính nếu tự luận)</answer>
    <explanation>Giải thích lý do lựa chọn đáp án bằng tiếng Việt</explanation>
  </question>
  ... (tiếp tục với các câu hỏi tiếp theo trong nhóm) ...
</group_question>
"""
    else:
        xml_format = """
<question>
  <stem>Nội dung câu hỏi cụ thể...</stem>
  <option>Phương án A</option>
  <option>Phương án B</option>
  <option>Phương án C</option>
  <option>Phương án D</option>
  <answer>Đáp án đúng (ví dụ: A, B, C, hoặc D đối với trắc nghiệm, hoặc nội dung đáp án/dàn ý nếu tự luận)</answer>
  <explanation>Giải thích chi tiết bằng tiếng Việt</explanation>
</question>
"""

    prompt = f"""Hãy thiết kế một câu hỏi chất lượng cao cho môn học '{subject.value}', lớp {grade}.
- Loại câu hỏi: {question_type.value}
- Độ khó: {difficulty.value}
- Chương: {chapter_name}
- Bài học: {unit_name}
- Dạng lý thuyết/bài tập: {pt_name} (Mã dạng: {pt_id})
{f"- Công thức/Yếu tố lý thuyết áp dụng: {pt_details}" if pt_details else ""}
{f"- Ví dụ mẫu: {pt_examples}" if pt_examples else ""}

{strict_no_instructions_guideline}
{english_guidelines}
{toeic_guidelines}
{literature_guidelines}

Cấu trúc định dạng XML yêu cầu:
{xml_format}

Yêu cầu kỹ thuật:
1. Chỉ xuất ra cấu trúc XML hợp lệ nằm trong một khối mã duy nhất (như ```xml ... ```). Không viết thêm lời giới thiệu hoặc kết luận.
2. Không chèn các câu chỉ thị làm bài chung ở mức đề thi vào trường dữ liệu <stem> hoặc <stimulus>.
3. Đáp án trong thẻ <answer> của câu trắc nghiệm nhiều phương án lựa chọn (multiple_choice) chỉ chứa duy nhất một ký tự chữ cái viết hoa (A, B, C, hoặc D).
"""

    try:
        raw_response = chat(
            prompt=prompt,
            system=system_prompt,
            model=model or "deepseek-v4-flash",
            thinking=thinking,
            provider=provider,
        )
        parsed = parse_question_xml(raw_response)

        # Thử lại một lần nữa nếu phân tích cú pháp XML thất bại
        if parsed is None:
            raw_response = chat(
                prompt=prompt
                + "\nLƯU Ý: Phản hồi trước của bạn gặp lỗi phân tích cú pháp XML. Hãy đảm bảo thẻ mở và đóng hợp lệ.",
                system=system_prompt,
                model=model or "deepseek-v4-flash",
                thinking=thinking,
                provider=provider,
            )
            parsed = parse_question_xml(raw_response)

        if parsed:
            parsed["subject"] = subject.value
            parsed["grade"] = grade
            parsed["question_type"] = question_type.value
            parsed["difficulty"] = difficulty.value
            parsed["chapter"] = chapter_name
            parsed["unit"] = unit_name
            parsed["problem_type_id"] = pt_id
            parsed["problem_type_name"] = pt_name
            parsed["problem_type_level"] = pt_level
            return parsed
    except Exception as e:
        print(f"Error during generator execution: {e}")

    return None
