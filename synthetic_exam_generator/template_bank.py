"""
Comprehensive Combinatorial Template Bank for Vietnamese and Bilingual Exam Papers.
Contains rich collections of Section Headers, Explanation Prefixes, Barem & Scoring Rubrics,
and Answer Key Grids across THPTQG (GDPT 2018), DGNL, DGTD, Literature, English, and TOEIC.
"""

import random
from typing import List, Dict, Tuple, Optional, Any

# ==============================================================================
# 1. SECTION HEADERS BANK (92 Mẫu)
# ==============================================================================

SECTION_TEMPLATES: Dict[str, List[str]] = {
    # 1.1 THPT Quốc gia 2018 (30 mẫu)
    "thpt_mc": [
        "PHẦN I. Câu trắc nghiệm nhiều phương án lựa chọn",
        "PHẦN 1: CÂU HỎI TRẮC NGHIỆM 4 PHƯƠNG ÁN",
        "Phần I. Thí sinh trả lời từ câu 1 đến câu {num_q}. Mỗi câu hỏi chỉ chọn một phương án đúng nhất.",
        "Phần 1. Trắc nghiệm lựa chọn (Chọn một đáp án đúng trong 4 phương án A, B, C, D)",
        "A. PHẦN TRẮC NGHIỆM NHIỀU LỰA CHỌN",
        "PHẦN I: TRẮC NGHIỆM KHÁCH QUAN (Nhiều phương án lựa chọn)",
        "I. TRẮC NGHIỆM NHIỀU PHƯƠNG ÁN LỰA CHỌN (Mỗi câu trả lời đúng được 0,25 điểm)",
        "Phần 1. Trắc nghiệm bốn lựa chọn (A, B, C, D)"
    ],
    "thpt_tf": [
        "PHẦN II. Câu trắc nghiệm đúng sai",
        "PHẦN 2: TRẮC NGHIỆM ĐÚNG/SAI",
        "Phần II. Thí sinh trả lời từ câu 1 đến câu {num_q}. Trong mỗi ý a), b), c), d) ở mỗi câu, thí sinh chọn đúng hoặc sai.",
        "Phần 2. Đúng / Sai (Mỗi câu gồm 4 ý khẳng định)",
        "B. PHẦN TRẮC NGHIỆM ĐÚNG - SAI",
        "PHẦN II: CÂU HỎI ĐÚNG - SAI (Chọn Đ hoặc S cho mỗi mệnh đề)",
        "II. TRẮC NGHIỆM ĐÚNG SAI (Thí sinh chọn Đúng hoặc Sai cho từng lệnh hỏi)",
        "Phần 2. Câu hỏi lựa chọn Đúng/Sai theo chương trình mới"
    ],
    "thpt_sa": [
        "PHẦN III. Câu trắc nghiệm trả lời ngắn",
        "PHẦN 3: ĐIỀN KẾT QUẢ / TRẢ LỜI NGẮN",
        "Phần III. Thí sinh trả lời từ câu 1 đến câu {num_q}. Điền kết quả vào phiếu trả lời.",
        "Phần 3. Trắc nghiệm điền khuyết / Trả lời ngắn",
        "C. PHẦN CÂU HỎI TRẢ LỜI NGẮN",
        "PHẦN III: TRẢ LỜI NGẮN (Ghi kết quả dưới dạng số thập phân hoặc phân số tối giản)",
        "III. CÂU TRẮC NGHIỆM TRẢ LỜI NGẮN (Thí sinh tự điền đáp số)",
        "Phần 3. Câu hỏi tự điền đáp án ngắn"
    ],
    "thpt_essay": [
        "PHẦN IV. Tự luận",
        "PHẦN 4: CÂU HỎI TỰ LUẬN",
        "Phần IV. Thí sinh giải chi tiết các câu hỏi tự luận sau đây vào tờ giấy thi.",
        "D. PHẦN BÀI TẬP TỰ LUẬN",
        "PHẦN TỰ LUẬN (Trình bày lời giải chi tiết và vẽ hình nếu có)",
        "IV. TỰ LUẬN VÀ GIẢI BÀI TOÁN THỰC TẾ"
    ],

    # 1.2 Đánh giá Năng lực (ĐGNL) & Đánh giá Tư duy (ĐGTD) (20 mẫu)
    "dgnl_quant": [
        "PHẦN 1: TƯ DUY ĐỊNH LƯỢNG (TOÁN HỌC)",
        "PHẦN I. ĐỊNH LƯỢNG (TOÁN HỌC, XỬ LÝ SỐ LIỆU)",
        "Lĩnh vực: Tư duy định lượng và Khoa học dữ liệu",
        "Phần 1: Tư duy Toán học và Giải quyết vấn đề",
        "PHẦN I: TƯ DUY TOÁN HỌC (ĐHBK HÀ NỘI)"
    ],
    "dgnl_qual": [
        "PHẦN 2: TƯ DUY ĐỊNH TÍNH (NGỮ VĂN & NGÔN NGỮ)",
        "PHẦN II. ĐỊNH TÍNH (TIẾNG VIỆT VÀ VĂN HỌC)",
        "Lĩnh vực: Tư duy định tính và Cảm thụ văn học",
        "Phần 2: Sử dụng ngôn ngữ Tiếng Việt",
        "PHẦN II: TƯ DUY ĐỌC HIỂU VÀ PHÂN TÍCH VĂN BẢN"
    ],
    "dgnl_science": [
        "PHẦN 3: KHOA HỌC (VẬT LÝ, HÓA HỌC, SINH HỌC, LỊCH SỬ, ĐỊA LÝ)",
        "PHẦN III. TƯ DUY KHOA HỌC VÀ GIẢI QUYẾT VẤN ĐỀ",
        "Chủ đề: Vật lý ứng dụng và Đời sống",
        "Chuyên đề: Hóa học và Môi trường",
        "Chuyên đề: Sinh học hiện đại và Công nghệ di truyền",
        "Chủ đề Địa lí: Địa lí kinh tế - xã hội Việt Nam",
        "Chủ đề Lịch sử: Lịch sử thế giới và Lịch sử Việt Nam hiện đại",
        "Chuyên đề: Giáo dục kinh tế và Pháp luật"
    ],

    # 1.3 Ngữ văn GDPT 2018 & HSGQG (20 mẫu)
    "lit_reading": [
        "I. ĐỌC HIỂU (4,0 điểm)",
        "PHẦN I. ĐỌC HIỂU (4,0 điểm)",
        "Phần 1: Đọc hiểu văn bản (4,0 điểm)",
        "I. PHẦN ĐỌC HIỂU VĂN BẢN (Đọc đoạn trích sau và trả lời các câu hỏi)",
        "A. ĐỌC HIỂU (Ngữ liệu ngoài sách giáo khoa)",
        "PHẦN I. ĐỌC HIỂU VÀ CẢM THỤ TÁC PHẨM",
        "I. ĐỌC - HIỂU VĂN BẢN (Từ câu 1 đến câu 5)",
        "Phần I: Đọc hiểu (Đọc kỹ ngữ liệu và thực hiện các yêu cầu bên dưới)"
    ],
    "lit_writing_social": [
        "II. VIẾT (6,0 điểm) - 1. Nghị luận xã hội (2,0 điểm)",
        "1. Nghị luận xã hội (2,0 điểm)",
        "Câu 1 (2,0 điểm): Viết đoạn văn nghị luận xã hội khoảng 200 chữ",
        "Phần 2. Viết (Nghị luận xã hội - 2,0 điểm)",
        "II. PHẦN LÀM VĂN - Câu 1: Đoạn văn nghị luận xã hội",
        "1. Tạo lập đoạn văn nghị luận xã hội (2,0 điểm)"
    ],
    "lit_writing_literary": [
        "2. Nghị luận văn học (4,0 điểm)",
        "Câu 2 (4,0 điểm): Viết bài văn nghị luận văn học khoảng 600 chữ",
        "Phần 2. Viết (Nghị luận văn học - 4,0 điểm)",
        "II. PHẦN LÀM VĂN - Câu 2: Bài văn nghị luận văn học",
        "2. Viết bài văn so sánh, đánh giá tác phẩm văn học (4,0 điểm)",
        "Câu 2 (4,0 điểm): Nghị luận văn học chuyên sâu"
    ],
    "lit_hsg": [
        "PHẦN I. NGHỊ LUẬN XÃ HỘI (8,0 điểm)",
        "PHẦN II. NGHỊ LUẬN VĂN HỌC (12,0 điểm)",
        "Câu 1. Nghị luận xã hội (8,0 điểm) - Bàn luận về nhận định nhân sinh",
        "Câu 2. Nghị luận văn học (12,0 điểm) - Lý luận văn học và trải nghiệm tác phẩm"
    ],

    # 1.4 Tiếng Anh & TOEIC (22 mẫu)
    "english_directions": [
        "Mark the letter A, B, C, or D on your answer sheet to indicate the correct answer to each of the following questions.",
        "Read the following passage and mark the letter A, B, C, or D on your answer sheet to indicate the correct word or phrase that best fits each of the numbered blanks.",
        "Read the following passage and mark the letter A, B, C, or D on your answer sheet to indicate the correct answer to each of the questions.",
        "Mark the letter A, B, C, or D on your answer sheet to indicate the word whose underlined part differs from the other three in pronunciation.",
        "Mark the letter A, B, C, or D on your answer sheet to indicate the word that differs from the other three in the position of primary stress.",
        "Mark the letter A, B, C, or D on your answer sheet to indicate the word(s) CLOSEST in meaning to the underlined word(s).",
        "Mark the letter A, B, C, or D on your answer sheet to indicate the word(s) OPPOSITE in meaning to the underlined word(s).",
        "Mark the letter A, B, C, or D on your answer sheet to indicate the sentence that best combines each pair of sentences."
    ],
    "toeic_parts": [
        "PART 1 - PHOTOGRAPHS",
        "PART 2 - QUESTION-RESPONSE",
        "PART 3 - SHORT CONVERSATIONS",
        "PART 4 - SHORT TALKS",
        "PART 5 - INCOMPLETE SENTENCES",
        "PART 6 - TEXT COMPLETION",
        "PART 7 - READING COMPREHENSION",
        "Part 1: Photographs (Directions: For each question, choose the statement that best describes the picture)",
        "Part 2: Question-Response (Directions: You will hear a question and three responses)",
        "Part 3: Short Conversations (Directions: Listen to conversations between two or more people)",
        "Part 4: Short Talks (Directions: Listen to short talks given by a single speaker)",
        "Part 5: Incomplete Sentences (Directions: Select the best answer to complete each sentence)",
        "Part 6: Text Completion (Directions: Read the texts below and choose the best word or phrase for each blank)",
        "Part 7: Reading Comprehension (Directions: Read a variety of texts and answer the questions)"
    ]
}

# ==============================================================================
# 2. EXPLANATION & SOLUTION PREFIXES (40 Mẫu)
# ==============================================================================

INLINE_EXPLANATION_PREFIXES: List[str] = [
    "* Lời giải: ",
    "* Lời giải chi tiết: ",
    "* Hướng dẫn giải: ",
    "* Hướng dẫn giải chi tiết: ",
    "[LỜI GIẢI CHI TIẾT]: ",
    "➤ Hướng dẫn: ",
    "* Đáp án & Lời giải: ",
    "* Giải thích: ",
    "* Phân tích: ",
    "* Lời giải tham khảo: ",
    "* Đáp án chi tiết: ",
    "Gợi ý làm bài: ",
    "Giải: ",
    "Hướng dẫn: ",
    "* Gợi ý trả lời: ",
    "* Hướng dẫn chấm: "
]

METHOD_EXPLANATION_PREFIXES: List[str] = [
    "* Phương pháp giải: ",
    "* Cách giải: ",
    "* Kiến thức áp dụng: ",
    "* Phân tích và chọn đáp án: ",
    "* Các bước thực hiện: ",
    "* Chú ý lý thuyết: ",
    "* Nhận xét & Đánh giá: ",
    "* Phân tích đề bài: ",
    "* Công thức sử dụng: ",
    "* Dấu hiệu nhận biết: "
]

EXPLANATION_PREFIXES: List[str] = INLINE_EXPLANATION_PREFIXES + METHOD_EXPLANATION_PREFIXES

END_EXAM_SOLUTION_HEADERS: List[str] = [
    "# PHẦN II. ĐÁP ÁN VÀ LỜI GIẢI CHI TIẾT",
    "## ĐÁP ÁN VÀ HƯỚNG DẪN GIẢI CHI TIẾT",
    "### HƯỚNG DẪN CHẤM VÀ ĐÁP ÁN",
    "## ĐÁP ÁN - THANG ĐIỂM ĐỀ THI THỬ",
    "### BẢNG ĐÁP ÁN VÀ LỜI GIẢI THAM KHẢO",
    "## LỜI GIẢI VÀ BIỂU ĐIỂM CHẤM THI",
    "# HƯỚNG DẪN GIẢI VÀ ĐÁP ÁN CHÍNH THỨC",
    "## HƯỚNG DẪN CHẤM VÀ BIỂU ĐIỂM CHI TIẾT",
    "### BẢNG ĐÁP ÁN VÀ THANG ĐIỂM",
    "## ĐÁP ÁN VÀ LỜI GIẢI THAM KHẢO",
    "# BẢNG ĐÁP ÁN VÀ HƯỚNG DẪN GIẢI CHI TIẾT",
    "## GỢI Ý ĐÁP ÁN VÀ THANG ĐIỂM ĐỀ THI",
    "### ĐÁP ÁN CHI TIẾT VÀ BAREM CHẤM",
    "## HƯỚNG DẪN LÀM BÀI VÀ THANG ĐIỂM"
]

# ==============================================================================
# 3. BAREM & RUBRIC TABLE STYLES (18 Kiểu)
# ==============================================================================

BAREM_TABLE_HEADER_STYLES: List[Tuple[str, str]] = [
    # Format 1: Tiêu chuẩn 3 cột
    ("| Câu | Hướng dẫn giải / Đáp án | Điểm |", "| :-- | :---------------------- | :--- |"),
    ("| Câu hỏi | Nội dung yêu cầu cần đạt | Thang điểm |", "| :----- | :----------------------- | :--------- |"),
    ("| STT | Đáp án gợi ý | Điểm |", "| :-- | :----------- | :--- |"),
    ("| Câu | Nội dung hướng dẫn chấm | Điểm tối đa |", "| :-- | :---------------------- | :---------- |"),
    ("| Câu | Đáp án / Lời giải tóm tắt | Điểm số |", "| :-- | :------------------------ | :------ |"),
    ("| Phần / Câu | Nội dung trả lời | Biểu điểm |", "| :---------- | :---------------- | :-------- |"),

    # Format 2: Phân rã tiêu chí Văn / Tự luận (4 cột)
    ("| Phần | Câu | Nội dung yêu cầu cần đạt | Điểm |", "| :---- | :-- | :----------------------- | :--- |"),
    ("| Phần | Câu | Tiêu chí đánh giá | Điểm |", "| :---- | :-- | :---------------- | :--- |"),
    ("| TT | Câu hỏi | Yêu cầu chuẩn kiến thức | Thang điểm |", "| :-- | :------ | :---------------------- | :--------- |"),
    ("| STT | Phần | Gợi ý đáp án và tiêu chí chấm | Điểm |", "| :-- | :---- | :----------------------------- | :--- |"),

    # Format 3: Biểu điểm phân bước Toán/Lý/Hóa
    ("| Bước | Nội dung giải chi tiết | Điểm |", "| :--- | :--------------------- | :--- |"),
    ("| Ý | Lời giải và công thức | Điểm |", "| :- | :-------------------- | :--- |"),
    ("| Giai đoạn | Thao tác thực hiện / Phương trình | Thang điểm |", "| :-------- | :-------------------------------- | :--------- |"),
    ("| Bước | Yêu cầu lập luận / Tính toán | Điểm |", "| :--- | :--------------------------- | :--- |")
]

# ==============================================================================
# 4. ANSWER KEY GRIDS FORMATS (18 Kiểu)
# ==============================================================================

ANSWER_GRID_HEADER_PAIRS: List[List[str]] = [
    ["Câu", "Đ/A"],
    ["Câu", "Đáp án"],
    ["Câu hỏi", "Đ/A"],
    ["STT", "Đáp án"],
    ["Câu", "Key"],
    ["No.", "Ans"]
]

# ==============================================================================
# HELPER SAMPLING FUNCTIONS
# ==============================================================================

def get_random_section_header(category: str, rng: random.Random, num_q: int = 10) -> str:
    """Gets a random section header formatted with dynamic parameters."""
    templates = SECTION_TEMPLATES.get(category, SECTION_TEMPLATES["thpt_mc"])
    raw_template = rng.choice(templates)
    return raw_template.format(num_q=num_q)

def get_random_explanation_prefix(rng: random.Random, method_prefix_prob: float = 0.20) -> str:
    """Gets a random explanation prefix."""
    if rng.random() < method_prefix_prob:
        return rng.choice(METHOD_EXPLANATION_PREFIXES)
    return rng.choice(INLINE_EXPLANATION_PREFIXES)

def get_random_end_solution_header(rng: random.Random) -> str:
    """Gets a random end-of-exam solution section header."""
    return rng.choice(END_EXAM_SOLUTION_HEADERS)

def format_barem_table(
    questions_data: List[Tuple[int, str, str]],
    rng: random.Random,
    style_idx: Optional[int] = None
) -> str:
    """
    Builds a markdown scoring table barem using one of 18 distinct layout styles.
    questions_data: list of (q_num, expl_text, points_str)
    """
    if not questions_data:
        return ""

    if style_idx is None:
        style_idx = rng.randint(0, len(BAREM_TABLE_HEADER_STYLES) - 1)

    header_row, sep_row = BAREM_TABLE_HEADER_STYLES[style_idx % len(BAREM_TABLE_HEADER_STYLES)]
    body_rows = []

    is_4_col = "Phần" in header_row or "TT" in header_row

    for idx, (q_num, expl, pts) in enumerate(questions_data, start=1):
        clean_expl = expl.replace("\n", " ").strip() if expl else "Đáp án đúng"
        # Truncate very long explanations in table format for clean readability
        if len(clean_expl) > 180:
            clean_expl = clean_expl[:177] + "..."

        if is_4_col:
            part_name = f"Phần {1 if q_num <= 10 else 2}"
            body_rows.append(f"| {part_name} | {q_num} | {clean_expl} | {pts} |")
        else:
            body_rows.append(f"| {q_num} | {clean_expl} | {pts} |")

    return "\n".join([header_row, sep_row] + body_rows)

def format_answer_grid(
    questions_data: List[Tuple[int, str]],
    rng: random.Random,
    format_type: Optional[str] = None
) -> str:
    """
    Builds a markdown answer key grid using one of 18 distinct layout styles.
    questions_data: list of (q_num, answer_str)
    """
    if not questions_data:
        return ""

    if format_type is None:
        format_type = rng.choice(["horizontal", "vertical", "compact_text", "dash_separated"])

    # Format Type 1: Horizontal Matrix (10 items per row)
    if format_type == "horizontal":
        chunk_size = rng.choice([5, 8, 10])
        chunks = [questions_data[i:i+chunk_size] for i in range(0, len(questions_data), chunk_size)]
        table_lines = []
        sep_style = rng.choice(["---", ":---:", "---:"])
        for ch in chunks:
            header_row = "| " + " | ".join(str(q_num) for q_num, _ in ch) + " |"
            sep_row = "| " + " | ".join(sep_style for _ in ch) + " |"
            ans_row = "| " + " | ".join((ans if ans else "A").strip() for _, ans in ch) + " |"
            table_lines.extend([header_row, sep_row, ans_row, ""])
        return "\n".join(table_lines).strip()

    # Format Type 2: Vertical Multi-Column (2, 4, 5, or 6 columns)
    elif format_type == "vertical":
        col_pairs = rng.choice([2, 4, 5]) if len(questions_data) >= 15 else 2
        rows_count = (len(questions_data) + col_pairs - 1) // col_pairs

        header_pair = rng.choice(ANSWER_GRID_HEADER_PAIRS)
        headers = []
        for _ in range(col_pairs):
            headers.extend(header_pair)

        header_row = "| " + " | ".join(headers) + " |"
        sep_row = "| " + " | ".join(":---:" for _ in headers) + " |"

        body_rows = []
        for r in range(rows_count):
            row_cells = []
            for c in range(col_pairs):
                idx = c * rows_count + r
                if idx < len(questions_data):
                    q_num, ans = questions_data[idx]
                    row_cells.extend([str(q_num), (ans if ans else "A").strip()])
                else:
                    row_cells.extend(["", ""])
            body_rows.append("| " + " | ".join(row_cells) + " |")

        return "\n".join([header_row, sep_row] + body_rows)

    # Format Type 3: Compact Inline Text (e.g. 1.A  2.B  3.C)
    elif format_type == "compact_text":
        sep = rng.choice(["  ", "   ", " | ", "\t"])
        line_chunk = rng.choice([5, 8, 10])
        lines = []
        for i in range(0, len(questions_data), line_chunk):
            chunk = questions_data[i:i+line_chunk]
            dot_style = rng.choice([".", ":", "/", ")"])
            items = [f"{q_num}{dot_style}{ans.strip() if ans else 'A'}" for q_num, ans in chunk]
            lines.append(sep.join(items))
        return "\n".join(lines)

    # Format Type 4: Dash / Bullet format (e.g. Câu 1: A | Câu 2: B)
    else:
        line_chunk = rng.choice([4, 5])
        lines = []
        prefix = rng.choice(["Câu ", "C", "Q"])
        for i in range(0, len(questions_data), line_chunk):
            chunk = questions_data[i:i+line_chunk]
            items = [f"{prefix}{q_num}: {ans.strip() if ans else 'A'}" for q_num, ans in chunk]
            lines.append("   ".join(items))
        return "\n".join(lines)


# ==============================================================================
# 5. ADMINISTRATIVE & SCHOOL EXAM HEADERS (Untagged background 'O' lines)
# ==============================================================================

ADMIN_AUTHORITIES: List[str] = [
    "SỞ GIÁO DỤC VÀ ĐÀO TẠO HÀ NỘI",
    "SỞ GIÁO DỤC VÀ ĐÀO TẠO TP. HỒ CHÍ MINH",
    "SỞ GIÁO DỤC VÀ ĐÀO TẠO HẢI PHÒNG",
    "SỞ GIÁO DỤC VÀ ĐÀO TẠO ĐÀ NẴNG",
    "SỞ GIÁO DỤC VÀ ĐÀO TẠO NGHỆ AN",
    "SỞ GIÁO DỤC VÀ ĐÀO TẠO THANH HÓA",
    "SỞ GIÁO DỤC VÀ ĐÀO TẠO BẮC NINH",
    "SỞ GIÁO DỤC VÀ ĐÀO TẠO NAM ĐỊNH",
    "SỞ GIÁO DỤC VÀ ĐÀO TẠO VĨNH PHÚC",
    "BỘ GIÁO DỤC VÀ ĐÀO TẠO",
    "CỤM CÁC TRƯỜNG THPT CHUYÊN ĐỒNG BẰNG BẮC BỘ",
    "HỘI ĐỒNG BỘ MÔN TRUNG HỌC PHỔ THÔNG"
]

ADMIN_SCHOOLS: List[str] = [
    "TRƯỜNG THPT CHUYÊN HÀ NỘI - AMSTERDAM",
    "TRƯỜNG THPT CHUYÊN LÊ HỒNG PHONG",
    "TRƯỜNG THPT CHUYÊN LAM SƠN",
    "TRƯỜNG THPT CHUYÊN QUỐC HỌC HUẾ",
    "TRƯỜNG THPT CHUYÊN BẮC GIANG",
    "TRƯỜNG THPT CHUYÊN LƯƠNG VĂN TỤY",
    "TRƯỜNG THPT CHUYÊN PHAN BỘI CHÂU",
    "TRƯỜNG THPT NGUYỄN TẤT THÀNH",
    "TRƯỜNG THPT ĐOÀN THỊ ĐIỂM",
    "TRƯỜNG THPT NGUYỄN THỊ MINH KHAI",
    "TRƯỜNG THPT CHUYÊN ĐẠI HỌC SƯ PHẠM",
    "TRƯỜNG THPT CHU VĂN AN"
]

ADMIN_EXAM_TITLES: List[str] = [
    "KỲ THI TỐT NGHIỆP TRUNG HỌC PHỔ THÔNG NĂM {year}",
    "ĐỀ THI KHẢO SÁT CHẤT LƯỢNG LỚP {grade} NĂM HỌC {academic_year}",
    "KỲ THI CHỌN HỌC SINH GIỎI CẤP TỈNH NĂM HỌC {academic_year}",
    "ĐỀ THI THỬ TỐT NGHIỆP THPT QUỐC GIA LẦN {attempt} - NĂM {year}",
    "ĐỀ KIỂM TRA ĐÁNH GIÁ ĐỊNH KỲ HỌC KỲ I - NĂM HỌC {academic_year}",
    "ĐỀ THI THỬ THPT CHƯƠNG TRÌNH MỚI NĂM {year}",
    "ĐỀ KHẢO SÁT ĐÁNH GIÁ NĂNG LỰC HỌC SINH LỚP {grade}"
]


def generate_random_administrative_header(
    subject_display: str,
    grade: Any,
    rng: random.Random
) -> str:
    """
    Generates a realistic Vietnamese school/department administrative header.
    In downstream sequence labeling, these lines are UNTAGGED (Background 'O'),
    training the model to reject headers as questions or sections.
    """
    year = rng.choice([2024, 2025, 2026])
    academic_year = f"{year - 1} - {year}"
    attempt = rng.choice([1, 2, 3, "I", "II"])
    test_code = rng.choice(["101", "102", "204", "305", "408", "512", "624", "709"])
    duration = rng.choice([45, 50, 60, 90])

    authority = rng.choice(ADMIN_AUTHORITIES)
    school = rng.choice(ADMIN_SCHOOLS)
    raw_title = rng.choice(ADMIN_EXAM_TITLES)
    exam_title = raw_title.format(
        year=year,
        academic_year=academic_year,
        grade=grade or 12,
        attempt=attempt
    )

    style = rng.choice(["two_column", "stacked_formal", "compact_exam"])

    if style == "two_column":
        header_lines = [
            f"{authority:<42} {exam_title}",
            f"{school:<42} Môn thi: {subject_display.upper()} - Lớp {grade}",
            f"{'':<42} Thời gian làm bài: {duration} phút (không kể thời gian phát đề)",
            f"MÃ ĐỀ THI: {test_code}",
            "Họ và tên thí sinh: ................................................................",
            "Số báo danh: .......................................................................",
            "--------------------------------------------------------------------------------"
        ]
    elif style == "stacked_formal":
        header_lines = [
            authority,
            school,
            exam_title,
            f"BÀI THI: {subject_display.upper()} - LỚP {grade}",
            f"Thời gian làm bài: {duration} phút (không kể thời gian giao đề)",
            f"(Đề thi gồm 04 trang - Mã đề: {test_code})",
            "Họ, tên thí sinh: ........................................... Số báo danh: ............",
            ""
        ]
    else:  # compact_exam
        header_lines = [
            f"UBND TỈNH / THÀNH PHỐ - {authority}",
            f"{school} - MÃ ĐỀ: {test_code}",
            f"{exam_title} - MÔN: {subject_display.upper()}",
            f"Thời gian: {duration} phút",
            "--------------------------------------------------"
        ]

    return "\n".join(header_lines)
