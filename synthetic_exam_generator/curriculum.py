import os
import re
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

try:
    from synthetic_exam_generator.deepseek_client import chat
except ImportError:
    from .deepseek_client import chat

# Display names for subjects to construct prompts
SUBJECT_DISPLAY = {
    "economics_law": "Kinh tế pháp luật",
    "geography": "Địa lý",
    "history": "Lịch sử",
    "math_algebra": "Toán Đại số",
    "math_geometry": "Toán hình học",
    "physics": "Vật lý",
    "chemistry": "Hóa học",
    "biology": "Sinh học",
    "english": "Tiếng Anh",
    "literature": "Ngữ văn",
    "toeic": "TOEIC English"
}

def get_curriculum_path(subject: str, grade: int, base_dir: Optional[Path | str] = None) -> Path:
    """Returns the path where the curriculum JSON file is stored."""
    if base_dir is not None:
        return Path(base_dir) / f"{subject}_{grade}.json"
    data_path = Path("data") / "curriculum" / f"{subject}_{grade}.json"
    if data_path.exists():
        return data_path
    output_path = Path("output") / "curriculum" / f"{subject}_{grade}.json"
    if output_path.exists():
        return output_path
    return data_path

def escape_outside_text(text: str) -> str:
    """Double-escapes backslashes that are not part of a valid JSON escape sequence."""
    res = []
    i = 0
    n = len(text)
    while i < n:
        char = text[i]
        if char == '\\':
            if i + 1 < n:
                next_char = text[i + 1]
                # If it's a valid JSON escape, keep it as is
                if next_char in ['"', '\\', '/', 'b', 'f', 'n', 'r', 't']:
                    res.append('\\')
                    res.append(next_char)
                    i += 2
                    continue
                # If it's a unicode escape \uXXXX, keep it
                elif next_char == 'u' and i + 5 < n and all(c in '0123456789abcdefABCDEF' for c in text[i+2:i+6]):
                    res.append(text[i:i+6])
                    i += 6
                    continue
            # Otherwise, double escape the single backslash
            res.append('\\\\')
            i += 1
        else:
            res.append(char)
            i += 1
    return "".join(res)

def fix_json_strings(raw_json: str) -> str:
    """
    Sanitizes unescaped characters in JSON string values to ensure successful parsing.
    - Isolates LaTeX blocks ($...$) and escapes single backslashes inside them.
    - Sanitizes text outside LaTeX blocks by escaping invalid escape sequences.
    - Replaces literal newlines/tabs with escaped equivalents.
    """
    def string_replacer(match):
        inner = match.group(1)
        
        parts = []
        last_idx = 0
        # Find LaTeX blocks non-greedily
        for m in re.finditer(r'\$[^\$]+\$', inner):
            # Process outside text
            outside_text = inner[last_idx:m.start()]
            outside_text = escape_outside_text(outside_text)
            parts.append(outside_text)
            
            # Process LaTeX block (double escape single backslashes)
            latex_text = m.group(0)
            latex_text = re.sub(r'(?<!\\)\\(?!\\)', r'\\\\', latex_text)
            parts.append(latex_text)
            
            last_idx = m.end()
            
        # Process trailing outside text
        outside_text = inner[last_idx:]
        outside_text = escape_outside_text(outside_text)
        parts.append(outside_text)
        
        result = "".join(parts)
        
        # Replace literal newlines and tabs
        result = result.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        
        return f'"{result}"'
        
    # Match double-quoted JSON strings (handles escaped quotes correctly)
    return re.sub(r'"((?:[^"\\]|\\.)*)"', string_replacer, raw_json, flags=re.DOTALL)

def clean_json_response(response: str) -> str:
    """Extracts JSON content from a response that might contain Markdown wrappers."""
    cleaned = response.strip()
    
    # Try searching for markdown codeblock
    markdown_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if markdown_match:
        return markdown_match.group(1).strip()
        
    # Fallback to finding first brace and last brace
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1:
        return cleaned[first_brace:last_brace + 1]
        
    return cleaned

def generate_curriculum(
    subject: str,
    grade: int,
    model: Optional[str] = None,
    thinking: Optional[bool] = None,
    provider: Optional[str] = None,
) -> dict:
    """
    Calls the LLM to generate a curriculum structured with chapters, units, and problem types (Dạng)
    for the specified subject and grade. Saves the result to output/curriculum/{subject}_{grade}.json.
    """
    subject_display = SUBJECT_DISPLAY.get(subject, subject)
    
    system_prompt = (
        "Bạn là một chuyên gia giáo dục Việt Nam, am hiểu sâu sắc về chương trình giáo dục phổ thông mới "
        "(GDPT 2018) và các đề thi tốt nghiệp, thi học sinh giỏi."
    )
    
    subject_guideline = ""
    if subject == "english":
        subject_guideline = """
YÊU CẦU ĐẶC BIỆT cho môn Tiếng Anh:
Chương trình học bắt buộc phải chứa các Chương và Dạng bài sau:
1. Phát âm và Trọng âm (Pronunciation & Word Stress), với dạng bài "pronunciation" và "stress".
2. Ngữ pháp và Từ vựng (Vocabulary & Grammar), với dạng bài "grammar_vocabulary".
3. Sửa lỗi sai và Kết hợp câu (Error Correction & Sentence Combination), với dạng bài "error_correction_combination".
4. Điền từ (Cloze Word), với dạng bài "cloze_word".
5. Điền câu (Cloze Sentence), với dạng bài "cloze_sentence".
6. Đọc hiểu (Reading Comprehension), với dạng bài "reading_comprehension".
"""
    elif subject == "literature":
        subject_guideline = """
YÊU CẦU ĐẶC BIỆT cho môn Ngữ văn:
Chương trình học bắt buộc phải chứa các Chương và Dạng bài sau:
1. Đọc hiểu văn bản (Reading Comprehension), với dạng bài "reading_comprehension_literature" (văn bản tự luận đọc hiểu).
2. Nghị luận xã hội (Social Argumentation Essay), với dạng bài "social_argumentation_essay" (~200 chữ).
3. Nghị luận văn học (Literary Analysis Essay), với dạng bài "literary_analysis_essay" (~600 chữ).
"""

    prompt = f"""Hãy biên soạn một chương trình học (curriculum) chi tiết cho môn học '{subject_display}' lớp {grade} của Việt Nam.
Chương trình học này phải được cấu trúc thành các Chương (Chapters), mỗi Chương có các Bài học (Units), và mỗi Bài học có các Dạng bài tập (Dạng - ở đây dịch là 'problem_types') cụ thể.
{subject_guideline}
Chương trình học này phải bao quát từ mức độ Nhận biết - Thông hiểu (NB_TH), Vận dụng (VD) đến Vận dụng cao (VDC).

Đối với mỗi Dạng bài tập (problem_type), bạn phải cung cấp:
- `id`: Mã dạng bài tập duy nhất dạng snake_case, ví dụ: `{subject}_{grade}_wave_light_wave_dang1`
- `name`: Tên dạng bài tập cụ thể bằng tiếng Việt (ví dụ: "Tính các đại lượng cơ bản của giao thoa khe Y-âng")
- `cognitive_level`: Mức độ nhận thức, chỉ nhận một trong ba giá trị: "NB_TH", "VD", hoặc "VDC"
- `details`: Nội dung lý thuyết ngắn gọn, công thức chính (sử dụng LaTeX $...$), phương pháp giải chi tiết.
- `examples`: Danh sách gồm 1-2 ví dụ câu hỏi minh họa tiêu biểu cho dạng đó.

Đầu ra bắt buộc phải tuân theo cấu trúc JSON sau đây:
{{
  "subject": "{subject}",
  "grade": {grade},
  "chapters": [
    {{
      "name": "Tên chương (ví dụ: Sóng)",
      "units": [
        {{
          "name": "Tên bài học (ví dụ: Giao thoa ánh sáng)",
          "problem_types": [
            {{
              "id": "Mã dạng bài (snake_case)",
              "name": "Tên dạng bài",
              "cognitive_level": "NB_TH | VD | VDC",
              "details": "Lý thuyết, công thức chính (LaTeX bọc trong $...$) và phương pháp giải...",
              "examples": [
                "Ví dụ câu hỏi 1...",
                "Ví dụ câu hỏi 2..."
              ]
            }}
          ]
        }}
      ]
    }}
  ]
}}

Yêu cầu định dạng và quy tắc cú pháp JSON cực kỳ quan trọng:
1. Chỉ xuất ra chuỗi JSON hợp lệ bọc trong markdown codeblock ```json ... ```.
2. KHÔNG tự ý ngắt dòng (newline) giữa chừng ở trong các chuỗi văn bản (ví dụ ở các trường name, details, examples). Mọi chuỗi văn bản JSON phải nằm hoàn toàn trên một dòng đơn, không được có ký tự ngắt dòng vật lý. Nếu muốn mô tả xuống dòng, hãy viết chuỗi '\\n' (đã double-escape).
3. Mọi công thức toán học/ký hiệu bắt buộc dùng LaTeX bọc trong $...$.
4. Tất cả các ký tự gạch chéo ngược (backslash '\\') trong chuỗi JSON (ví dụ trong các ký hiệu LaTeX như '\\lambda', '\\frac') bắt buộc phải được viết double-escaped thành '\\\\' để đảm bảo tính hợp lệ của chuỗi JSON (ví dụ: '\\\\lambda', '\\\\frac').
"""
    
    print(f"Generating curriculum for {subject_display} Grade {grade} using LLM...")
    chat_kwargs = {}
    if model:
        chat_kwargs["model"] = model
    if thinking is not None:
        chat_kwargs["thinking"] = thinking
    if provider:
        chat_kwargs["provider"] = provider
        
    response = chat(prompt=prompt, system=system_prompt, **chat_kwargs)
    cleaned_res = clean_json_response(response)
    fixed_res = fix_json_strings(cleaned_res)
    
    try:
        curriculum_data = json.loads(fixed_res)
        
        # Ensure directory exists
        file_path = get_curriculum_path(subject, grade)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(curriculum_data, f, ensure_ascii=False, indent=2)
            
        print(f"Successfully generated and saved curriculum to: {file_path}")
        return curriculum_data
    except Exception as e:
        print(f"Error parsing curriculum JSON response: {e}")
        # Print raw/fixed response for debugging in case of JSON parse failure
        print("Fixed response preview:")
        print(fixed_res[:1000])
        raise RuntimeError(f"Failed to generate valid curriculum JSON for {subject} Grade {grade}: {e}")

def load_curriculum(
    subject: str,
    grade: int,
    autogenerate: bool = True,
    model: Optional[str] = None,
    thinking: Optional[bool] = None,
    provider: Optional[str] = None,
) -> Optional[dict]:
    """
    Loads the curriculum JSON file from output/curriculum/{subject}_{grade}.json.
    If it doesn't exist and autogenerate is True, it triggers Stage 1 to generate the curriculum.
    """
    file_path = get_curriculum_path(subject, grade)
    
    if not file_path.exists():
        if autogenerate:
            try:
                return generate_curriculum(subject, grade, model=model, thinking=thinking, provider=provider)
            except Exception as e:
                print(f"Auto-generation of curriculum failed: {e}")
                return None
        else:
            return None
            
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_content = f.read()
            # Clean and fix just in case the stored file has unescaped characters
            fixed_content = fix_json_strings(raw_content)
            return json.loads(fixed_content)
    except Exception as e:
        print(f"Error loading curriculum from {file_path}: {e}")
        return None

def select_curriculum_path(
    curriculum: dict,
    chapter_filter: Optional[str] = None,
    unit_filter: Optional[str] = None,
    problem_type_filter: Optional[str] = None
) -> Optional[Tuple[dict, dict, dict]]:
    """
    Selects a random path (Chapter, Unit, Problem Type) from the curriculum,
    respecting any optional text filters (case-insensitive substring match).
    
    Returns a tuple of dicts: (chapter, unit, problem_type), or None if no match is found.
    """
    chapters = curriculum.get("chapters", [])
    if not chapters:
        return None
        
    # Apply chapter filter if present
    filtered_chapters = chapters
    if chapter_filter:
        c_filt = chapter_filter.strip().lower()
        filtered_chapters = [c for c in chapters if c_filt in c.get("name", "").lower()]
        
    if not filtered_chapters:
        return None
        
    # Shuffle chapters to ensure randomness
    random.shuffle(filtered_chapters)
    
    for chapter in filtered_chapters:
        units = chapter.get("units", [])
        if not units:
            continue
            
        # Apply unit filter if present
        filtered_units = units
        if unit_filter:
            u_filt = unit_filter.strip().lower()
            filtered_units = [u for u in units if u_filt in u.get("name", "").lower()]
            
        if not filtered_units:
            continue
            
        random.shuffle(filtered_units)
        
        for unit in filtered_units:
            pts = unit.get("problem_types", [])
            if not pts:
                continue
                
            # Apply problem type filter if present (checks both id and name)
            filtered_pts = pts
            if problem_type_filter:
                p_filt = problem_type_filter.strip().lower()
                filtered_pts = [
                    pt for pt in pts 
                    if p_filt in pt.get("name", "").lower() or p_filt in pt.get("id", "").lower()
                ]
                
            if not filtered_pts:
                continue
                
            # Select random problem type
            pt = random.choice(filtered_pts)
            return chapter, unit, pt
            
    return None

def map_cognitive_level_to_difficulty(level: str) -> str:
    """
    Maps cognitive level ('NB_TH', 'VD', 'VDC') to a randomly selected Difficulty string
    suitable for the old/existing difficulty system.
    """
    if level == "NB_TH":
        return random.choice(["recognize", "comprehend"])
    elif level == "VD":
        return random.choice(["low_application", "application"])
    elif level == "VDC":
        return "high_application"
    else:
        # Fallback
        return random.choice(["recognize", "comprehend", "low_application", "application", "high_application"])


def generate_all_curricula(model: Optional[str] = None, thinking: Optional[Any] = None, concurrency: int = 4, provider: Optional[str] = None):
    """Orchestrates concurrent curriculum generation for all subjects and grades [10-12]."""
    import concurrent.futures
    try:
        from synthetic_exam_generator.generator import Subject
    except ImportError:
        from .generator import Subject
    
    subjects = [s.value for s in Subject if s.value not in ["english", "literature"]]
    grades = [10, 11, 12]
    
    print("=" * 60)
    print(f"Orchestrating concurrent curriculum generation for all subjects and grades [10-12]")
    print(f"Model: {model}")
    print(f"Provider: {provider}")
    print(f"Thinking Level: {thinking}")
    print(f"Concurrency: {concurrency}")
    print(f"Subjects: {', '.join(subjects)}")
    print(f"Grades: {', '.join(map(str, grades))}")
    print("=" * 60)
    
    print("\n--- Starting Curriculum Generation (Concurrent) ---")
    
    tasks = []
    for subject in subjects:
        for grade in grades:
            tasks.append((subject, grade))
            
    def generate_single_curriculum_task(subj, grd):
        curr_path = get_curriculum_path(subj, grd)
        if curr_path.exists():
            try:
                with open(curr_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data and "chapters" in data and len(data["chapters"]) > 0:
                    print(f"[Curriculum Exists] Skipping Subject={subj}, Grade={grd} (found {curr_path.name})")
                    return True
            except Exception:
                pass

        try:
            print(f"[Curriculum Start] Subject={subj}, Grade={grd}")
            generate_curriculum(
                subject=subj,
                grade=grd,
                model=model,
                thinking=thinking,
                provider=provider
            )
            print(f"[Curriculum Success] Subject={subj}, Grade={grd}")
            return True
        except Exception as e:
            print(f"[Curriculum Error] Subject={subj}, Grade={grd}: {e}")
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(generate_single_curriculum_task, s, g): (s, g) for s, g in tasks}
        concurrent.futures.wait(futures.keys())
        
    print("\nCurriculum generation completed.")
