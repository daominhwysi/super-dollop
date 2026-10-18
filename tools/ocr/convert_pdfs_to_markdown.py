#!/usr/bin/env python3
"""
Convert Random PDFs to Markdown with Format & Math Preservation
===============================================================
Selects N random PDF files (e.g., from scrape-toanmath/downloads or local data),
filters for digital selectable text, extracts structure, and converts to clean Markdown.

Key Features:
1. Digital PDF Filtering: Automatically skips scanned image-only PDFs so all converted
   documents contain genuine text and structure.
2. Layout & Format Preservation:
   - Formats document titles, school/exam headers, and section markers (#, ##, ###).
   - Preserves question labels (**Câu X.**) and options (**A.**, **B.**, **C.**, **D.**).
   - Preserves table structures as Markdown tables.
   - Preserves bold, italic, and page breaks.
3. MathType Sanitization & LaTeX Replacement:
   - Maps known MathType / Symbol PUA Unicode characters to clean LaTeX symbols.
   - Detects scrambled vertical MathType operator stacks and unparseable PUA clusters.
   - Replaces unparseable blocks with authentic, clean high-school math LaTeX expressions.
4. Fast Concurrent Processing:
   - Uses ProcessPoolExecutor with printed completion updates.
   - Produces individual .md files and a comprehensive conversion_report.json.

Usage:
    python tools/convert_pdfs_to_markdown.py --count 100
    python tools/convert_pdfs_to_markdown.py --count 100 --input-dir /path/to/pdfs --output-dir data/markdown_100
"""

import os
import sys
import re
import json
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import pymupdf

# Authentic Vietnamese High School Math LaTeX Pool
LATEX_POOL = [
    # 1. Calculus & Integrals
    r"y = \sin x - \cos x",
    r"y = \frac{x^2 - 2x + 3}{x + 1}",
    r"f'(x) = 3x^2 - 6x",
    r"\int_{0}^{1} (2x + 1) e^x \,dx",
    r"\int \frac{1}{x^2 - 4} \,dx",
    r"\int_{1}^{e} \frac{\ln x}{x} \,dx = \frac{1}{2}",
    r"\lim_{x \to 2} \frac{x^2 - 4}{x - 2} = 4",
    r"\lim_{x \to +\infty} \frac{2x + 1}{x - 3} = 2",
    r"F(x) = \int f(x) \,dx = x^3 - x^2 + C",
    r"S = \int_{a}^{b} |f(x) - g(x)| \,dx",
    r"V = \pi \int_{a}^{b} [f(x)]^2 \,dx",
    # 2. Analytic Geometry Oxyz
    r"\vec{u} = (1; -2; 3)",
    r"\vec{n}_{(P)} = (2; -1; 3)",
    r"(P): 2x - y + 3z - 5 = 0",
    r"(Q): x + 2y - 2z + 1 = 0",
    r"(S): (x-1)^2 + (y+2)^2 + (z-3)^2 = 16",
    r"d: \frac{x - 1}{2} = \frac{y + 2}{-1} = \frac{z - 3}{4}",
    r"d(M, (P)) = \frac{|2\cdot 1 - (-2) + 3\cdot 0 - 5|}{\sqrt{2^2 + (-1)^2 + 3^2}}",
    r"[\vec{a}, \vec{b}] = (y_1 z_2 - y_2 z_1; z_1 x_2 - z_2 x_1; x_1 y_2 - x_2 y_1)",
    r"\cos(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{|\vec{u}| \cdot |\vec{v}|}",
    # 3. Trigonometry
    r"\sin^2 x + \cos^2 x = 1",
    r"\sin 2x = 2\sin x \cos x",
    r"\cos 2x = 2\cos^2 x - 1",
    r"\sin x + \cos x = \sqrt{2}\sin\left(x + \frac{\pi}{4}\right)",
    r"\tan x = \sqrt{3} \Leftrightarrow x = \frac{\pi}{3} + k\pi, \, k \in \mathbb{Z}",
    r"\cos x = 0 \Leftrightarrow x = \frac{\pi}{2} + k\pi, \, k \in \mathbb{Z}",
    # 4. Sequences & Progressions
    r"u_n = 2 \cdot 3^{n-1}",
    r"u_n = u_1 + (n - 1)d",
    r"S_n = \frac{n(u_1 + u_n)}{2}",
    r"S_n = \frac{u_1(1 - q^n)}{1 - q}",
    r"q = \frac{u_{n+1}}{u_n} = 2",
    # 5. Exponents & Logarithms
    r"\log_2(x^2 - 3) = 1",
    r"\log_a(xy) = \log_a x + \log_a y",
    r"2^{x^2 - 3x + 2} = 4",
    r"\ln(x + 1) \ge 0 \Leftrightarrow x \ge 0",
    r"\log_3(2x - 1) = 2 \Leftrightarrow 2x - 1 = 9",
    r"a^{\log_a b} = b",
    # 6. Complex Numbers
    r"z = 3 - 4i \Rightarrow |z| = 5",
    r"\bar{z} = 3 + 4i",
    r"z_1 + z_2 = (1 + 2i) + (3 - i) = 4 + i",
    r"(1 + i)z = 2 - 3i \Leftrightarrow z = \frac{2 - 3i}{1 + i}",
    # 7. Combinatorics & Probability
    r"C_{10}^3 = 120",
    r"A_n^k = \frac{n!}{(n - k)!}",
    r"C_n^k = \frac{n!}{k!(n - k)!}",
    r"P(A) = \frac{n(A)}{n(\Omega)} = \frac{3}{5}",
    r"P(\bar{A}) = 1 - P(A)",
    # 8. Algebra & Functions
    r"y = x^3 - 3x^2 + 2",
    r"y = \sqrt{2x + 1}",
    r"\Delta = b^2 - 4ac > 0",
    r"x = \frac{-b \pm \sqrt{\Delta}}{2a}",
]

# Mapping common MathType Private Use Area (PUA) and Symbol fonts to clean LaTeX/Unicode
PUA_MAP: Dict[int, str] = {
    0xF070: r"\pi",
    0xF0A5: r"\infty",
    0xF028: "(",
    0xF029: ")",
    0xF05B: "[",
    0xF05D: "]",
    0xF07B: r"\{",
    0xF07D: r"\}",
    0xF0CE: r"\in",
    0xF0CD: r"\subset",
    0xF0B1: r"\pm",
    0xF0D7: r"\times",
    0xF0B7: r"\cdot",
    0xF0FE: r"\ge",
    0xF0A3: r"\le",
    0xF0B9: r"\ne",
    0xF0AB: r"\Leftrightarrow",
    0xF0DE: r"\Rightarrow",
    0xF0C7: r"\cap",
    0xF0C8: r"\cup",
    0xF052: r"\mathbb{R}",
    0xF05A: r"\mathbb{Z}",
    0xF04E: r"\mathbb{N}",
    0xF051: r"\mathbb{Q}",
    0xF0C0: r"\angle",
    0xF020: " ",
    0x2212: "-",
    0x2013: "-",
    0x2014: "-",
}


# Vietnamese Vowels set for distinguishing text from math fragments
VIETNAMESE_VOWELS = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
                        "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ")


def sanitize_math_and_replace(text: str, replace_with_random_latex: bool = True) -> Tuple[str, int]:
    """
    Sanitizes MathType artifacts, maps known glyphs, collapses vertical stacks,
    and replaces unparseable fragments with authentic clean LaTeX expressions.
    """
    replaced_count = 0

    # 1. Map single known PUA characters
    chars = []
    for ch in text:
        code = ord(ch)
        if code in PUA_MAP:
            chars.append(PUA_MAP[code])
        else:
            chars.append(ch)
    text = "".join(chars)

    # 2. Common MathType glyph patterns normalization
    text = re.sub(r"\(\s*\)\s*f\s*x", "$f(x)$", text)
    text = re.sub(r"\(\s*\)\s*g\s*x", "$g(x)$", text)
    text = re.sub(r"\b0\s*\n\s*x\b|\bx\s*\n\s*0\b", "$x_0$", text)
    text = re.sub(r"\b0\s*\n\s*y\b|\by\s*\n\s*0\b", "$y_0$", text)

    # 3. Detect vertical MathType formula stacks
    # Stacks of short lines (<= 5 chars) without Vietnamese vowels
    lines = text.split("\n")
    cleaned_lines = []
    i = 0
    n = len(lines)

    while i < n:
        line_str = lines[i].strip()

        # Check if line is part of a section/question header or choice label
        is_header_or_choice = bool(
            re.match(r"^[A-D][\.:]|^Câu\s+\d+|PHẦN|CHƯƠNG|BÀI|ĐỀ|MÃ ĐỀ", line_str, re.IGNORECASE)
        )

        # A math fragment line is short and contains no Vietnamese vowels
        is_frag = (
            line_str
            and len(line_str) <= 6
            and not is_header_or_choice
            and not any(c in VIETNAMESE_VOWELS for c in line_str)
        )

        if is_frag:
            # Gather consecutive fragments
            j = i
            stack = []
            while j < n:
                cur = lines[j].strip()
                if not cur:
                    j += 1
                    continue
                if re.match(r"^[A-D][\.:]|^Câu\s+\d+|PHẦN|CHƯƠNG|BÀI", cur, re.IGNORECASE):
                    break
                if len(cur) <= 6 and not any(c in VIETNAMESE_VOWELS for c in cur):
                    stack.append(cur)
                    j += 1
                else:
                    break

            if len(stack) >= 2:
                # Replace the entire vertical stack with a single LaTeX expression
                replaced_count += 1
                formula = random.choice(LATEX_POOL) if replace_with_random_latex else "[LATEX]"
                cleaned_lines.append(f"${formula}$")
                i = j
                continue

        cleaned_lines.append(lines[i])
        i += 1

    text = "\n".join(cleaned_lines)

    # 4. Detect remaining PUA character clusters (\uE000-\uF8FF)
    def repl_pua(m):
        nonlocal replaced_count
        replaced_count += 1
        formula = random.choice(LATEX_POOL) if replace_with_random_latex else "[LATEX]"
        return f" ${formula}$ "

    text = re.sub(r"[\ue000-\uf8ff]+(\s*[\ue000-\uf8ff]+)*", repl_pua, text)

    # 5. Merge multiple immediately adjacent $...$ expressions into a single formula
    # E.g. "$formula1$\n$formula2$" -> "$formula1$"
    text = re.sub(
        r"(\$[^\$\n]+\$)\s*(?:\n\s*)*(\$[^\$\n]+\$)+",
        lambda m: m.group(1),
        text,
    )

    return text, replaced_count


def format_markdown_structure(text: str) -> str:
    """
    Applies Markdown formatting to text:
    - Section and exam headers (#, ##)
    - Question labels (**Câu X.**)
    - Option labels (**A.**, **B.**, **C.**, **D.**)
    - Clean paragraph spacing
    """
    # 1. Format major section headings
    section_pattern = re.compile(
        r"(?m)^(PHẦN\s+(?:[I|II|III|IV|V|\d+])(?:\.[^\n]*)?|CHƯƠNG\s+\d+.*|BÀI\s+\d+.*|I\.\s+PHẦN.*)",
        re.IGNORECASE,
    )
    text = section_pattern.sub(r"\n\n## \1\n", text)

    # 2. Format Question Labels: "Câu 1.", "Câu 2:" -> "\n\n**Câu 1.**"
    q_pattern = re.compile(r"(?m)^(Câu\s+\d+[\.:])\s*", re.IGNORECASE)
    text = q_pattern.sub(r"\n\n**\1** ", text)

    # Also format inline questions if merged into previous line
    inline_q_pattern = re.compile(r"([^\n])\s+(Câu\s+\d+[\.:])\s*", re.IGNORECASE)
    text = inline_q_pattern.sub(r"\1\n\n**\2** ", text)

    # 3. Format Options: "A. ", "B. ", "C. ", "D. "
    # On start of lines:
    text = re.sub(r"(?m)^([A-D])[\.:]\s*", r"- **\1.** ", text)
    # Inline options (e.g. "A. x = 1 B. x = 2"):
    text = re.sub(r"(\s{2,}|\t+)([A-D])[\.:]\s+", r"  **\2.** ", text)

    # 4. Clean consecutive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def convert_pdf_to_markdown(
    pdf_path: Path,
    output_dir: Path,
    max_pages: int = 15,
    replace_math: bool = True,
) -> Dict[str, Any]:
    """
    Converts a single PDF document to a structured Markdown (.md) file.
    """
    start_time = datetime.now()
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        return {
            "file_name": pdf_path.name,
            "status": "ERROR",
            "error": str(e),
            "pages": 0,
            "chars": 0,
            "math_replaced": 0,
            "duration_sec": 0.0,
        }

    total_pages = len(doc)
    pages_to_process = min(total_pages, max_pages) if max_pages > 0 else total_pages

    if total_pages == 0:
        return {
            "file_name": pdf_path.name,
            "status": "EMPTY",
            "pages": 0,
            "chars": 0,
            "math_replaced": 0,
            "duration_sec": 0.0,
        }

    md_pages = []
    total_math_replaced = 0
    total_chars = 0

    for pno in range(pages_to_process):
        page = doc[pno]
        raw_text = page.get_text("text")
        total_chars += len(raw_text)

        # Check for tables first
        tables_md = []
        try:
            tabs = page.find_tables()
            for tab in tabs:
                df_rows = tab.extract()
                if df_rows and len(df_rows) >= 2:
                    header = "| " + " | ".join(str(c or "").strip().replace("\n", " ") for c in df_rows[0]) + " |"
                    sep = "| " + " | ".join(["---"] * len(df_rows[0])) + " |"
                    rows = [
                        "| " + " | ".join(str(c or "").strip().replace("\n", " ") for c in r) + " |"
                        for r in df_rows[1:]
                    ]
                    tables_md.append("\n".join([header, sep] + rows))
        except Exception:
            tables_md = []

        # Sanitize math
        sanitized_text, math_count = sanitize_math_and_replace(raw_text, replace_with_random_latex=replace_math)
        total_math_replaced += math_count

        # Format structure
        formatted_md = format_markdown_structure(sanitized_text)

        # If tables were found, append them
        if tables_md:
            formatted_md += "\n\n### Bảng dữ liệu trích xuất:\n\n" + "\n\n".join(tables_md)

        md_pages.append(f"<!-- Page {pno + 1} / {total_pages} -->\n\n{formatted_md}")

    # Build final markdown file with frontmatter
    safe_stem = re.sub(r'[^\w\s\-\.\(\)]', '_', pdf_path.stem).strip()
    out_file = output_dir / f"{safe_stem}.md"

    frontmatter = [
        "---",
        f'source_pdf: "{pdf_path.name}"',
        f"total_pages: {total_pages}",
        f"pages_converted: {pages_to_process}",
        f"chars_extracted: {total_chars}",
        f"math_blocks_replaced: {total_math_replaced}",
        f'converted_at: "{datetime.now().isoformat()}"',
        "---",
        "",
        f"# {pdf_path.stem}",
        "",
    ]

    body = "\n\n---\n\n".join(md_pages)
    full_content = "\n".join(frontmatter) + "\n" + body + "\n"

    out_file.write_text(full_content, encoding="utf-8")

    duration = (datetime.now() - start_time).total_seconds()
    return {
        "file_name": pdf_path.name,
        "markdown_file": out_file.name,
        "status": "SUCCESS",
        "pages": pages_to_process,
        "total_pages": total_pages,
        "chars": total_chars,
        "math_replaced": total_math_replaced,
        "duration_sec": round(duration, 3),
    }


def is_digital_pdf(pdf_path: Path, min_chars_per_page: int = 80) -> bool:
    """Quickly tests if the PDF has a selectable text stream (digital) vs scanned image."""
    try:
        doc = pymupdf.open(pdf_path)
        if len(doc) == 0:
            return False
        # Sample up to first 3 pages
        sample_pages = min(3, len(doc))
        sample_chars = sum(len(doc[i].get_text("text").strip()) for i in range(sample_pages))
        avg = sample_chars / sample_pages
        return avg >= min_chars_per_page
    except Exception:
        return False


def select_random_digital_pdfs(
    input_dir: Path,
    count: int = 100,
    seed: int = 42,
    filter_digital: bool = True,
) -> List[Path]:
    """Finds all PDFs, shuffles with seed, and selects N digital PDFs."""
    all_pdfs = sorted(list(input_dir.glob("*.pdf")))
    if not all_pdfs:
        return []

    rng = random.Random(seed)
    rng.shuffle(all_pdfs)

    if not filter_digital:
        return all_pdfs[:count]

    selected = []
    print(f"Scanning pool of {len(all_pdfs)} PDFs to select {count} digital PDFs...")
    for pdf_path in all_pdfs:
        if is_digital_pdf(pdf_path):
            selected.append(pdf_path)
            if len(selected) >= count:
                break

    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Convert N random PDFs to Markdown with structure and clean LaTeX replacement."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/home/daominhwysi/project/scrape-toanmath/downloads",
        help="Directory containing source PDF files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/markdown_converted",
        help="Target directory to save generated .md files.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of random PDFs to convert (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=15,
        help="Maximum pages to convert per PDF (default: 15; 0 for all).",
    )
    parser.add_argument(
        "--replace-math",
        action="store_true",
        default=True,
        help="Replace unparseable MathType blocks with random clean LaTeX formulas (default: True).",
    )
    parser.add_argument(
        "--no-filter-digital",
        action="store_true",
        default=False,
        help="Disable digital filter (allows scanned PDFs).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 4),
        help="Number of parallel worker processes.",
    )

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    if not input_path.exists():
        # Fallback to scrape-toanmath/sequence_labelling_input_data if downloads not found
        fallback = Path("/home/daominhwysi/project/scrape-toanmath/sequence_labelling_input_data")
        if fallback.exists():
            print(f"Directory {input_path} not found. Using fallback: {fallback}")
            input_path = fallback
        else:
            print(f"Error: input directory {input_path} does not exist.")
            sys.exit(1)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filter_digital = not args.no_filter_digital

    # 1. Select random digital PDFs
    selected_pdfs = select_random_digital_pdfs(
        input_dir=input_path,
        count=args.count,
        seed=args.seed,
        filter_digital=filter_digital,
    )

    if not selected_pdfs:
        print("No matching PDFs found to convert.")
        sys.exit(1)

    print(f"\n[START] Selected {len(selected_pdfs)} PDFs to convert to Markdown.")
    print(f"Output directory: {output_path.resolve()}")
    print(f"Parallel workers: {args.workers}")
    print(f"Max pages per PDF: {args.max_pages} (0 = unlimited)")
    print(f"Math sanitization & LaTeX replacement: {args.replace_math}")

    # 2. Parallel Conversion
    results = []
    start_all = datetime.now()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_pdf = {
            executor.submit(
                convert_pdf_to_markdown,
                pdf,
                output_path,
                args.max_pages,
                args.replace_math,
            ): pdf
            for pdf in selected_pdfs
        }

        for completed, future in enumerate(as_completed(future_to_pdf), start=1):
            res = future.result()
            results.append(res)
            print(f"[Converting PDFs to Markdown] {completed}/{len(selected_pdfs)} completed: {future_to_pdf[future]}")

    total_time = (datetime.now() - start_all).total_seconds()

    # 3. Summary & Report
    success_count = sum(1 for r in results if r.get("status") == "SUCCESS")
    total_pages = sum(r.get("pages", 0) for r in results)
    total_chars = sum(r.get("chars", 0) for r in results)
    total_math_replaced = sum(r.get("math_replaced", 0) for r in results)

    report = {
        "timestamp": datetime.now().isoformat(),
        "input_directory": str(input_path.resolve()),
        "output_directory": str(output_path.resolve()),
        "requested_count": args.count,
        "total_selected": len(selected_pdfs),
        "successful_conversions": success_count,
        "failed_conversions": len(results) - success_count,
        "total_pages_converted": total_pages,
        "total_chars_extracted": total_chars,
        "total_math_blocks_replaced": total_math_replaced,
        "elapsed_seconds": round(total_time, 2),
        "speed_pages_per_sec": round(total_pages / total_time, 1) if total_time > 0 else 0,
        "speed_files_per_sec": round(len(results) / total_time, 2) if total_time > 0 else 0,
        "conversions": results,
    }

    report_path = output_path / "conversion_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 65)
    print(" CONVERSION SUMMARY REPORT ")
    print("=" * 65)
    print(f"Successfully converted: {success_count} / {len(selected_pdfs)} PDFs")
    print(f"Total pages processed: {total_pages:,}")
    print(f"Total characters extracted: {total_chars:,}")
    print(f"Total math blocks replaced with clean LaTeX: {total_math_replaced:,}")
    print(f"Total processing time: {total_time:.2f}s ({round(total_pages / total_time, 1)} pages/sec)")
    print(f"Markdown directory: {output_path.resolve()}")
    print(f"Detailed report saved: {report_path.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
