"""
curate_500_diverse_pdfs.py — Stratified Diversity Sampler & Deduplicator.

Selects the 500 most diverse digital math exam PDFs from scrape-toanmath/downloads
while guaranteeing 100% deduplication against all existing documents in the project.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import logging
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pymupdf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCRAPE_DIR = Path("/home/daominhwysi/project/scrape-toanmath/downloads")
PROJECT_DATA_DIR = Path("data")
OUTPUT_DIR = Path("data/batch_500_diverse_input")
TARGET_COUNT = 500

PROVINCES = [
    ('Hà Nội', 'Miền Bắc'), ('TP HCM', 'Miền Nam'), ('Đà Nẵng', 'Miền Trung'), 
    ('Hải Phòng', 'Miền Bắc'), ('Cần Thơ', 'Miền Nam'), ('Nghệ An', 'Miền Trung'), 
    ('Thanh Hóa', 'Miền Trung'), ('Nam Định', 'Miền Bắc'), ('Bắc Giang', 'Miền Bắc'), 
    ('Vĩnh Phúc', 'Miền Bắc'), ('Huế', 'Miền Trung'), ('Bình Định', 'Miền Trung'), 
    ('Đồng Nai', 'Miền Nam'), ('Long An', 'Miền Nam'), ('An Giang', 'Miền Nam'), 
    ('Lâm Đồng', 'Tây Nguyên'), ('Phú Thọ', 'Miền Bắc'), ('Thái Bình', 'Miền Bắc'), 
    ('Quảng Ninh', 'Miền Bắc'), ('Hưng Yên', 'Miền Bắc'), ('Hải Dương', 'Miền Bắc'), 
    ('Hà Tĩnh', 'Miền Trung'), ('Quảng Nam', 'Miền Trung'), ('Khánh Hòa', 'Miền Trung'), 
    ('Bình Dương', 'Miền Nam'), ('Bến Tre', 'Miền Nam'), ('Tây Ninh', 'Miền Nam'), 
    ('Quảng Trị', 'Miền Trung'), ('Quảng Bình', 'Miền Trung'), ('Ninh Bình', 'Miền Bắc'), 
    ('Lạng Sơn', 'Miền Bắc'), ('Bắc Ninh', 'Miền Bắc'), ('Tiền Giang', 'Miền Nam'), 
    ('Kiên Giang', 'Miền Nam'), ('Bà Rịa', 'Miền Nam'), ('Gia Lai', 'Tây Nguyên'), 
    ('Đắk Lắk', 'Tây Nguyên'), ('Thái Nguyên', 'Miền Bắc'), ('Sơn La', 'Miền Bắc'),
    ('Hà Nam', 'Miền Bắc'), ('Vĩnh Long', 'Miền Nam'), ('Yên Bái', 'Miền Bắc'), 
    ('Lào Cai', 'Miền Bắc'), ('Cao Bằng', 'Miền Bắc'), ('Sóc Trăng', 'Miền Nam'), 
    ('Bạc Liêu', 'Miền Nam'), ('Cà Mau', 'Miền Nam'), ('Kon Tum', 'Tây Nguyên'),
    ('Bình Thuận', 'Miền Trung'), ('Ninh Thuận', 'Miền Trung'), ('Phú Yên', 'Miền Trung')
]


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_existing_project_hashes() -> Set[str]:
    """Collects SHA-256 hashes of all existing PDFs in this project."""
    hashes = set()
    logger.info("Scanning existing project PDFs for deduplication...")
    for p in PROJECT_DATA_DIR.rglob("*.pdf"):
        # Exclude our target output dir if re-running
        if OUTPUT_DIR.name in str(p):
            continue
        try:
            hashes.add(compute_sha256(p))
        except Exception:
            pass
    logger.info(f"Found {len(hashes)} unique known PDF hashes in project.")
    return hashes


def analyze_pdf_metadata(filepath: Path) -> Dict[str, Any]:
    """Extracts grade, doc_type, topic, region, and curriculum metadata from filename."""
    filename = filepath.name
    lower = filename.lower()

    # 1. Grade Level
    if any(k in lower for k in ['toán 10', 'lớp 10', 'đs_ 10', 'hh 10', 'kntt 10', 'chân trời 10', 'cánh diều 10']):
        grade = 'Toán 10'
    elif any(k in lower for k in ['toán 11', 'lớp 11', 'đs_ 11', 'hh 11', 'kntt 11', 'chân trời 11', 'cánh diều 11']):
        grade = 'Toán 11'
    elif any(k in lower for k in ['toán 12', 'lớp 12', 'đs_ 12', 'hh 12', 'kntt 12', 'chân trời 12', 'cánh diều 12']):
        grade = 'Toán 12'
    elif any(k in lower for k in ['hsg', 'học sinh giỏi', 'olympic', 'mtct']):
        grade = 'HSG / Olympic'
    elif any(k in lower for k in ['thptqg', 'thpt quốc gia', 'tn thpt', 'tốt nghiệp thpt', 'đại học', 'thi thử thpt', 'khảo sát']):
        grade = 'Thi thử THPTQG'
    elif any(k in lower for k in ['chuyên đề', 'tài liệu', 'lý thuyết', 'tổng hợp', 'bài tập']):
        grade = 'Chuyên đề / Lý thuyết'
    else:
        grade = 'Toán 12'

    # 2. Document / Exam Type
    if any(k in lower for k in ['giữa kì 1', 'giữa học kì 1', 'giữa kỳ 1', 'giữa học kỳ 1', 'gk1']):
        doc_type = 'Giữa học kì 1'
    elif any(k in lower for k in ['cuối kì 1', 'cuối học kì 1', 'cuối kỳ 1', 'cuối học kỳ 1', 'hk1', 'học kì 1', 'học kỳ 1']):
        doc_type = 'Cuối học kì 1'
    elif any(k in lower for k in ['giữa kì 2', 'giữa học kì 2', 'giữa kỳ 2', 'giữa học kỳ 2', 'gk2']):
        doc_type = 'Giữa học kì 2'
    elif any(k in lower for k in ['cuối kì 2', 'cuối học kì 2', 'cuối kỳ 2', 'cuối học kỳ 2', 'hk2', 'học kì 2', 'học kỳ 2']):
        doc_type = 'Cuối học kì 2'
    elif any(k in lower for k in ['hsg', 'học sinh giỏi', 'olympic', 'mtct']):
        doc_type = 'Thi chọn HSG / Olympic'
    elif any(k in lower for k in ['thi thử', 'khảo sát', 'tn thpt', 'thpt quốc gia']):
        doc_type = 'Thi thử THPTQG / Tốt nghiệp'
    elif any(k in lower for k in ['chương', '1 tiết', 'định kỳ']):
        doc_type = 'Kiểm tra 1 tiết / Chương'
    else:
        doc_type = 'Chuyên đề & Bài tập'

    # 3. Topic / Mathematical Domain
    topic = 'Đề thi tổng hợp'
    if any(k in lower for k in ['tích phân', 'nguyên hàm']):
        topic = 'Nguyên hàm & Tích phân'
    elif any(k in lower for k in ['oxyz', 'tọa độ trong không gian']):
        topic = 'Hình học Oxyz'
    elif any(k in lower for k in ['khối đa diện', 'thể tích', 'hình không gian', 'hh không gian']):
        topic = 'Hình học không gian'
    elif any(k in lower for k in ['xác suất', 'tổ hợp', 'chỉnh hợp', 'nhị thức']):
        topic = 'Tổ hợp & Xác suất'
    elif any(k in lower for k in ['số phức']):
        topic = 'Số phức'
    elif any(k in lower for k in ['đạo hàm', 'khảo sát hàm', 'tiệm cận', 'cực trị']):
        topic = 'Hàm số & Đạo hàm'
    elif any(k in lower for k in ['lượng giác']):
        topic = 'Lượng giác'
    elif any(k in lower for k in ['vecto', 'tích vô hướng']):
        topic = 'Vectơ'
    elif any(k in lower for k in ['bất đẳng thức', 'bđt', 'cực trị đại số']):
        topic = 'Bất đẳng thức'

    # 4. Geography
    location = 'Toàn quốc'
    region = 'Toàn quốc'
    for prov, reg in PROVINCES:
        if prov.lower() in lower:
            location = prov
            region = reg
            break

    # 5. Curriculum
    if any(k in lower for k in ['kntt', 'kết nối tri thức', 'ket noi tri thuc']):
        curr = 'Kết Nối Tri Thức'
    elif any(k in lower for k in ['cánh diều', 'canh dieu']):
        curr = 'Cánh Diều'
    elif any(k in lower for k in ['chân trời', 'chân trời sáng tạo', 'chan troi sang tao']):
        curr = 'Chân Trời Sáng Tạo'
    elif any(k in lower for k in ['cấu trúc mới', 'định dạng mới', '2025']):
        curr = 'Cấu trúc 2025'
    else:
        curr = 'Chương trình chuẩn'

    return {
        "grade": grade,
        "doc_type": doc_type,
        "topic": topic,
        "location": location,
        "region": region,
        "curr": curr,
    }


def curate_500_diverse_pdfs() -> List[Dict[str, Any]]:
    """Gathers and filters the candidate pool, then selects 500 diverse digital PDFs."""
    known_hashes = get_existing_project_hashes()

    logger.info(f"Scanning {SCRAPE_DIR} for candidate PDF documents...")
    all_scrape_files = list(SCRAPE_DIR.glob("*.pdf"))
    logger.info(f"Found {len(all_scrape_files)} total files in scrape repository.")

    # Shuffle to ensure random fair sampling before filtering
    random.seed(42)
    random.shuffle(all_scrape_files)

    valid_candidates: List[Dict[str, Any]] = []
    scanned_hashes: Set[str] = set()

    for idx, p in enumerate(all_scrape_files, start=1):
        if len(valid_candidates) >= 1500:
            # We have more than enough candidates for stratified sampling
            break

        # Basic size check: 50 KB <= size <= 15 MB
        sz = p.stat().st_size
        if sz < 50_000 or sz > 15_000_000:
            continue

        try:
            # Compute hash and check deduplication
            file_hash = compute_sha256(p)
            if file_hash in known_hashes or file_hash in scanned_hashes:
                continue

            doc = pymupdf.open(str(p))
            page_count = len(doc)
            if page_count < 3 or page_count > 15:
                doc.close()
                continue

            # Verify digital text (not scanned image)
            p1_text = doc[0].get_text().strip()
            doc.close()
            if len(p1_text) < 80:
                continue

            # Valid digital document!
            scanned_hashes.add(file_hash)
            meta = analyze_pdf_metadata(p)
            meta["path"] = str(p)
            meta["filename"] = p.name
            meta["sha256"] = file_hash
            meta["page_count"] = page_count
            meta["size_bytes"] = sz
            meta["size_mb"] = round(sz / (1024 * 1024), 2)
            valid_candidates.append(meta)

            if len(valid_candidates) % 100 == 0:
                logger.info(f"Verified {len(valid_candidates)} clean, digital, non-duplicate candidates...")

        except Exception:
            continue

    logger.info(f"Candidate collection complete: {len(valid_candidates)} verified candidates.")

    # Multi-dimensional Stratified Diversity Selection
    # Target counts per Grade:
    # Toán 10: 90
    # Toán 11: 100
    # Toán 12: 120
    # Thi thử THPTQG: 120
    # HSG / Olympic: 40
    # Chuyên đề: 30
    grade_targets = {
        'Toán 10': 90,
        'Toán 11': 100,
        'Toán 12': 120,
        'Thi thử THPTQG': 120,
        'HSG / Olympic': 40,
        'Chuyên đề / Lý thuyết': 30,
    }

    by_grade = defaultdict(list)
    for c in valid_candidates:
        by_grade[c['grade']].append(c)

    selected: List[Dict[str, Any]] = []
    used_hashes: Set[str] = set()

    for grade, target_k in grade_targets.items():
        cands = by_grade.get(grade, [])
        # Sub-stratify by exam type and region
        by_subtype = defaultdict(list)
        for c in cands:
            key = (c['doc_type'], c['region'], c['topic'])
            by_subtype[key].append(c)

        # Round-robin selection across subtypes
        subkeys = list(by_subtype.keys())
        random.shuffle(subkeys)

        picked_for_grade = []
        cycle = 0
        while len(picked_for_grade) < target_k and subkeys:
            key = subkeys[cycle % len(subkeys)]
            bucket = by_subtype[key]
            if bucket:
                item = bucket.pop(0)
                if item['sha256'] not in used_hashes:
                    picked_for_grade.append(item)
                    used_hashes.add(item['sha256'])
            else:
                subkeys.remove(key)
                continue
            cycle += 1

        selected.extend(picked_for_grade)
        logger.info(f"Grade '{grade}': selected {len(picked_for_grade)} / {target_k} requested.")

    # If any remaining slots to reach 500, top up from unused candidates
    if len(selected) < TARGET_COUNT:
        remaining = [c for c in valid_candidates if c['sha256'] not in used_hashes]
        random.shuffle(remaining)
        needed = TARGET_COUNT - len(selected)
        selected.extend(remaining[:needed])

    selected = selected[:TARGET_COUNT]
    logger.info(f"Successfully selected exactly {len(selected)} diverse PDF documents.")

    # Copy files into OUTPUT_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_records = []

    logger.info(f"Copying files to '{OUTPUT_DIR}'...")
    for idx, item in enumerate(selected, start=1):
        clean_name = f"doc_{idx:03d}_{item['filename']}"
        dest_path = OUTPUT_DIR / clean_name
        shutil.copy2(item['path'], dest_path)

        rec = {
            "index": idx,
            "filename": clean_name,
            "original_filename": item["filename"],
            "grade": item["grade"],
            "doc_type": item["doc_type"],
            "topic": item["topic"],
            "curriculum": item["curr"],
            "location": item["location"],
            "region": item["region"],
            "page_count": item["page_count"],
            "size_mb": item["size_mb"],
            "sha256": item["sha256"],
        }
        manifest_records.append(rec)

    # Save manifest.json
    manifest_json_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, ensure_ascii=False, indent=2)

    # Save manifest.csv
    manifest_csv_path = OUTPUT_DIR / "manifest.csv"
    with open(manifest_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["index", "filename", "original_filename", "grade", "doc_type", "topic", "curriculum", "location", "region", "page_count", "size_mb", "sha256"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest_records)

    logger.info(f"Curation complete! 500 documents staged in {OUTPUT_DIR}")
    logger.info(f"Manifest written: {manifest_json_path}")
    return manifest_records


if __name__ == "__main__":
    curate_500_diverse_pdfs()
