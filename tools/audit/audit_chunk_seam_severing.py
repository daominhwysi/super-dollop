"""Audit script to identify severed stimuli and cross-page questions across multi-chunk documents."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sequence_labelling.parser.long_parser.document_cartographer import (
    CartographyManifest,
    DocumentCartographer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audit_seams")


def get_multi_chunk_documents(annotated_dir: Path) -> List[Tuple[str, Path, int]]:
    """Returns list of (doc_id, doc_path, chunk_count) for all multi-chunk documents."""
    multi_chunk = []
    for d in annotated_dir.iterdir():
        if not d.is_dir():
            continue
        chunks_dir = d / "chunks"
        if chunks_dir.exists():
            xmls = list(chunks_dir.glob("*.xml"))
            if len(xmls) > 1:
                multi_chunk.append((d.name, d, len(xmls)))
    return sorted(multi_chunk, key=lambda x: x[0])


def extract_historical_cut_seams(doc_dir: Path) -> List[int]:
    """Extracts the page cut seams used between consecutive chunks."""
    chunks_dir = doc_dir / "chunks"
    chunk_json_files = sorted(chunks_dir.glob("chunk_*.json"), key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else 999)
    
    cut_seams = []
    for idx in range(len(chunk_json_files) - 1):
        try:
            with open(chunk_json_files[idx], "r", encoding="utf-8") as f:
                c_data = json.load(f)
            p_ids = c_data.get("page_ids", [])
            if p_ids:
                last_page = max(p_ids)
                cut_seams.append(last_page)
        except Exception:
            pass
    return cut_seams


def audit_document_seams(
    doc_id: str,
    doc_dir: Path,
    input_dir: Path,
    metadata_catalog: Dict[str, Any],
    cartographer: DocumentCartographer,
    cartography_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Audits a single multi-chunk document for severed entities."""
    md_file = input_dir / f"{doc_id}.md"
    if not md_file.exists():
        # Try matching by stem
        candidates = list(input_dir.glob(f"*{doc_id}*.md"))
        if candidates:
            md_file = candidates[0]
        else:
            return {"doc_id": doc_id, "status": "missing_input_md"}

    raw_text = md_file.read_text(encoding="utf-8", errors="ignore")
    meta = metadata_catalog.get(doc_id, {})
    subject = meta.get("subject", "Unknown")

    pages = DocumentCartographer.extract_pages(raw_text)
    total_pages = len(pages)

    # Check if pre-computed cartography manifest exists
    cart_file = (cartography_dir / f"{doc_id}.json") if cartography_dir else None
    if cart_file and cart_file.exists():
        try:
            with open(cart_file, "r", encoding="utf-8") as f:
                c_dict = json.load(f)
            manifest = cartographer._build_manifest(
                doc_id=doc_id,
                total_pages=total_pages,
                raw_objects=c_dict.get("objects", []),
                subject=subject,
            )
        except Exception:
            manifest = cartographer.survey_document(raw_text, doc_id=doc_id, subject=subject)
    else:
        manifest = cartographer.survey_document(raw_text, doc_id=doc_id, subject=subject)

    historical_cuts = extract_historical_cut_seams(doc_dir)
    clean_seams = set(manifest.clean_cut_seams)

    severed_cuts = []
    for cut in historical_cuts:
        if cut not in clean_seams:
            # Identify which objects straddle this cut
            straddling_questions = [
                q.id for q in manifest.questions.values()
                if q.page_span[0] <= cut < q.page_span[1]
            ]
            straddling_stimuli = [
                s.id for s in manifest.stimuli.values()
                if s.page_span[0] <= cut < max(s.page_span[1], max([manifest.questions[qid].page_span[1] for qid in s.serves_questions if qid in manifest.questions] or [s.page_span[1]]))
            ]
            severed_cuts.append({
                "cut_seam": cut,
                "straddling_questions": straddling_questions,
                "straddling_stimuli": straddling_stimuli,
            })

    return {
        "doc_id": doc_id,
        "subject": subject,
        "total_pages": total_pages,
        "historical_cuts": historical_cuts,
        "clean_seams": sorted(list(clean_seams)),
        "severed_cuts_count": len(severed_cuts),
        "severed_cuts": severed_cuts,
        "has_defect": len(severed_cuts) > 0,
    }


def main():
    annotated_dir = Path("data/sequence_labelling_annotated")
    input_dir = Path("data/sequence_labelling_input_data")
    cartography_dir = Path("data/sequence_labelling_cartography")
    catalog_path = Path("data/sequence_labelling_metadata_catalog.json")

    catalog = {}
    if catalog_path.exists():
        with open(catalog_path, "r", encoding="utf-8") as f:
            catalog = json.load(f).get("documents", {})

    multi_chunks = get_multi_chunk_documents(annotated_dir)
    logger.info(f"Auditing {len(multi_chunks)} multi-chunk documents for severed entity boundaries...")

    cartographer = DocumentCartographer()
    results = []
    severed_count = 0

    for doc_id, doc_path, chunk_cnt in multi_chunks:
        audit_res = audit_document_seams(
            doc_id, doc_path, input_dir, catalog, cartographer, cartography_dir=cartography_dir
        )
        if audit_res.get("has_defect"):
            severed_count += 1
            logger.warning(
                f"🚨 SEVERED CUT in '{doc_id}' ({audit_res.get('subject')}): "
                f"{audit_res['severed_cuts_count']} cut(s) severed objects!"
            )
            for sc in audit_res["severed_cuts"]:
                logger.warning(
                    f"   Seam {sc['cut_seam']} -> Questions: {sc['straddling_questions']} | Stimuli: {sc['straddling_stimuli']}"
                )
        results.append(audit_res)

    out_path = Path("artifacts/reports/chunk_seam_severing_audit.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_multi_chunk_audited": len(multi_chunks),
                "total_severed_documents": severed_count,
                "defect_rate_percent": (severed_count / len(multi_chunks) * 100) if multi_chunks else 0,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info("=" * 60)
    logger.info(f"AUDIT COMPLETE: {severed_count} / {len(multi_chunks)} multi-chunk documents had severed cuts.")
    logger.info(f"Detailed audit saved to {out_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
