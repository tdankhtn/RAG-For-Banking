# -*- coding: utf-8 -*-
"""
Chunking cho RAG từ PDF biểu phí:
- Dùng parser 3-level (fee_type / card_type / segment)
- Mỗi dòng phí -> 1 chunk/segment, kèm Q&A để match câu hỏi người dùng.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.config import settings
from src.rag.fee_fact_chunker import extract_fee_fact_chunks
from src.rag.fee_tree_3level import sha256_file


def write_jsonl(items: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def write_json(items: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def build_docs_meta(pdf_paths: List[Path]) -> List[Dict[str, Any]]:
    meta = []
    for p in pdf_paths:
        meta.append(
            {
                "doc_id": p.stem,
                "source_file": p.name,
                "sha256": sha256_file(p),
            }
        )
    return meta


if __name__ == "__main__":
    data_dir = Path(settings.data_dir)
    pdfs = list(data_dir.glob("*.pdf"))

    if not pdfs:
        print(f"[WARN] No PDF files found in {data_dir}")
        raise SystemExit(1)

    all_chunks: List[Dict[str, Any]] = []
    for pdf in pdfs:
        print(f"Processing: {pdf.name}")
        chunks = extract_fee_fact_chunks(pdf)
        print(f"  → {len(chunks)} chunks")
        all_chunks.extend(chunks)

    out_dir = Path(settings.artifacts_dir)

    write_jsonl(all_chunks, out_dir / "rag_chunks_fee_fact_segmented_qa.jsonl")
    print(f"\n✅ Exported {len(all_chunks)} chunks to:")
    print(f"   - {out_dir / 'rag_chunks_fee_fact_segmented_qa.jsonl'}")

    write_json(all_chunks, out_dir / "rag_chunks_fee_fact_segmented_qa.json")
    print(f"   - {out_dir / 'rag_chunks_fee_fact_segmented_qa.json'}")

    with (out_dir / "docs_meta.json").open("w", encoding="utf-8") as f:
        json.dump(build_docs_meta(pdfs), f, ensure_ascii=False, indent=2)
    print(f"   - {out_dir / 'docs_meta.json'}")

    print(f"\n📊 Summary:")
    print(f"   - Total PDFs: {len(pdfs)}")
    print(f"   - Total chunks: {len(all_chunks)}")
    print(f"   - Avg chunks/PDF: {len(all_chunks)/len(pdfs):.1f}")
