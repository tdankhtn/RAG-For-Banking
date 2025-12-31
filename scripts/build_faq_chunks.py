# -*- coding: utf-8 -*-
"""
Chunking FAQ từ PDF (cau_hoi_thuong_gap.pdf):
- Tách theo "Câu hỏi:" để tạo Q&A chunks
- Lọc dòng table để tránh nhiễu retriever
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.config import settings
from src.rag.faq_chunker import extract_faq_chunks


def write_jsonl(items: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def write_json(items: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(settings.artifacts_dir))
    ap.add_argument(
        "pdfs",
        nargs="*",
        help="paths to FAQ PDFs (default: data_dir/cau_hoi_thuong_gap.pdf)",
    )
    args = ap.parse_args()

    pdfs = [Path(p) for p in args.pdfs]
    if not pdfs:
        pdfs = [Path(settings.data_dir) / "cau_hoi_thuong_gap.pdf"]

    out_dir = Path(args.out)
    all_chunks: List[Dict[str, Any]] = []

    for pdf in pdfs:
        if not pdf.exists():
            print(f"[WARN] Missing file: {pdf}")
            continue
        print(f"[INFO] Processing: {pdf.name}")
        chunks = extract_faq_chunks(pdf)
        print(f"  → {len(chunks)} chunks")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("[WARN] No FAQ chunks extracted.")
        return

    out_jsonl = out_dir / "rag_chunks_faq_qa.jsonl"
    out_json = out_dir / "rag_chunks_faq_qa.json"
    write_jsonl(all_chunks, out_jsonl)
    write_json(all_chunks, out_json)

    print(f"\n✅ Exported {len(all_chunks)} chunks to:")
    print(f"   - {out_jsonl}")
    print(f"   - {out_json}")


if __name__ == "__main__":
    main()
