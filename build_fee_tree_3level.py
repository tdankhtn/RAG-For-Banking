from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.rag.fee_tree_3level import extract_fee_tree_3level


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="processed", help="output folder")
    ap.add_argument(
        "--default-segment",
        type=str,
        default="KH thường",
        help="segment mặc định nếu file không có cột phân khúc",
    )
    ap.add_argument("pdfs", nargs="+", help="paths to PDF files")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for pdf in args.pdfs:
        pdf_path = Path(pdf)
        if not pdf_path.exists():
            print(f"[WARN] Missing file: {pdf_path}")
            continue

        print(f"[INFO] Processing: {pdf_path.name}")
        data = extract_fee_tree_3level(
            pdf_path, default_segment_if_missing=args.default_segment
        )

        out_path = out_dir / f"{pdf_path.stem}.fee_tree_3level.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[OK] Wrote: {out_path}")


if __name__ == "__main__":
    main()
