# -*- coding: utf-8 -*-
"""
Chunking cho RAG từ PDF biểu phí:
- Với bảng có nhiều phân khúc (KH Private/Priority/Inspire/KH thường): tách 1 dòng -> N chunk (mỗi chunk 1 phân khúc)
- Với bảng chỉ có 1 mức phí (thẻ tín dụng KH thường): tạo chunk segment='KH thường'
- Mỗi chunk có Q/A để match câu hỏi người dùng.

Yêu cầu: pip install pdfplumber
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber
from src.config import settings


# ----------------------------
# Helpers: normalize / regex
# ----------------------------
ROMAN_RE = re.compile(r"^[IVXLC]+$")
STT_RE = re.compile(r"^\d+(\.\d+)*$")

# Mã phí kiểu CN-95-001, CN11155, CN-98-111 ...
FEE_CODE_RE = re.compile(r"\bCN[-]?\d{2,5}[-]?\d{2,4}\b", re.IGNORECASE)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def strip_accents(s: str) -> str:
    # chuẩn hoá để match header có/không dấu
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", s)

def norm_space(s: str) -> str:
    s = s.replace("\t", " ")
    s = re.sub(r"[ ]+", " ", s)
    return s.strip()

def norm_key(s: str) -> str:
    s = strip_accents(s)
    s = norm_space(s)
    return s.upper()

def norm_cell(x: Any) -> str:
    if x is None:
        return ""
    return norm_space(str(x).replace("\n", " "))

def any_fee_code(row: List[str]) -> Optional[str]:
    joined = " ".join(row)
    m = FEE_CODE_RE.search(joined)
    return m.group(0).upper() if m else None

def looks_like_section_row(row: List[str]) -> bool:
    # ví dụ: ["I", "PHÍ DỊCH VỤ THẺ ...", ...]
    nonempty = [c for c in row if c]
    if not nonempty:
        return False
    has_roman = any(ROMAN_RE.match(c) for c in nonempty[:2])
    has_phi = any("PHI" in norm_key(c) for c in nonempty)
    return has_roman and has_phi

def looks_like_headerish_row(row: List[str]) -> bool:
    # header có thể nhiều dòng
    k = norm_key(" ".join(row))
    if any_fee_code(row):
        return False
    return any(w in k for w in ["STT", "MA PHI", "DICH VU", "MUC PHI", "KH "])

def is_group_heading_row(row: List[str], service_idx: Optional[int], segment_idxs: Dict[str, int]) -> bool:
    # nhóm kiểu "6 Phí rút tiền mặt" thường không có mã phí, segment rỗng
    if any_fee_code(row):
        return False
    if service_idx is None or service_idx >= len(row):
        return False
    service = row[service_idx].strip()
    if not service:
        return False
    # segment cells rỗng (nếu có segment)
    if segment_idxs:
        seg_vals = [row[i].strip() if i < len(row) else "" for i in segment_idxs.values()]
        if any(seg_vals):
            return False
    # có STT nhưng không có mã phí
    has_stt = any(STT_RE.match(c.strip()) for c in row[:2] if c.strip())
    return has_stt

def extract_year_fees(fee_text: str) -> Dict[str, str]:
    """
    Bắt các pattern kiểu:
    - "Miễn phí ... năm đầu ... từ năm thứ 2 ... 60.000 VND/năm"
    """
    out: Dict[str, str] = {}
    t = norm_space(fee_text)

    # year 1: miễn phí năm đầu
    if re.search(r"nam\s+dau", norm_key(t)) and "MIEN PHI" in norm_key(t):
        out["year1_fee"] = "Miễn phí"

    # year2+: cố gắng bắt số tiền VND/năm đi kèm "từ năm thứ 2"
    m2 = re.search(
        r"(tu\s+nam\s+thu\s*2[^0-9]*)(\d[\d\.\, ]*\s*VND\s*/\s*nam)",
        norm_key(t),
    )
    if m2:
        # lấy lại theo bản gốc (đỡ mất định dạng)
        m2_raw = re.search(
            r"(từ\s+năm\s+thứ\s*2[^0-9]*)(\d[\d\.\, ]*\s*VND\s*/\s*năm)",
            t,
            flags=re.IGNORECASE,
        )
        if m2_raw:
            out["year2_plus_fee"] = norm_space(m2_raw.group(2))

    return out

def segment_key(seg: str) -> str:
    k = norm_key(seg)
    if "PRIVATE" in k:
        return "kh_private"
    if "PRIORITY" in k:
        return "kh_priority"
    if "INSPIRE" in k:
        return "kh_inspire"
    if "THUONG" in k or "THƯỜNG" in seg.upper():
        return "kh_thuong"
    return re.sub(r"[^a-z0-9]+", "_", strip_accents(seg).lower()).strip("_")


# ----------------------------
# Header mapping
# ----------------------------
@dataclass
class HeaderMap:
    stt_idx: Optional[int]
    fee_code_idx: Optional[int]
    service_idx: Optional[int]
    fee_value_idx: Optional[int]            # dùng cho bảng không có phân khúc
    segment_idxs: Dict[str, int]            # dùng cho bảng có phân khúc

SEGMENT_CANON = {
    "KH PRIVATE": "KH Private",
    "KH PRIORITY": "KH Priority",
    "KH INSPIRE": "KH Inspire",
    "KH THUONG": "KH thường",
    "KH THƯỜNG": "KH thường",
}

def canonical_segment_from_header(cell_text: str) -> Optional[str]:
    k = norm_key(cell_text)
    for raw, canon in SEGMENT_CANON.items():
        if raw in k:
            return canon
    return None

def build_header_map_from_rows(rows: List[List[str]], start_i: int) -> Tuple[Optional[HeaderMap], int]:
    """
    Gom 1-3 dòng header để tìm index cột.
    Trả về (HeaderMap, next_row_index_after_header)
    """
    header_rows: List[List[str]] = []
    i = start_i
    while i < len(rows) and looks_like_headerish_row(rows[i]) and len(header_rows) < 3:
        header_rows.append(rows[i])
        i += 1

    if not header_rows:
        return None, start_i

    max_len = max(len(r) for r in header_rows)
    combined_headers: List[str] = []
    for col in range(max_len):
        parts = []
        for hr in header_rows:
            if col < len(hr):
                v = hr[col].strip()
                if v:
                    parts.append(v)
        combined_headers.append(norm_space(" ".join(parts)))

    stt_idx = None
    fee_code_idx = None
    service_idx = None
    fee_value_idx = None
    segment_idxs: Dict[str, int] = {}

    for idx, h in enumerate(combined_headers):
        hk = norm_key(h)
        if "STT" in hk and stt_idx is None:
            stt_idx = idx
        if ("MA PHI" in hk or "MÃ PHÍ" in h.upper()) and fee_code_idx is None:
            fee_code_idx = idx
        if "DICH VU" in hk and service_idx is None:
            service_idx = idx
        # segment columns
        seg = canonical_segment_from_header(h)
        if seg:
            segment_idxs[seg] = idx
        # fee column (for single-fee tables)
        if "MUC PHI" in hk and fee_value_idx is None:
            fee_value_idx = idx

    # nếu có segment, fee_value_idx không quan trọng
    hm = HeaderMap(
        stt_idx=stt_idx,
        fee_code_idx=fee_code_idx,
        service_idx=service_idx,
        fee_value_idx=fee_value_idx,
        segment_idxs=segment_idxs,
    )
    return hm, i


# ----------------------------
# Chunk creation
# ----------------------------
def make_qa(service: str, segment: str, fee_value: str) -> str:
    service = norm_space(service)
    fee_value = norm_space(fee_value)

    # câu hỏi chính
    q1 = f"Mức phí {service} cho {segment} là bao nhiêu?"
    # thêm biến thể để match query tự nhiên
    if norm_key(segment).endswith("THUONG"):
        q2 = f"Mức phí {service} cho khách hàng thường là bao nhiêu?"
    else:
        q2 = f"Phí {service} dành cho {segment} là bao nhiêu?"

    a = fee_value
    return f"Q: {q1}\nQ: {q2}\nA: {a}"

def chunk_text(doc_name: str, page: int, section_path: List[str], service: str, fee_code: Optional[str],
               segment: str, fee_value: str) -> str:
    sp = " / ".join([s for s in section_path if s])
    head = f"[{doc_name} | page {page}]"
    parts = [
        head,
        (sp if sp else "").strip(),
        f"Dịch vụ: {service}".strip(),
        (f"Mã phí: {fee_code}" if fee_code else "").strip(),
        f"Phân khúc: {segment}".strip(),
        f"Mức phí: {fee_value}".strip(),
        "",
        make_qa(service, segment, fee_value),
    ]
    return "\n".join([p for p in parts if p])

def write_jsonl(items: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def write_json(items: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


# ----------------------------
# Main extractor
# ----------------------------
def extract_fee_fact_chunks(pdf_path: Path, default_segment_if_missing: str = "KH thường") -> List[Dict[str, Any]]:
    doc_id = pdf_path.stem
    sha = sha256_file(pdf_path)

    # settings giúp pdfplumber bắt bảng có đường kẻ tốt hơn
    table_settings = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "edge_min_length": 3,
        "intersection_tolerance": 3,
        "text_tolerance": 3,
    }

    chunks: List[Dict[str, Any]] = []
    section_path: List[str] = []
    current_group: Optional[str] = None

    with pdfplumber.open(str(pdf_path)) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables(table_settings=table_settings) or []
            for t_idx, table in enumerate(tables):
                # normalize rows
                rows = [[norm_cell(c) for c in r] for r in table if r]
                if not rows:
                    continue

                hm: Optional[HeaderMap] = None
                r = 0
                while r < len(rows):
                    row = rows[r]
                    if looks_like_section_row(row):
                        # section như "I PHÍ DỊCH VỤ ..."
                        section_path = [norm_space(" ".join([c for c in row if c]))]
                        current_group = None
                        r += 1
                        continue

                    # detect header block
                    if hm is None and looks_like_headerish_row(row):
                        hm, r_next = build_header_map_from_rows(rows, r)
                        r = r_next
                        continue

                    # nếu chưa có header_map, fallback nhẹ: bỏ qua table lạ
                    if hm is None:
                        r += 1
                        continue

                    # group heading: "6 Phí rút tiền mặt"
                    if is_group_heading_row(row, hm.service_idx, hm.segment_idxs):
                        current_group = row[hm.service_idx] if hm.service_idx is not None and hm.service_idx < len(row) else None
                        r += 1
                        continue

                    # data row
                    fee_code = None
                    if hm.fee_code_idx is not None and hm.fee_code_idx < len(row):
                        fee_code = any_fee_code([row[hm.fee_code_idx]]) or any_fee_code(row)
                    else:
                        fee_code = any_fee_code(row)

                    # lấy service
                    service = ""
                    if hm.service_idx is not None and hm.service_idx < len(row):
                        service = row[hm.service_idx]
                    service = norm_space(service)

                    # nếu service ngắn kiểu "Tại ATM ..." và có group, ghép vào cho rõ nghĩa
                    if current_group and service and not any_fee_code([service]) and service.lower().startswith("tại"):
                        service_full = f"{current_group} - {service}"
                    elif current_group and service and (current_group.lower() not in service.lower()):
                        # nhiều trường hợp service đã đầy đủ thì thôi, còn nếu không thì ghép
                        service_full = service
                    else:
                        service_full = service or (current_group or "")

                    if not service_full:
                        r += 1
                        continue

                    # CASE A: bảng có phân khúc
                    if hm.segment_idxs:
                        for seg, idx in hm.segment_idxs.items():
                            fee_val = row[idx] if idx < len(row) else ""
                            fee_val = norm_space(fee_val)
                            if not fee_val:
                                continue

                            struct = extract_year_fees(fee_val)
                            cid = f"{doc_id}::p{pno:03d}::t{t_idx:02d}::r{r:03d}::{segment_key(seg)}::{fee_code or 'NO_CODE'}"

                            chunks.append({
                                "id": cid,
                                "chunk_type": "fee_fact",
                                "doc_id": doc_id,
                                "source_file": pdf_path.name,
                                "sha256": sha,
                                "page": pno,
                                "table_index": t_idx,
                                "row_index": r,
                                "section_path": section_path,
                                "service": service_full,
                                "fee_code": fee_code,
                                "segment": seg,
                                "fee_value": fee_val,
                                "structured": struct,
                                "text": chunk_text(pdf_path.name, pno, section_path, service_full, fee_code, seg, fee_val),
                            })

                    # CASE B: bảng chỉ có 1 mức phí (không phân khúc) -> segment mặc định
                    else:
                        if hm.fee_value_idx is None or hm.fee_value_idx >= len(row):
                            r += 1
                            continue
                        fee_val = norm_space(row[hm.fee_value_idx])
                        if not fee_val:
                            r += 1
                            continue

                        seg = default_segment_if_missing
                        struct = extract_year_fees(fee_val)
                        cid = f"{doc_id}::p{pno:03d}::t{t_idx:02d}::r{r:03d}::{segment_key(seg)}::{fee_code or 'NO_CODE'}"

                        chunks.append({
                            "id": cid,
                            "chunk_type": "fee_fact",
                            "doc_id": doc_id,
                            "source_file": pdf_path.name,
                            "sha256": sha,
                            "page": pno,
                            "table_index": t_idx,
                            "row_index": r,
                            "section_path": section_path,
                            "service": service_full,
                            "fee_code": fee_code,
                            "segment": seg,
                            "fee_value": fee_val,
                            "structured": struct,
                            "text": chunk_text(pdf_path.name, pno, section_path, service_full, fee_code, seg, fee_val),
                        })

                    r += 1

    return chunks


def build_docs_meta(pdf_paths: List[Path]) -> List[Dict[str, Any]]:
    meta = []
    for p in pdf_paths:
        meta.append({
            "doc_id": p.stem,
            "source_file": p.name,
            "sha256": sha256_file(p),
        })
    return meta


if __name__ == "__main__":
    # Lấy PDF từ data_dir trong settings
    data_dir = Path(settings.data_dir)
    pdfs = list(data_dir.glob("*.pdf"))
    
    if not pdfs:
        print(f"[WARN] No PDF files found in {data_dir}")
        exit(1)

    all_chunks: List[Dict[str, Any]] = []
    for pdf in pdfs:
        print(f"Processing: {pdf.name}")
        chunks = extract_fee_fact_chunks(pdf)
        print(f"  → {len(chunks)} chunks")
        all_chunks.extend(chunks)

    # Xuất ra artifacts directory
    out_dir = Path(settings.artifacts_dir)
    
    # Xuất JSONL (mỗi dòng 1 JSON object - tốt cho streaming)
    write_jsonl(all_chunks, out_dir / "rag_chunks_fee_fact_segmented_qa.jsonl")
    print(f"\n✅ Exported {len(all_chunks)} chunks to:")
    print(f"   - {out_dir / 'rag_chunks_fee_fact_segmented_qa.jsonl'}")
    
    # Xuất JSON thông thường (dễ đọc hơn)
    write_json(all_chunks, out_dir / "rag_chunks_fee_fact_segmented_qa.json")
    print(f"   - {out_dir / 'rag_chunks_fee_fact_segmented_qa.json'}")
    
    # Xuất metadata
    with (out_dir / "docs_meta.json").open("w", encoding="utf-8") as f:
        json.dump(build_docs_meta(pdfs), f, ensure_ascii=False, indent=2)
    print(f"   - {out_dir / 'docs_meta.json'}")
    
    print(f"\n📊 Summary:")
    print(f"   - Total PDFs: {len(pdfs)}")
    print(f"   - Total chunks: {len(all_chunks)}")
    print(f"   - Avg chunks/PDF: {len(all_chunks)/len(pdfs):.1f}")
