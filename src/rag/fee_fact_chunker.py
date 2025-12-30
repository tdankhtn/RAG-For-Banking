from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from src.rag.fee_tree_3level import extract_fee_tree_3level


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", s)


def norm_space(s: str) -> str:
    s = (s or "").replace("\t", " ")
    s = re.sub(r"[ ]+", " ", s)
    return s.strip()


def norm_key(s: str) -> str:
    s = strip_accents(s)
    s = norm_space(s)
    return s.upper()


def extract_year_fees(fee_text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    t = norm_space(fee_text)

    if re.search(r"nam\s+dau", norm_key(t)) and "MIEN PHI" in norm_key(t):
        out["year1_fee"] = "Miễn phí"

    m2 = re.search(
        r"(tu\s+nam\s+thu\s*2[^0-9]*)(\d[\d\.\, ]*\s*VND\s*/\s*nam)",
        norm_key(t),
    )
    if m2:
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


def build_service_label(
    service_detail: str, card_type: Optional[str], fee_type: Optional[str]
) -> str:
    detail = norm_space(service_detail)
    card = norm_space(card_type or "")
    fee = norm_space(fee_type or "")
    if norm_key(card) == "CHUNG":
        card = ""

    if detail and card:
        if norm_key(card) not in norm_key(detail):
            return norm_space(f"{detail} {card}")
        return detail
    if detail:
        return detail
    if card:
        return card
    return fee


def make_qa(service: str, segment: str, fee_value: str) -> str:
    service = norm_space(service)
    fee_value = norm_space(fee_value)

    q1 = f"Mức phí {service} cho {segment} là bao nhiêu?"
    if norm_key(segment).endswith("THUONG"):
        q2 = f"Mức phí {service} cho khách hàng thường là bao nhiêu?"
    else:
        q2 = f"Phí {service} dành cho {segment} là bao nhiêu?"

    return f"Q: {q1}\nQ: {q2}\nA: {fee_value}"


def chunk_text(
    doc_name: str,
    page: int,
    section_path: List[str],
    fee_type: Optional[str],
    card_type: Optional[str],
    service: str,
    fee_code: Optional[str],
    segment: str,
    fee_value: str,
) -> str:
    sp = " / ".join([s for s in section_path if s])
    head = f"[{doc_name} | page {page}]"
    parts = [
        head,
        (sp if sp else "").strip(),
        f"Nhóm phí: {fee_type}".strip() if fee_type else "",
        f"Loại thẻ: {card_type}".strip() if card_type else "",
        f"Dịch vụ: {service}".strip(),
        (f"Mã phí: {fee_code}" if fee_code else "").strip(),
        f"Phân khúc: {segment}".strip(),
        f"Mức phí: {fee_value}".strip(),
        "",
        make_qa(service, segment, fee_value),
    ]
    return "\n".join([p for p in parts if p])


def extract_fee_fact_chunks(
    pdf_path: Path, default_segment_if_missing: str = "KH thường"
) -> List[Dict[str, Any]]:
    data = extract_fee_tree_3level(
        pdf_path, default_segment_if_missing=default_segment_if_missing
    )
    doc_id = data["doc_id"]
    sha = data["sha256"]
    source_file = data["source_file"]
    tree = data["tree_3level"]

    chunks: List[Dict[str, Any]] = []
    for fee_type, card_map in tree.items():
        for card_type, seg_map in card_map.items():
            for seg, records in seg_map.items():
                for rec in records:
                    fee_value = norm_space(rec.get("fee_value", ""))
                    if not fee_value:
                        continue

                    service_detail = norm_space(rec.get("service_detail", ""))
                    service_label = build_service_label(
                        service_detail, card_type, fee_type
                    )
                    if not service_label:
                        continue

                    section = rec.get("section")
                    section_path = [section] if section else []
                    struct = extract_year_fees(fee_value)
                    fee_code = rec.get("fee_code")
                    page = rec.get("page") or 0
                    table_index = rec.get("table_index") or 0
                    row_index = rec.get("row_index") or 0

                    cid = (
                        f"{doc_id}::p{page:03d}::t{table_index:02d}::r{row_index:03d}"
                        f"::{segment_key(seg)}::{fee_code or 'NO_CODE'}"
                    )

                    chunks.append(
                        {
                            "id": cid,
                            "chunk_type": "fee_fact",
                            "doc_id": doc_id,
                            "source_file": source_file,
                            "sha256": sha,
                            "page": page,
                            "table_index": table_index,
                            "row_index": row_index,
                            "stt": rec.get("stt"),
                            "section": section,
                            "section_path": section_path,
                            "fee_type": fee_type,
                            "fee_type_stt": rec.get("fee_type_stt"),
                            "card_type": card_type,
                            "card_type_stt": rec.get("card_type_stt"),
                            "service": service_label,
                            "service_detail": service_detail,
                            "fee_code": fee_code,
                            "segment": seg,
                            "fee_value": fee_value,
                            "structured": struct,
                            "text": chunk_text(
                                source_file,
                                page,
                                section_path,
                                fee_type,
                                card_type,
                                service_label,
                                fee_code,
                                seg,
                                fee_value,
                            ),
                        }
                    )

    return chunks


def extract_fee_fact_documents(
    pdf_path: Path, default_segment_if_missing: str = "KH thường"
) -> List[Document]:
    chunks = extract_fee_fact_chunks(
        pdf_path, default_segment_if_missing=default_segment_if_missing
    )
    docs: List[Document] = []
    for chunk in chunks:
        metadata = {
            "source": str(pdf_path),
            "file_type": "fee_fact",
            "chunk_type": "fee_fact",
            "page": chunk.get("page"),
            "segment": chunk.get("segment"),
            "fee_code": chunk.get("fee_code"),
            "service": chunk.get("service"),
            "fee_type": chunk.get("fee_type"),
            "card_type": chunk.get("card_type"),
            "is_table": True,
        }
        docs.append(
            Document(
                page_content=json.dumps(chunk, ensure_ascii=False),
                metadata=metadata,
            )
        )
    return docs
