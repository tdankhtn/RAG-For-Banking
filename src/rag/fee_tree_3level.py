from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROMAN_RE = re.compile(r"^[IVXLC]+$", re.IGNORECASE)
FEE_CODE_RE = re.compile(r"\bCN[-]?\d{2,5}[-]?\d{2,4}\b", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", s)


def norm_space(s: str) -> str:
    s = (s or "").replace("\t", " ")
    s = re.sub(r"[ ]+", " ", s)
    return s.strip()


def norm_key(s: str) -> str:
    return norm_space(strip_accents(s)).upper()


def norm_cell(x: Any) -> str:
    if x is None:
        return ""
    return norm_space(str(x).replace("\n", " "))


def any_fee_code(text: str) -> Optional[str]:
    m = FEE_CODE_RE.search(text or "")
    return m.group(0).upper() if m else None


def normalize_stt(s: str) -> str:
    """
    - keep roman numeral (I, II, III...)
    - join STT broken by line: "7.2.1 0" -> "7.2.10"
    """
    s = norm_space(s or "")
    if not s:
        return ""
    if ROMAN_RE.fullmatch(s.upper()):
        return s.upper()

    s2 = re.sub(r"\s+", "", s)
    s2 = re.sub(r"[^0-9\.]", "", s2)
    s2 = s2.strip(".")
    return s2


def stt_dot_count(stt: str) -> Optional[int]:
    if not stt:
        return None
    if ROMAN_RE.fullmatch(stt.upper()):
        return None
    if re.fullmatch(r"\d+(?:\.\d+)*", stt):
        return stt.count(".")
    return None


SEGMENT_CANON = {
    "KH PRIVATE": "KH Private",
    "KH PRIORITY": "KH Priority",
    "KH INSPIRE": "KH Inspire",
    "KH THUONG": "KH thường",
    "KH THƯỜNG": "KH thường",
}


def canonical_segment_from_header(cell_text: str) -> Optional[str]:
    hk = norm_key(cell_text)
    for raw, canon in SEGMENT_CANON.items():
        if raw in hk:
            return canon
    return None


@dataclass
class HeaderMap:
    stt_idx: int
    fee_code_idx: Optional[int]
    service_idx: int
    fee_value_idx: Optional[int]
    segment_idxs: Dict[str, int]
    min_idx: Optional[int]
    max_idx: Optional[int]
    type_idx: Optional[int]


def build_header_map(rows: List[List[str]], start_i: int) -> Tuple[Optional[HeaderMap], int]:
    header_rows: List[List[str]] = []
    i = start_i

    while i < len(rows) and len(header_rows) < 3:
        hk = norm_key(" ".join(rows[i]))
        if "STT" in hk and ("DICH VU" in hk or "DỊCH VỤ" in " ".join(rows[i]).upper()):
            header_rows.append(rows[i])
            i += 1
            continue
        if header_rows and ("KH" in hk or "MUC PHI" in hk or "MỨC PHÍ" in " ".join(rows[i]).upper()):
            header_rows.append(rows[i])
            i += 1
            continue
        break

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

    stt_idx = 0
    fee_code_idx = None
    service_idx = 0
    fee_value_idx = None
    min_idx = None
    max_idx = None
    type_idx = None
    segment_idxs: Dict[str, int] = {}

    for idx, h in enumerate(combined_headers):
        hk = norm_key(h)
        if "STT" in hk:
            stt_idx = idx
        if "MA PHI" in hk or "MÃ PHÍ" in h.upper():
            fee_code_idx = idx
        if "DICH VU" in hk:
            service_idx = idx
        if "MUC PHI" in hk:
            fee_value_idx = idx
        if "TOI THIEU" in hk:
            min_idx = idx
        if "TOI DA" in hk:
            max_idx = idx
        if hk == "LOAI" or " LOAI" in hk or "LOẠI" in h.upper():
            type_idx = idx

        seg = canonical_segment_from_header(h)
        if seg:
            segment_idxs[seg] = idx

    return (
        HeaderMap(
            stt_idx=stt_idx,
            fee_code_idx=fee_code_idx,
            service_idx=service_idx,
            fee_value_idx=fee_value_idx,
            segment_idxs=segment_idxs,
            min_idx=min_idx,
            max_idx=max_idx,
            type_idx=type_idx,
        ),
        i,
    )


def join_range(row: List[str], start: int, end: int) -> str:
    parts = []
    for i in range(start, min(end + 1, len(row))):
        v = row[i].strip()
        if v:
            parts.append(v)
    return norm_space(" ".join(parts))


def field_ranges(hm: HeaderMap, row_len: int) -> Dict[str, Tuple[int, int]]:
    starts: List[Tuple[str, int]] = []
    starts.append(("stt", hm.stt_idx))
    if hm.fee_code_idx is not None:
        starts.append(("fee_code", hm.fee_code_idx))
    starts.append(("service", hm.service_idx))

    if hm.segment_idxs:
        for seg, idx in hm.segment_idxs.items():
            starts.append((f"seg::{seg}", idx))
    elif hm.fee_value_idx is not None:
        starts.append(("fee_value", hm.fee_value_idx))

    for name, idx in [("min", hm.min_idx), ("max", hm.max_idx), ("type", hm.type_idx)]:
        if idx is not None:
            starts.append((name, idx))

    starts_sorted = sorted(starts, key=lambda x: x[1])
    ranges: Dict[str, Tuple[int, int]] = {}

    for j, (name, idx) in enumerate(starts_sorted):
        next_idx = starts_sorted[j + 1][1] if j + 1 < len(starts_sorted) else row_len
        end = next_idx - 1
        ranges[name] = (idx, end)

    return ranges


def infer_default_card_type_from_section(section_text: str) -> Optional[str]:
    if not section_text:
        return None
    up = norm_key(section_text)

    if "F@STACCESS" in section_text.upper():
        return "Thẻ F@STACCESS"

    if "THE TIN DUNG" in up or "THẺ TÍN DỤNG" in section_text.upper():
        return "Thẻ tín dụng"

    m = re.search(r"THẺ\s+([A-Z0-9@]+)", strip_accents(section_text).upper())
    if m:
        return "Thẻ " + m.group(1)
    return None


def extract_fee_tree_3level(
    pdf_path: Path,
    default_segment_if_missing: str = "KH thường",
) -> Dict[str, Any]:
    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError("pdfplumber is required for fee tree extraction") from exc

    lines_settings = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "edge_min_length": 3,
        "intersection_tolerance": 3,
        "text_tolerance": 3,
    }
    text_settings = {
        "vertical_strategy": "text",
        "horizontal_strategy": "text",
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "intersection_tolerance": 3,
        "text_tolerance": 3,
    }

    doc_id = pdf_path.stem
    sha = sha256_file(pdf_path)

    tree: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = {}

    current_section: Optional[str] = None
    current_section_stt: Optional[str] = None
    default_card_type: Optional[str] = None

    current_fee_type: Optional[str] = None
    current_fee_type_stt: Optional[str] = None

    current_card_type: Optional[str] = None
    current_card_type_stt: Optional[str] = None

    last_stt: str = ""
    last_dot: Optional[int] = None
    last_hm: Optional[HeaderMap] = None

    with pdfplumber.open(str(pdf_path)) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables(table_settings=lines_settings) or []
            if not tables:
                tables = page.extract_tables(table_settings=text_settings) or []

            for t_idx, table in enumerate(tables):
                rows = [[norm_cell(c) for c in r] for r in table if r]
                if not rows:
                    continue

                r = 0
                hm: Optional[HeaderMap] = None

                while r < len(rows):
                    row = rows[r]

                    if hm is None:
                        hm_try, r_next = build_header_map(rows, r)
                        if hm_try:
                            hm = hm_try
                            last_hm = hm_try
                            r = r_next
                            continue

                        if last_hm and last_hm.stt_idx < len(row) and last_hm.service_idx < len(row):
                            hm = last_hm
                        else:
                            r += 1
                            continue

                    ranges = field_ranges(hm, len(row))

                    stt_raw = join_range(row, *ranges["stt"])
                    stt = normalize_stt(stt_raw)

                    cont = False
                    if not stt:
                        stt = last_stt
                        dot = last_dot
                        cont = True
                    else:
                        last_stt = stt
                        dot = stt_dot_count(stt)
                        last_dot = dot

                    service_part = join_range(row, *ranges["service"])

                    if stt and ROMAN_RE.fullmatch(stt.upper()):
                        if cont and current_section_stt == stt and current_section:
                            if service_part:
                                current_section = norm_space(current_section + " " + service_part)
                        else:
                            current_section = service_part or current_section
                            current_section_stt = stt
                            if not cont:
                                default_card_type = infer_default_card_type_from_section(current_section or "")
                                current_fee_type = None
                                current_fee_type_stt = None
                                current_card_type = None
                                current_card_type_stt = None

                        r += 1
                        continue

                    fee_code = None
                    if "fee_code" in ranges and hm.fee_code_idx is not None:
                        fee_code = any_fee_code(join_range(row, *ranges["fee_code"]))
                    if not fee_code:
                        fee_code = any_fee_code(" ".join(row))

                    seg_vals: Dict[str, str] = {}
                    if hm.segment_idxs:
                        for seg in hm.segment_idxs:
                            key = f"seg::{seg}"
                            if key in ranges:
                                val = join_range(row, *ranges[key])
                                if val:
                                    seg_vals[seg] = val
                    else:
                        if "fee_value" in ranges:
                            val = join_range(row, *ranges["fee_value"])
                            if val:
                                seg_vals[default_segment_if_missing] = val

                    def add_records(card_key: str, detail_service: str) -> None:
                        assert current_fee_type is not None
                        for seg, val in seg_vals.items():
                            rec = {
                                "stt": stt,
                                "fee_type": current_fee_type,
                                "fee_type_stt": current_fee_type_stt,
                                "card_type": card_key,
                                "card_type_stt": current_card_type_stt,
                                "segment": seg,
                                "fee_code": fee_code,
                                "service_detail": detail_service,
                                "fee_value": val,
                                "page": pno,
                                "section": current_section,
                                "table_index": t_idx,
                                "row_index": r,
                            }
                            (
                                tree.setdefault(current_fee_type, {})
                                .setdefault(card_key, {})
                                .setdefault(seg, [])
                                .append(rec)
                            )

                    if dot == 0:
                        if service_part:
                            if cont and current_fee_type_stt == stt and current_fee_type:
                                current_fee_type = norm_space(current_fee_type + " " + service_part)
                            else:
                                current_fee_type = service_part
                                current_fee_type_stt = stt
                                current_card_type = None
                                current_card_type_stt = None

                        if seg_vals and current_fee_type:
                            card_key = default_card_type or "Chung"
                            add_records(card_key, service_part or current_fee_type)

                    elif dot == 1:
                        if service_part:
                            if cont and current_card_type_stt == stt and current_card_type:
                                current_card_type = norm_space(current_card_type + " " + service_part)
                            else:
                                current_card_type = service_part
                                current_card_type_stt = stt

                        if seg_vals and current_fee_type:
                            card_key = current_card_type or default_card_type or "Chung"
                            add_records(card_key, service_part or current_card_type or "")

                    elif dot is not None and dot >= 2:
                        if seg_vals and current_fee_type:
                            card_key = current_card_type or default_card_type or "Chung"
                            add_records(card_key, service_part)

                    r += 1

    return {
        "doc_id": doc_id,
        "source_file": pdf_path.name,
        "sha256": sha,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "tree_3level": tree,
    }
