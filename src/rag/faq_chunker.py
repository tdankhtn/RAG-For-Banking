from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.rag.fee_tree_3level import sha256_file


QA_MARKER_RE = re.compile(r"Câu hỏi:", re.IGNORECASE)
TOPIC_MARKER_RE = re.compile(r"Chủ đề:", re.IGNORECASE)


def norm_space(s: str) -> str:
    s = (s or "").replace("\t", " ")
    s = re.sub(r"[ ]+", " ", s)
    return s.strip()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def is_table_like_line(line: str) -> bool:
    if not line:
        return False
    lowered = line.lower()
    if "câu hỏi" in lowered or "chủ đề" in lowered:
        return False
    if line.lstrip().startswith(("-", "•", "*")):
        return False
    if "|" in line or "\t" in line:
        return True
    if re.search(r"[-_]{3,}", line):
        return True
    cols = [c for c in re.split(r"\s{2,}", line) if c]
    if len(cols) >= 3 and len(line) >= 25:
        return True
    digits = sum(ch.isdigit() for ch in line)
    letters = sum(ch.isalpha() for ch in line)
    if digits >= 6 and digits > letters * 2:
        return True
    return False


def filter_table_lines(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines()]
    kept = [ln for ln in lines if ln and not is_table_like_line(ln)]
    return "\n".join(kept)


def clean_page_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.splitlines()]
    kept = []
    for line in lines:
        if not line.strip():
            continue
        if is_table_like_line(line):
            continue
        kept.append(line.strip())
    cleaned = "\n".join(kept)
    return normalize_text(cleaned)


def extract_page_texts(pdf_path: Path) -> List[Tuple[int, str]]:
    pages: List[Tuple[int, str]] = []
    try:
        import pdfplumber

        with pdfplumber.open(str(pdf_path)) as pdf:
            for pno, page in enumerate(pdf.pages, start=1):
                raw = page.extract_text() or ""
                cleaned = clean_page_text(raw)
                pages.append((pno, cleaned))
        return pages
    except ImportError:
        pass

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("pypdf or pdfplumber is required for FAQ extraction") from exc

    reader = PdfReader(str(pdf_path))
    for pno, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        cleaned = clean_page_text(raw)
        pages.append((pno, cleaned))
    return pages


def split_question_answer(block: str) -> Tuple[str, str]:
    cleaned = filter_table_lines(normalize_text(block))
    cleaned = norm_space(cleaned)
    if not cleaned:
        return "", ""

    qmark = cleaned.find("?")
    if qmark != -1:
        question = cleaned[: qmark + 1].strip()
        answer = cleaned[qmark + 1 :].strip()
        return question, answer

    period = cleaned.find(".")
    if period != -1:
        question = cleaned[: period + 1].strip()
        answer = cleaned[period + 1 :].strip()
        return question, answer

    return cleaned, ""


def extract_topic(text: str) -> Optional[str]:
    topic = normalize_text(text)
    topic = re.split(QA_MARKER_RE, topic, maxsplit=1)[0]
    topic = topic.split("\n", 1)[0]
    topic = norm_space(topic)
    if len(topic) < 3:
        return None
    return topic


def infer_heading(prefix: str) -> Optional[str]:
    if not prefix:
        return None
    window = prefix.replace("\n", " ").strip()
    if not window:
        return None
    boundary = max(window.rfind("."), window.rfind("!"), window.rfind("?"))
    candidate = window[boundary + 1 :].strip()
    if not candidate:
        return None
    if len(candidate) > 80:
        return None
    lowered = candidate.lower()
    if "câu hỏi" in lowered or "chủ đề" in lowered:
        return None
    if candidate.startswith(("-", "•", "*")):
        return None
    if re.search(r"\d", candidate):
        return None
    if candidate.endswith(":"):
        candidate = candidate[:-1].strip()
    if not candidate:
        return None
    return candidate


def chunk_text(
    doc_name: str,
    page: int,
    topic: Optional[str],
    heading: Optional[str],
    question: str,
    answer: str,
) -> str:
    parts = [
        f"[{doc_name} | page {page}]",
        (f"Chủ đề: {topic}" if topic else "").strip(),
        (f"Mục: {heading}" if heading else "").strip(),
        f"Câu hỏi: {question}".strip(),
        (f"Trả lời: {answer}" if answer else "").strip(),
    ]
    return "\n".join([p for p in parts if p])


def extract_faq_chunks(pdf_path: Path) -> List[Dict[str, Any]]:
    pages = extract_page_texts(pdf_path)
    doc_id = pdf_path.stem
    sha = sha256_file(pdf_path)

    parts: List[str] = []
    offsets: List[Tuple[int, int]] = []
    pos = 0
    for page, text in pages:
        offsets.append((pos, page))
        parts.append(text)
        pos += len(text) + 1
    full_text = "\n".join(parts)

    markers = list(re.finditer(r"(Chủ đề:|Câu hỏi:)", full_text, re.IGNORECASE))
    if not markers:
        return []

    def page_for_offset(offset: int) -> int:
        page = offsets[0][1] if offsets else 1
        for start, pno in offsets:
            if offset >= start:
                page = pno
            else:
                break
        return page

    current_topic: Optional[str] = None
    chunks: List[Dict[str, Any]] = []
    for i, marker in enumerate(markers):
        label = marker.group(1).lower()
        next_start = markers[i + 1].start() if i + 1 < len(markers) else len(full_text)

        if "chủ đề" in label:
            topic_text = full_text[marker.end() : next_start]
            topic = extract_topic(topic_text)
            if topic:
                current_topic = topic
            continue

        block = full_text[marker.end() : next_start]
        question, answer = split_question_answer(block)
        if len(question) < 5:
            continue

        prefix_start = max(0, marker.start() - 160)
        heading = infer_heading(full_text[prefix_start : marker.start()])

        page = page_for_offset(marker.start())
        chunk_id = f"{doc_id}::p{page:03d}::faq::{len(chunks):03d}"
        text = chunk_text(
            pdf_path.name,
            page,
            current_topic,
            heading,
            question,
            answer,
        )

        chunks.append(
            {
                "id": chunk_id,
                "chunk_type": "faq",
                "doc_id": doc_id,
                "source_file": pdf_path.name,
                "sha256": sha,
                "page": page,
                "topic": current_topic,
                "heading": heading,
                "question": question,
                "answer": answer,
                "text": text,
            }
        )

    return chunks
