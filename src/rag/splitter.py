import re
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm


class TextSplitter:
    """
    Splitter optimized for FAQ-like Vietnamese content.

    - Prefer split by "Câu hỏi:" to keep Q&A pairs.
    - Fallback to RecursiveCharacterTextSplitter for large blocks.
    """

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def _split_by_qa_pattern(self, text: str) -> List[str]:
        pattern = r"Câu hỏi:"
        matches = list(re.finditer(pattern, text, re.IGNORECASE))

        if len(matches) < 2:
            return [text.strip()]

        chunks = []
        for i in range(len(matches)):
            start = matches[i].start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            chunk = text[start:end].strip()
            if chunk and len(chunk) > 20:
                chunks.append(chunk)

        return chunks

    def split(self, documents: List[Document]) -> List[Document]:
        result_chunks: List[Document] = []

        for doc in tqdm(documents, desc="Splitting documents"):
            if doc.metadata.get("is_table"):
                result_chunks.append(doc)
                continue

            text = doc.page_content
            qa_chunks = self._split_by_qa_pattern(text)

            for chunk_text in qa_chunks:
                if len(chunk_text) > 1500:
                    sub_doc = Document(page_content=chunk_text, metadata=doc.metadata)
                    sub_chunks = self.fallback_splitter.split_documents([sub_doc])
                    result_chunks.extend(sub_chunks)
                else:
                    result_chunks.append(
                        Document(page_content=chunk_text, metadata=doc.metadata)
                    )

        return result_chunks
