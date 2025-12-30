import argparse

from src.config import settings
from src.rag.loader import SimpleLoader
from src.rag.splitter import TextSplitter
from src.rag.vector_db import VectorDB


def write_split_docs(split_docs, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        file.write(f"Tong so chunks: {len(split_docs)}\n")
        file.write("=" * 80 + "\n\n")

        for idx, doc in enumerate(split_docs, 1):
            file.write(f"CHUNK {idx}\n")
            file.write("-" * 80 + "\n")
            file.write(f"Noi dung:\n{doc.page_content}\n\n")
            file.write(f"Metadata: {doc.metadata}\n")
            file.write("=" * 80 + "\n\n")


def build_index(save_splits: bool = True):
    loader = SimpleLoader(
        include_tables=settings.ingest_tables,
        table_rows_per_chunk=settings.table_rows_per_chunk,
        fee_fact_chunking=settings.fee_fact_chunking,
        fee_fact_patterns=settings.fee_fact_patterns,
        fee_fact_include_raw_text=settings.fee_fact_include_raw_text,
    )
    text_splitter = TextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )

    raw_docs = loader.load_dir(settings.data_dir, file_types=settings.ingest_file_types)
    split_docs = text_splitter.split(raw_docs)

    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    VectorDB(
        documents=split_docs,
        embedding_model=settings.embedding_model,
        persist_dir=str(settings.chroma_dir),
    )

    if save_splits:
        write_split_docs(split_docs, settings.split_docs_path)

    print(f"Built index with {len(split_docs)} chunks")
    print(f"Chroma path: {settings.chroma_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build Chroma vector index.")
    parser.add_argument(
        "--no-save-splits",
        action="store_true",
        help="Do not write split_documents.txt",
    )
    args = parser.parse_args()
    build_index(save_splits=not args.no_save_splits)


if __name__ == "__main__":
    main()
