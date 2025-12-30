"""
Build vector database index từ JSON chunks đã chunking sẵn.
Thay vì chunking từ PDF, load trực tiếp từ JSON với Q&A pairs.
"""

import json
from pathlib import Path

from langchain_core.documents import Document

from src.config import settings
from src.rag.vector_db import VectorDB


def load_chunks_from_json(json_path: Path) -> list[Document]:
    """Load chunks từ JSON file và convert sang LangChain Documents"""
    chunks = []
    
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    for item in data:
        # Sử dụng trường 'text' làm page_content (đã có Q&A)
        doc = Document(
            page_content=item["text"],
            metadata={
                "id": item["id"],
                "doc_id": item["doc_id"],
                "source_file": item["source_file"],
                "page": item["page"],
                "service": item["service"],
                "segment": item["segment"],
                "fee_code": item.get("fee_code"),
                "fee_value": item["fee_value"],
                "chunk_type": item["chunk_type"],
            }
        )
        chunks.append(doc)
    
    return chunks


def build_index_from_json(
    json_path: Path,
    clear_existing: bool = True
) -> None:
    """
    Build vector database từ JSON chunks
    
    Args:
        json_path: Path to JSON file với chunks
        clear_existing: Xóa database cũ trước khi build mới
    """
    print(f"📂 Loading chunks from: {json_path}")
    
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    # Load chunks
    chunks = load_chunks_from_json(json_path)
    print(f"✅ Loaded {len(chunks)} chunks")
    
    # Clear existing database nếu cần
    if clear_existing and settings.chroma_dir.exists():
        print(f"🗑️  Clearing existing database at {settings.chroma_dir}")
        import shutil
        shutil.rmtree(settings.chroma_dir)
    
    # Build new database
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🔨 Building vector database...")
    vdb = VectorDB(
        documents=chunks,
        embedding_model=settings.embedding_model,
        persist_dir=str(settings.chroma_dir),
    )
    
    print(f"\n✅ INDEX BUILT SUCCESSFULLY!")
    print(f"   📦 Total chunks: {len(chunks)}")
    print(f"   💾 Database path: {settings.chroma_dir}")
    print(f"   🔍 Embedding model: {settings.embedding_model}")
    
    # Test query
    print(f"\n🧪 Testing retrieval...")
    retriever = vdb.get_retriever(search_kwargs={"k": 3})
    test_query = "phí thường niên thẻ"
    results = retriever.invoke(test_query)
    
    print(f"   Query: '{test_query}'")
    print(f"   Retrieved: {len(results)} docs")
    if results:
        print(f"   Top result preview: {results[0].page_content[:100]}...")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build vector database from JSON chunks"
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=settings.artifacts_dir / "rag_chunks_fee_fact_segmented_qa.json",
        help="Path to JSON chunks file"
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Keep existing database (append mode)"
    )
    
    args = parser.parse_args()
    
    build_index_from_json(
        json_path=args.json,
        clear_existing=not args.keep_existing
    )


if __name__ == "__main__":
    main()
