"""
Build vector database index từ JSON chunks đã chunking sẵn.
Tự động tìm và load TẤT CẢ file JSON trong thư mục artifacts.
"""

import json
from pathlib import Path
from typing import List

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
            page_content=item.get("text", ""),
            metadata={
                "id": item.get("id"),
                "doc_id": item.get("doc_id"),
                "source_file": item.get("source_file"),
                "page": item.get("page"),
                "service": item.get("service"),
                "service_detail": item.get("service_detail"),
                "fee_type": item.get("fee_type"),
                "card_type": item.get("card_type"),
                "section": item.get("section"),
                "segment": item.get("segment"),
                "fee_code": item.get("fee_code"),
                "fee_value": item.get("fee_value"),
                "topic": item.get("topic"),
                "heading": item.get("heading"),
                "question": item.get("question"),
                "answer": item.get("answer"),
                "chunk_type": item.get("chunk_type"),
            }
        )
        chunks.append(doc)
    
    return chunks


def find_all_chunk_json_files(artifacts_dir: Path) -> List[Path]:
    """
    Tìm tất cả file JSON chunks trong thư mục artifacts.
    Bỏ qua file metadata như docs_meta.json
    """
    json_files = []
    
    # Pattern: rag_chunks_*.json (không lấy .jsonl)
    for json_file in artifacts_dir.glob("rag_chunks_*.json"):
        # Bỏ qua file .jsonl
        if json_file.suffix == ".json":
            json_files.append(json_file)
    
    return sorted(json_files)


def build_index_from_all_json(
    artifacts_dir: Path = None,
    clear_existing: bool = True
) -> None:
    """
    Build vector database từ TẤT CẢ JSON chunks trong artifacts
    
    Args:
        artifacts_dir: Thư mục chứa JSON files (default: settings.artifacts_dir)
        clear_existing: Xóa database cũ trước khi build mới
    """
    if artifacts_dir is None:
        artifacts_dir = settings.artifacts_dir
    
    # Tìm tất cả file JSON chunks
    json_files = find_all_chunk_json_files(artifacts_dir)
    
    if not json_files:
        raise FileNotFoundError(
            f"Không tìm thấy file JSON chunks trong {artifacts_dir}\n"
            f"Hãy chạy scripts/build_fee_fact_chunks.py và/hoặc các script chunking khác trước."
        )
    
    print(f"📂 Tìm thấy {len(json_files)} file JSON chunks:")
    for f in json_files:
        print(f"   - {f.name}")
    
    # Load chunks từ TẤT CẢ files
    all_chunks = []
    for json_path in json_files:
        print(f"\n📄 Loading: {json_path.name}")
        chunks = load_chunks_from_json(json_path)
        print(f"   ✅ {len(chunks)} chunks")
        all_chunks.extend(chunks)
    
    print(f"\n📦 Tổng cộng: {len(all_chunks)} chunks từ {len(json_files)} files")
    
    # Clear existing database nếu cần
    if clear_existing and settings.chroma_dir.exists():
        print(f"\n🗑️  Xóa database cũ tại {settings.chroma_dir}")
        import shutil
        shutil.rmtree(settings.chroma_dir)
    
    # Build new database
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔨 Building vector database...")
    vdb = VectorDB(
        documents=all_chunks,
        embedding_model=settings.embedding_model,
        persist_dir=str(settings.chroma_dir),
    )
    
    print(f"\n{'='*60}")
    print(f"✅ INDEX BUILT SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"   📦 Total chunks: {len(all_chunks)}")
    print(f"   📄 From files: {len(json_files)}")
    print(f"   💾 Database path: {settings.chroma_dir}")
    print(f"   🔍 Embedding model: {settings.embedding_model}")
    
    # Test queries
    print(f"\n🧪 Testing retrieval...")
    retriever = vdb.get_retriever(search_kwargs={"k": 3})
    
    test_queries = [
        "Tôi không tải được ứng dụng mới Techcombank Mobile?",
        "Phí thường niên thẻ cho khách hàng thường là bao nhiêu?",
    ]
    
    for query in test_queries:
        results = retriever.invoke(query)
        print(f"\n   Query: '{query[:50]}...'")
        print(f"   Retrieved: {len(results)} docs")
        if results:
            print(f"   Top result: {results[0].page_content[:80]}...")


# Backward compatibility: giữ lại hàm cũ
def build_index_from_json(json_path: Path, clear_existing: bool = True) -> None:
    """[Deprecated] Dùng build_index_from_all_json() thay thế"""
    print(f"⚠️  Đang load từ 1 file: {json_path}")
    print(f"   Tip: Dùng --all để load tất cả JSON files trong artifacts\n")
    
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    chunks = load_chunks_from_json(json_path)
    print(f"✅ Loaded {len(chunks)} chunks")
    
    if clear_existing and settings.chroma_dir.exists():
        import shutil
        shutil.rmtree(settings.chroma_dir)
    
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    
    VectorDB(
        documents=chunks,
        embedding_model=settings.embedding_model,
        persist_dir=str(settings.chroma_dir),
    )
    
    print(f"✅ INDEX BUILT with {len(chunks)} chunks")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build vector database from JSON chunks"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=True,
        help="Load TẤT CẢ file JSON trong artifacts (default)"
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Path to specific JSON file (nếu chỉ muốn load 1 file)"
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Giữ database cũ (append mode)"
    )
    
    args = parser.parse_args()
    
    # Nếu chỉ định file cụ thể, load file đó
    if args.json:
        build_index_from_json(
            json_path=args.json,
            clear_existing=not args.keep_existing
        )
    else:
        # Mặc định: load TẤT CẢ files
        build_index_from_all_json(
            clear_existing=not args.keep_existing
        )


if __name__ == "__main__":
    main()
