from pathlib import Path
import json

import streamlit as st
from langchain_core.documents import Document

from src.config import settings
from src.rag.llm import get_hf_llm
from src.rag.pipeline import OfflineRAG
from src.rag.vector_db import VectorDB


st.set_page_config(page_title="RAG Vietnamese QA", page_icon="📚", layout="wide")


def has_index(chroma_dir: Path) -> bool:
    return chroma_dir.exists() and any(chroma_dir.iterdir())


def load_chunks_from_json(json_path: Path) -> list:
    """Load chunks từ JSON file và convert sang LangChain Documents"""
    chunks = []
    
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    for item in data:
        doc = Document(
            page_content=item["text"],
            metadata={
                "id": item["id"],
                "doc_id": item["doc_id"],
                "source_file": item["source_file"],
                "page": item["page"],
                "service": item["service"],
                "service_detail": item.get("service_detail"),
                "fee_type": item.get("fee_type"),
                "card_type": item.get("card_type"),
                "section": item.get("section"),
                "segment": item["segment"],
                "fee_code": item.get("fee_code"),
                "fee_value": item["fee_value"],
                "chunk_type": item["chunk_type"],
            }
        )
        chunks.append(doc)
    
    return chunks


def build_index() -> int:
    """Build index từ JSON chunks (với Q&A pairs)"""
    json_path = settings.artifacts_dir / "rag_chunks_fee_fact_segmented_qa.json"
    
    if not json_path.exists():
        raise FileNotFoundError(
            f"JSON chunks không tồn tại tại {json_path}. "
            "Hãy chạy scripts/build_fee_fact_chunks.py trước."
        )
    
    # Load chunks từ JSON
    chunks = load_chunks_from_json(json_path)
    
    # Clear existing database
    if settings.chroma_dir.exists():
        import shutil
        shutil.rmtree(settings.chroma_dir)
    
    # Build new database
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    VectorDB(
        documents=chunks,
        embedding_model=settings.embedding_model,
        persist_dir=str(settings.chroma_dir),
    )

    return len(chunks)


@st.cache_resource(show_spinner=False)
def load_retriever():
    vdb = VectorDB(
        documents=None,
        embedding_model=settings.embedding_model,
        persist_dir=str(settings.chroma_dir),
    )
    return vdb.get_retriever(search_kwargs={"k": settings.retriever_k})


@st.cache_resource(show_spinner=False)
def load_rag_chain():
    llm = get_hf_llm(
        model_name=settings.llm_model,
        temperature=settings.llm_temperature,
        max_new_tokens=settings.llm_max_new_tokens,
        top_p=settings.llm_top_p,
    )
    retriever = load_retriever()
    rag = OfflineRAG(
        llm,
        use_reranking=settings.use_reranking,
        reranker_model=settings.reranker_model,
    )
    return rag.get_chain(retriever), retriever


st.title("RAG - Hỏi đáp Biểu phí Ngân hàng")
st.caption("Demo RAG với JSON chunks + Q&A pairs (LangChain + Chroma + Streamlit)")

with st.sidebar:
    st.subheader("Cấu hình")
    st.write(f"JSON chunks: `{settings.artifacts_dir / 'rag_chunks_fee_fact_segmented_qa.json'}`")
    st.write(f"Chroma dir: `{settings.chroma_dir}`")
    st.write(f"Embedding: `{settings.embedding_model}`")
    st.write(f"LLM: `{settings.llm_model}`")
    st.write(f"Reranking: `{settings.use_reranking}`")
    
    st.divider()
    st.caption("💡 Tip: Nếu chưa có JSON chunks, chạy:")
    st.code("python scripts/build_fee_fact_chunks.py", language="bash")

    if st.button("Xây dựng lại index từ JSON"):
        json_path = settings.artifacts_dir / "rag_chunks_fee_fact_segmented_qa.json"
        if not json_path.exists():
            st.error("JSON chunks không tồn tại. Hãy chạy build_fee_fact_chunks.py trước.")
        else:
            with st.spinner("Đang tạo index từ JSON chunks..."):
                try:
                    total_chunks = build_index()
                except Exception as exc:
                    st.error(f"Lỗi khi tạo index: {exc}")
                else:
                    st.cache_resource.clear()
                    st.success(f"✅ Đã tạo index với {total_chunks} chunks (có Q&A)")

if not has_index(settings.chroma_dir):
    st.warning(
        "⚠️ Chưa có vector index. "
        "Hãy chạy `python scripts/build_index_from_json.py` hoặc bấm nút trong sidebar."
    )
    st.stop()

rag_chain, retriever = load_rag_chain()

with st.form("qa_form"):
    question = st.text_area(
        "Câu hỏi",
        placeholder="Ví dụ: Phí thường niên thẻ F@STACCESS cho khách hàng thường là bao nhiêu?",
        height=120,
    )
    show_context = st.checkbox("Hiển thị context", value=False)
    submitted = st.form_submit_button("Gửi")

if submitted:
    if not question.strip():
        st.warning("Vui lòng nhập câu hỏi.")
    else:
        with st.spinner("Đang trả lời..."):
            answer = rag_chain.invoke(question)

        st.subheader("Câu trả lời")
        st.write(answer)

        if show_context:
            docs = retriever.invoke(question)
            with st.expander("Context đã retrieve (từ JSON chunks)"):
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"**Doc {i}**")
                    meta_bits = []
                    section = doc.metadata.get("section")
                    fee_type = doc.metadata.get("fee_type")
                    card_type = doc.metadata.get("card_type")
                    fee_code = doc.metadata.get("fee_code")
                    if section:
                        meta_bits.append(f"Section: {section}")
                    if fee_type:
                        meta_bits.append(f"Fee type: {fee_type}")
                    if card_type:
                        meta_bits.append(f"Card type: {card_type}")
                    meta_bits.extend(
                        [
                            f"Segment: {doc.metadata.get('segment')}",
                            f"Page: {doc.metadata.get('page')}",
                        ]
                    )
                    if fee_code:
                        meta_bits.append(f"Fee code: {fee_code}")
                    st.caption(" | ".join(meta_bits))
                    st.write(doc.page_content)
