import argparse

from src.config import settings
from src.rag.pipeline import OfflineRAG
from src.rag.vector_db import VectorDB


def debug_rag_process(question: str, use_reranking: bool) -> None:
    vdb = VectorDB(
        documents=None,
        embedding_model=settings.embedding_model,
        persist_dir=str(settings.chroma_dir),
    )
    retriever = vdb.get_retriever(search_kwargs={"k": settings.retriever_k})

    rag = OfflineRAG(
        llm=None, use_reranking=use_reranking, reranker_model=settings.reranker_model
    )

    retrieved_docs = retriever.invoke(question)

    print("=" * 80)
    print(f"Query: {question}")
    print("=" * 80)

    print(f"\nRetrieved {len(retrieved_docs)} docs:\n")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"--- Doc {i} ---")
        print(f"Content: {doc.page_content[:300]}...")
        print(f"Metadata: {doc.metadata}\n")

    formatted = []
    seen = set()
    for doc in retrieved_docs:
        content = doc.page_content.strip()
        if content and len(content) > 40 and content not in seen:
            formatted.append(content)
            seen.add(content)

    context = "\n\n".join(formatted)

    print("Context sent to model:")
    print("-" * 80)
    print(context)
    print("-" * 80)

    final_prompt = rag.prompt.format(context=context, question=question)
    print("\nFinal prompt:")
    print("-" * 80)
    print(final_prompt)
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Debug RAG retrieval and prompt.")
    parser.add_argument("question", help="Question to debug")
    parser.add_argument(
        "--no-rerank", action="store_true", help="Disable reranking"
    )
    args = parser.parse_args()

    debug_rag_process(args.question, use_reranking=not args.no_rerank)


if __name__ == "__main__":
    main()
