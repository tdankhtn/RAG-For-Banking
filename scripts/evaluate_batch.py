import argparse

from src.config import settings
from src.evaluation.gemini_evaluator import GeminiRAGEvaluator
from src.rag.llm import get_hf_llm
from src.rag.pipeline import OfflineRAG
from src.rag.vector_db import VectorDB


def build_rag_chain():
    llm = get_hf_llm(
        model_name=settings.llm_model,
        temperature=settings.llm_temperature,
        max_new_tokens=settings.llm_max_new_tokens,
        top_p=settings.llm_top_p,
    )
    vdb = VectorDB(
        documents=None,
        embedding_model=settings.embedding_model,
        persist_dir=str(settings.chroma_dir),
    )
    retriever = vdb.get_retriever(search_kwargs={"k": settings.retriever_k})
    rag = OfflineRAG(
        llm,
        use_reranking=settings.use_reranking,
        reranker_model=settings.reranker_model,
    )
    return rag.get_chain(retriever), retriever


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG with Gemini batch metrics.")
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model name",
    )
    args = parser.parse_args()

    if not settings.gemini_api_key:
        raise SystemExit("Missing GEMINI_API_KEY in environment or .env")

    rag_chain, retriever = build_rag_chain()

    test_cases_with_gt = [
        {
            "question": "Tôi phải làm gì khi bị mất điện thoại cài đặt ứng dụng Techcombank Mobile?",
            "answer": rag_chain.invoke(
                "Tôi phải làm gì khi bị mất điện thoại cài đặt ứng dụng Techcombank Mobile?"
            ),
            "contexts": [
                doc.page_content
                for doc in retriever.invoke(
                    "Tôi phải làm gì khi bị mất điện thoại cài đặt ứng dụng Techcombank Mobile?"
                )
            ],
            "ground_truth": "liên hệ tổng đài 1800 588 822 hoặc tới CN/PGD Techcombank gần nhất",
        },
        {
            "question": "Tôi có được gạch nợ ngay lập tức sau khi thanh toán hóa đơn thành công không?",
            "answer": rag_chain.invoke(
                "Tôi có được gạch nợ ngay lập tức sau khi thanh toán hóa đơn thành công không?"
            ),
            "contexts": [
                doc.page_content
                for doc in retriever.invoke(
                    "Tôi có được gạch nợ ngay lập tức khi thanh toán hóa đơn không?"
                )
            ],
            "ground_truth": "Có, hóa đơn sẽ được gạch nợ ngay lập tức bên nhà cung cấp sau khi thanh toán hóa đơn thành công.",
        },
        {
            "question": "Tôi có thể chuyển tối đa bao nhiêu tiền bằng mã QR trên Techcombank Mobile?",
            "answer": rag_chain.invoke(
                "Tôi có thể chuyển tối đa bao nhiêu tiền bằng mã QR trên Techcombank Mobile?"
            ),
            "contexts": [
                doc.page_content
                for doc in retriever.invoke(
                    "Tôi có thể chuyển tối đa bao nhiêu tiền bằng mã QR trên Techcombank Mobile?"
                )
            ],
            "ground_truth": "theo quy định hạn mức chuyển tiền 24/7 của Techcombank.",
        },
        {
            "question": "Làm thế nào để chuyển khoản qua số điện thoại?",
            "answer": rag_chain.invoke("Làm thế nào để chuyển khoản qua số điện thoại?"),
            "contexts": [
                doc.page_content
                for doc in retriever.invoke(
                    "Làm thế nào để chuyển khoản qua số điện thoại?"
                )
            ],
            "ground_truth": "cần liên kết số điện thoại với một trong các tài khoản thanh toán bằng cách chọn Liên kết số điện thoại trong mục Cài đặt.",
        },
    ]

    gemini_evaluator = GeminiRAGEvaluator(
        api_key=settings.gemini_api_key,
        model_name=args.model,
    )

    df_results = gemini_evaluator.evaluate_batch(test_cases_with_gt)

    print("\nBang ket qua chi tiet:")
    print(df_results.to_string(index=False))


if __name__ == "__main__":
    main()
