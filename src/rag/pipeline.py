import re
from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from sentence_transformers import CrossEncoder


class FocusedAnswerParser(StrOutputParser):
    def parse(self, text: str) -> str:
        text = text.strip()
        if "[TRẢ LỜI]:" in text:
            answer = text.split("[TRẢ LỜI]:")[-1].strip()
        else:
            answer = text
        answer = re.sub(r"^\s*[\-\*]\s*", "", answer, flags=re.MULTILINE)
        answer = re.sub(r"\n+", " ", answer)
        lines = [
            line.strip()
            for line in answer.split(". ")
            if line.strip() and len(line.strip()) > 5
        ]
        return ". ".join(lines[:5]) + ("." if lines else "")


class OfflineRAG:
    def __init__(
        self,
        llm,
        use_reranking: bool = True,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.llm = llm
        self.use_reranking = use_reranking

        if use_reranking:
            self.reranker = CrossEncoder(reranker_model)
        else:
            self.reranker = None

        self.prompt = PromptTemplate.from_template(
            """Bạn là trợ lý AI phân tích tài liệu tiếng Việt.

[TÀI LIỆU]:
{context}

[CÂU HỎI]:
{question}

Hãy trả lời dựa trên tài liệu. Nếu tài liệu không có thông tin, nói rõ "Không có thông tin".
Trả lời đầy đủ thông tin (3-5 câu chi tiết), không thêm bất kỳ chi tiết nào ngoài tài liệu.

[TRẢ LỜI]:"""
        )
        self.answer_parser = FocusedAnswerParser()

    def rerank_docs(self, question: str, docs: List, top_k: int = 4):
        if not self.use_reranking or not docs or not self.reranker:
            return docs

        pairs = [[question, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)

        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _score in scored_docs[:top_k]]

    def get_chain(self, retriever):
        class RetrieveAndFormat:
            def __init__(self, retriever, rag_instance):
                self.retriever = retriever
                self.rag = rag_instance

            def invoke(self, question: str) -> str:
                docs = self.retriever.invoke(question)

                if self.rag.use_reranking and docs:
                    docs = self.rag.rerank_docs(question, docs, top_k=4)

                formatted = []
                seen = set()
                for doc in docs:
                    content = doc.page_content.strip()
                    if content and len(content) > 40 and content not in seen:
                        formatted.append(content)
                        seen.add(content)
                return "\n\n".join(formatted)

        retrieve_format = RetrieveAndFormat(retriever, self)

        def process_question(question: str) -> dict:
            context = retrieve_format.invoke(question)
            return {"context": context, "question": question}

        return (
            RunnablePassthrough()
            | process_question
            | self.prompt
            | self.llm
            | self.answer_parser
        )
