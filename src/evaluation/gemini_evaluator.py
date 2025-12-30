import time
from typing import Optional

import google.genai as genai
import pandas as pd


class GeminiRAGEvaluator:
    """
    Evaluate RAG outputs using Gemini-based scoring prompts.

    Metrics:
    - Faithfulness
    - Answer Relevancy
    - Context Precision
    - Context Recall (requires ground truth)
    """

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def _call_gemini(self, prompt: str) -> str:
        """Gọi Gemini API với retry"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                print(f"⚠️ Lỗi Gemini API (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Đợi trước khi retry
                else:
                    return f"Error: {str(e)}"


    def evaluate_faithfulness(self, answer: str, contexts: list) -> float:
        context_text = "\n\n".join(contexts)
        prompt = f"""Đánh giá Faithfulness: Mọi thông tin trong câu trả lời có xuất phát từ context không?

CONTEXT:
{context_text}

CÂU TRẢ LỜI:
{answer}

Hướng dẫn đánh giá:
- Kiểm tra từng câu/thông tin trong câu trả lời
- Xác định xem thông tin đó có trong context không
- Tính tỷ lệ: (số thông tin có trong context) / (tổng số thông tin)

Cho điểm từ 0-10:
- 10: Mọi thông tin đều có trong context (100% trung thực)
- 8-9: >80% thông tin có trong context
- 6-7: 60-80% thông tin có trong context
- 4-5: 40-60% thông tin có trong context
- 2-3: 20-40% thông tin có trong context
- 0-1: <20% thông tin có trong context (nhiều hallucination)

CHỈ TRẢ LỜI MỘT SỐ TỪ 0-10, KHÔNG GIẢI THÍCH."""

        response = self._call_gemini(prompt)
        try:
            score = float(response.strip())
            return min(max(score, 0), 10) / 10
        except ValueError:
            return 0.5

    def evaluate_answer_relevancy(self, question: str, answer: str) -> float:
        prompt = f"""Đánh giá Answer Relevancy: Câu trả lời có liên quan và trả lời đúng câu hỏi không?

CÂU HỎI:
{question}

CÂU TRẢ LỜI:
{answer}

Hướng dẫn đánh giá:
- Câu trả lời có address được câu hỏi không?
- Có thông tin không liên quan/dư thừa không?
- Câu trả lời có đầy đủ không?

Cho điểm từ 0-10:
- 10: Trả lời hoàn hảo, đúng trọng tâm, đầy đủ
- 8-9: Trả lời tốt, có thể thiếu chi tiết nhỏ
- 6-7: Trả lời được nhưng thiếu thông tin quan trọng
- 4-5: Trả lời một phần, nhiều thông tin thiếu
- 2-3: Ít liên quan, trả lời sai hướng
- 0-1: Không liên quan hoặc sai hoàn toàn

CHỈ TRẢ LỜI MỘT SỐ TỪ 0-10, KHÔNG GIẢI THÍCH."""

        response = self._call_gemini(prompt)
        try:
            score = float(response.strip())
            return min(max(score, 0), 10) / 10
        except ValueError:
            return 0.5

    def evaluate_context_precision(self, question: str, contexts: list) -> float:
        context_text = "\n\n---\n\n".join(
            [f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)]
        )
        prompt = f"""Đánh giá Context Precision: Các context được retrieve có liên quan với câu hỏi không?

CÂU HỎI:
{question}

CONTEXTS:
{context_text}

Hướng dẫn đánh giá:
- Đánh giá từng context: có liên quan với câu hỏi không?
- Tính tỷ lệ: (số context liên quan) / (tổng số context)
- Context precision cao = ít noise, context đều relevant

Cho điểm từ 0-10:
- 10: Tất cả context đều rất liên quan và hữu ích
- 8-9: Hầu hết context liên quan (>80%)
- 6-7: Khoảng 60-80% context liên quan
- 4-5: Chỉ 40-60% context liên quan
- 2-3: Chỉ 20-40% context liên quan
- 0-1: Hầu hết context không liên quan (<20%)

CHỈ TRẢ LỜI MỘT SỐ TỪ 0-10, KHÔNG GIẢI THÍCH."""

        response = self._call_gemini(prompt)
        try:
            score = float(response.strip())
            return min(max(score, 0), 10) / 10
        except ValueError:
            return 0.5

    def evaluate_context_recall(
        self, question: str, contexts: list, ground_truth: Optional[str]
    ) -> Optional[float]:
        if not ground_truth:
            return None

        context_text = "\n\n".join(contexts)
        prompt = f"""Đánh giá Context Recall: Context có đủ thông tin để tạo ra câu trả lời đúng không?

CÂU HỎI:
{question}

CONTEXTS:
{context_text}

CÂU TRẢ LỜI ĐÚNG (GROUND TRUTH):
{ground_truth}

Hướng dẫn đánh giá:
- Xác định các thông tin cần thiết trong ground truth
- Kiểm tra xem context có chứa các thông tin đó không
- Tính tỷ lệ: (thông tin có trong context) / (tổng thông tin cần thiết)

Cho điểm từ 0-10:
- 10: Context chứa 100% thông tin cần thiết
- 8-9: Context chứa >80% thông tin cần thiết
- 6-7: Context chứa 60-80% thông tin cần thiết
- 4-5: Context chứa 40-60% thông tin cần thiết
- 2-3: Context chứa 20-40% thông tin cần thiết
- 0-1: Context chứa <20% thông tin cần thiết

CHỈ TRẢ LỜI MỘT SỐ TỪ 0-10, KHÔNG GIẢI THÍCH."""

        response = self._call_gemini(prompt)
        try:
            score = float(response.strip())
            return min(max(score, 0), 10) / 10
        except ValueError:
            return 0.5

    def evaluate_single_qa(
        self, question: str, answer: str, contexts: list, ground_truth: Optional[str] = None
    ) -> dict:
        results = {}

        results["faithfulness"] = self.evaluate_faithfulness(answer, contexts)
        time.sleep(0.5)

        results["answer_relevancy"] = self.evaluate_answer_relevancy(question, answer)
        time.sleep(0.5)

        results["context_precision"] = self.evaluate_context_precision(question, contexts)
        time.sleep(0.5)

        if ground_truth:
            results["context_recall"] = self.evaluate_context_recall(
                question, contexts, ground_truth
            )
        else:
            results["context_recall"] = None

        return results

    def evaluate_batch(self, test_cases: list) -> pd.DataFrame:
        results = []

        for case in test_cases:
            result = self.evaluate_single_qa(
                question=case["question"],
                answer=case["answer"],
                contexts=case["contexts"],
                ground_truth=case.get("ground_truth"),
            )
            result["question"] = case["question"][:60] + "..."
            results.append(result)
            time.sleep(1)

        return pd.DataFrame(results)
