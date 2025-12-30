from src.evaluation.gemini_evaluator import GeminiRAGEvaluator
from src.config import settings

# Test basic import
print("✓ Import thành công")

# Test khởi tạo (cần GEMINI_API_KEY trong .env)
try:
    evaluator = GeminiRAGEvaluator(
        api_key=settings.gemini_api_key,
        model_name="gemini-2.5-flash"
    )
    print("✓ Khởi tạo evaluator thành công")
    
    # Test call đơn giản
    result = evaluator._call_gemini("Xin chào, trả lời số 5")
    print(f"✓ Test call API: {result}")
    
except Exception as e:
    print(f"✗ Lỗi: {type(e).__name__}: {e}")
