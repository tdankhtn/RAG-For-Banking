import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data" / "generative_ai"))
    chroma_dir: Path = Path(os.getenv("CHROMA_DIR", PROJECT_ROOT / "storage" / "chroma"))
    artifacts_dir: Path = Path(os.getenv("ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
    split_docs_path: Path = Path(
        os.getenv("SPLIT_DOCS_PATH", PROJECT_ROOT / "artifacts" / "split_documents.txt")
    )

    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-base"
    )
    llm_model: str = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
    reranker_model: str = os.getenv(
        "RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    llm_max_new_tokens: int = int(os.getenv("LLM_MAX_NEW_TOKENS", "450"))
    llm_top_p: float = float(os.getenv("LLM_TOP_P", "0.75"))

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "400"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    retriever_k: int = int(os.getenv("RETRIEVER_K", "3"))

    ingest_file_types: tuple[str, ...] = tuple(
        ext.strip().lower()
        for ext in os.getenv("INGEST_FILE_TYPES", "pdf,txt,md,docx,xlsx").split(",")
        if ext.strip()
    )
    ingest_tables: bool = os.getenv("INGEST_TABLES", "true").lower() in {
        "1",
        "true",
        "yes",
    }
    table_rows_per_chunk: int = int(os.getenv("TABLE_ROWS_PER_CHUNK", "8"))
    fee_fact_chunking: bool = os.getenv("FEE_FACT_CHUNKING", "true").lower() in {
        "1",
        "true",
        "yes",
    }
    fee_fact_patterns: tuple[str, ...] = tuple(
        ext.strip().lower()
        for ext in os.getenv("FEE_FACT_PATTERNS", "bieu_phi,bieu-phi,fee").split(",")
        if ext.strip()
    )
    fee_fact_include_raw_text: bool = os.getenv(
        "FEE_FACT_INCLUDE_RAW_TEXT", "true"
    ).lower() in {"1", "true", "yes"}

    use_reranking: bool = os.getenv("USE_RERANKING", "true").lower() in {
        "1",
        "true",
        "yes",
    }

    hf_token: str = os.getenv("HF_TOKEN", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")


settings = Settings()

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if settings.hf_token:
    os.environ.setdefault("HF_TOKEN", settings.hf_token)
