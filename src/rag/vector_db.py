from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class VectorDB:
    def __init__(
        self,
        documents=None,
        embedding_model: str = "intfloat/multilingual-e5-base",
        collection_name: str = "vietnamese_docs",
        persist_dir: str = "./storage/chroma",
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding = HuggingFaceEmbeddings(model_name=embedding_model)
        self.db = self._build_db(documents)

    def _build_db(self, documents):
        if documents is None or len(documents) == 0:
            db = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding,
                persist_directory=self.persist_dir,
            )
        else:
            db = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding,
                collection_name=self.collection_name,
                persist_directory=self.persist_dir,
            )
        return db

    def get_retriever(self, search_kwargs: dict | None = None):
        if search_kwargs is None:
            search_kwargs = {"k": 3}
        return self.db.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
        )
