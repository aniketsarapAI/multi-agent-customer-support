from pathlib import Path
from typing import Any, List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

from app.config import BASE_DIR, CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_K, settings
from app.infrastructure.llm import get_embeddings


def load_documents() -> list:
    pdf_files = sorted(Path(BASE_DIR / "documents").glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {BASE_DIR / 'documents'}. "
            "Place Company_Policies.pdf, Company_Profile.pdf, and Product_and_Pricing.pdf there."
        )
    docs = []
    for path in pdf_files:
        docs.extend(PyPDFLoader(str(path)).load())
    return docs


def split_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)


EMBEDDING_DIM = 768


class PineconeRetriever:
    """Lightweight retriever backed by Pinecone.

    Quacks like a LangChain BaseRetriever — supports ``.invoke(query)``
    which is the only interface the graph code uses.
    """

    def __init__(self, index: Any, embeddings: Any, k: int = 4):
        self._index = index
        self._embeddings = embeddings
        self.k = k

    def invoke(self, query: str) -> List[Document]:
        query_vector = self._embeddings.embed_query(query)
        results = self._index.query(
            vector=query_vector,
            top_k=self.k,
            include_metadata=True,
        )
        docs = []
        for match in results.matches:
            metadata = dict(match.metadata) if match.metadata else {}
            text = metadata.pop("text", "")
            docs.append(Document(page_content=text, metadata=metadata))
        return docs


def create_retriever():
    embeddings = get_embeddings()

    pc = Pinecone(api_key=settings.pinecone_api_key)

    if settings.pinecone_index_name not in pc.list_indexes().names():
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(settings.pinecone_index_name)

    stats = index.describe_index_stats()
    if stats.total_vector_count == 0:
        docs = load_documents()
        chunks = split_documents(docs)

        vectors = []
        for i, chunk in enumerate(chunks):
            vector = embeddings.embed_query(chunk.page_content)
            vectors.append(
                {
                    "id": f"doc-{i}",
                    "values": vector,
                    "metadata": {
                        "text": chunk.page_content,
                        "source": str(chunk.metadata.get("source", "")),
                        "page": chunk.metadata.get("page", 0),
                    },
                }
            )

        for i in range(0, len(vectors), 100):
            index.upsert(vectors=vectors[i : i + 100])

    return PineconeRetriever(index=index, embeddings=embeddings, k=RETRIEVER_K)
