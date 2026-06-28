from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import BASE_DIR, CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_K
from app.infrastructure.llm import get_embeddings

FAISS_INDEX_DIR = BASE_DIR / "faiss_index"


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


def create_retriever():
    embeddings = get_embeddings()

    if FAISS_INDEX_DIR.exists():
        vector_store = FAISS.load_local(
            str(FAISS_INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )
    else:
        docs = load_documents()
        chunks = split_documents(docs)
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(str(FAISS_INDEX_DIR))

    return vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})
