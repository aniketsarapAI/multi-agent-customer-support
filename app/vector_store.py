from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import DOCUMENTS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_K
from app.llm import get_embeddings


def load_documents() -> list:
    pdf_files = sorted(Path(DOCUMENTS_DIR).glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {DOCUMENTS_DIR}. "
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
    docs = load_documents()
    chunks = split_documents(docs)
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})
