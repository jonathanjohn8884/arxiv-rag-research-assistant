import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import settings
# test comment


def load_pdf(pdf_path: str) -> List[Document]:
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def build_faiss_index(chunks: List[Document], save_path: str = None) -> FAISS:
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )
    vector_store = FAISS.from_documents(chunks, embeddings)

    path = save_path or settings.vector_store_path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(path)

    return vector_store


def ingest_pdf(pdf_path: str, save_path: str = None) -> FAISS:
    """Load a PDF, chunk it, embed it, and save a FAISS index. Returns the store."""
    docs = load_pdf(pdf_path)
    chunks = chunk_documents(docs)
    return build_faiss_index(chunks, save_path)


def load_faiss_index(path: str = None) -> FAISS:
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )
    index_path = path or settings.vector_store_path
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
