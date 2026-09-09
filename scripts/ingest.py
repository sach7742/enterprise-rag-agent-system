import os
import glob
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import settings


def load_documents(data_dir: str = "data") -> List[Document]:
    documents = []
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        return documents

    for file_path in glob.glob(f"{data_dir}/*.txt"):
        print(f"Loading TXT: {file_path}")
        loader = TextLoader(file_path, encoding="utf-8")
        documents.extend(loader.load())

    for file_path in glob.glob(f"{data_dir}/*.pdf"):
        print(f"Loading PDF: {file_path}")
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())

    print(f"Total raw documents loaded: {len(documents)}")
    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks


def ingest_to_qdrant(chunks: List[Document]):
    if not chunks:
        print("No document chunks to ingest.")
        return

    print(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)

    qdrant_url = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"
    collection_name = settings.QDRANT_COLLECTION_NAME

    print(f"Ingesting {len(chunks)} chunks into Qdrant at {qdrant_url} | Collection: {collection_name}...")

    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=qdrant_url,
        collection_name=collection_name,
        force_recreate=True
    )
    print("Successfully ingested documents into Qdrant!")


if __name__ == "__main__":
    docs = load_documents("data")
    if docs:
        chunked = chunk_documents(docs)
        ingest_to_qdrant(chunked)
    else:
        print("No documents found in data/ directory.")
