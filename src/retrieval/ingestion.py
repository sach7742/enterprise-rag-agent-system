import os
import pickle
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi


def run_ingestion(data_dir: str = "data"):
    print("Starting document ingestion...")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Load both .txt and .pdf files
    documents = []
    txt_loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader)
    pdf_loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)

    documents.extend(txt_loader.load())
    documents.extend(pdf_loader.load())

    if not documents:
        print("No documents found in 'data/'.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    chunk_texts = [doc.page_content for doc in chunks]

    # 1. Disk-persistent Qdrant Vector Store
    qdrant_path = "qdrant_db"
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    qdrant_client = QdrantClient(path=qdrant_path)

    collection_name = "enterprise_docs"
    vector_size = 384  # MiniLM-L6-v2 dimension

    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    embeddings = embedding_model.embed_documents(chunk_texts)
    points = [
        PointStruct(
            id=i,
            vector=embeddings[i],
            payload={"text": chunk_texts[i], "metadata": chunks[i].metadata},
        )
        for i in range(len(chunks))
    ]
    qdrant_client.upsert(collection_name=collection_name, points=points)
    print(f"Indexed {len(points)} chunks into Qdrant at path '{qdrant_path}'.")

    # 2. Sparse BM25 Index
    tokenized_corpus = [doc.lower().split() for doc in chunk_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    with open("bm25_index.pkl", "wb") as f:
        pickle.dump({"index": bm25, "chunks": chunk_texts}, f)
    print("Saved BM25 index to 'bm25_index.pkl'.")


if __name__ == "__main__":
    run_ingestion()