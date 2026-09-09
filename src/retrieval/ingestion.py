import os
import glob
import pickle
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from langchain_huggingface import HuggingFaceEmbeddings

QDRANT_PATH = "qdrant_db"
COLLECTION_NAME = "enterprise_docs"
BM25_PATH = "bm25_index.pkl"

def run_ingestion():
    # 1. Load documents
    texts = []
    for file_path in glob.glob("data/*.txt") + glob.glob("data/*.pdf"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())

    if not texts:
        print("No documents found in data/")
        return

    # Split documents into chunks
    chunks = [chunk for doc in texts for chunk in doc.split("\n\n") if chunk.strip()]

    # 2. Dense Embeddings & Qdrant Storage
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectors = embeddings_model.embed_documents(chunks)

    client = QdrantClient(path=QDRANT_PATH)
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE)
        )

    points = [
        PointStruct(id=idx, vector=vec, payload={"text": chunk})
        for idx, (vec, chunk) in enumerate(zip(vectors, chunks))
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    client.close()

    # 3. BM25 Sparse Indexing
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"index": bm25, "chunks": chunks}, f)

    print(f"Successfully ingested {len(chunks)} chunks into {QDRANT_PATH} & {BM25_PATH}")

if __name__ == "__main__":
    run_ingestion()