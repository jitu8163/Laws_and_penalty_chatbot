import os
from dotenv import load_dotenv
from typing import List, Dict
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("Missing Qdrant credentials in .env")

COLLECTION_NAME = "indian_laws"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

embedder = SentenceTransformer(EMBEDDING_MODEL)
reranker = CrossEncoder(RERANK_MODEL)

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60
)

# -------------------------
# RETRIEVAL
# -------------------------
def retrieve_documents(query: str, top_k: int = 20) -> List[Dict]:

    query_vector = embedder.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True
    )

    docs = []
    for r in response.points or []:
        payload = r.payload or {}
        text = payload.get("text", "").strip()
        if not text:
            continue

        docs.append({
            "text": text,
            "law": payload.get("law"),
            "source_type": payload.get("source_type"),
            "authority_level": payload.get("authority_level"),
            "dataset": payload.get("dataset")
        })
    return docs


def rerank_documents(query: str, retrieved_docs: List[Dict], top_k: int = 5) -> List[Dict]:

    if not retrieved_docs:
        return []

    # safety: limit reranker load
    retrieved_docs = retrieved_docs[:30]

    pairs = [(query, doc["text"][:2000]) for doc in retrieved_docs]

    scores = reranker.predict(pairs)

    for doc, score in zip(retrieved_docs, scores):
        doc["rerank_score"] = float(score)

    return sorted(
        retrieved_docs, key=lambda x: x["rerank_score"], reverse=True )[:top_k]

# -------------------------
# PUBLIC ENTRY
# -------------------------
def ranked_docs( query: str, retrieval_k: int = 20) -> List[Dict]:

    retrieved_docs = retrieve_documents(
        query=query,
        top_k=retrieval_k
    )
    print(retrieved_docs)
    print(len(retrieved_docs))

    if not retrieved_docs:
        return []

    return rerank_documents(
        query=query,
        retrieved_docs=retrieved_docs
    )
