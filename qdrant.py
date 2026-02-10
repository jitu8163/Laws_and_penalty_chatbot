import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION")
VECTOR_SIZE = 384  # for e5-small-v2

def get_qdrant_client():
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("QDRANT_URL or QDRANT_API_KEY missing in .env")
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def create_collection_if_not_exists():
    client = get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME not in collections:
        print(f"Creating Qdrant collection: {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")

class QdrantStore:
    def __init__(self):
        self.client = get_qdrant_client()
        create_collection_if_not_exists()

    def add(self, texts, embeddings, metadatas, batch_size=200):
        total = len(texts)
        print(f"Uploading {total} vectors to Qdrant in batches of {batch_size}...")

        for i in range(0, total, batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_vectors = embeddings[i:i+batch_size]
            batch_meta = metadatas[i:i+batch_size]

            points = []
            for j, (text, vector, meta) in enumerate(zip(batch_texts, batch_vectors, batch_meta)):
                payload = {"text": text}
                payload.update(meta)

                points.append({
                    "id": i + j,
                    "vector": vector,
                    "payload": payload
                })

            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )

            print(f"Uploaded batch {i//batch_size + 1} / {(total + batch_size - 1)//batch_size}")

