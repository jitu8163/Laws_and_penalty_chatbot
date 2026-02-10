import os
import uuid
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("QDRANT_URL or QDRANT_API_KEY missing")


DATASET_DIR = "dataset"
COLLECTION_NAME = "indian_laws"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


embedder = SentenceTransformer(EMBEDDING_MODEL)

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)


# Create collection if not exists
existing = [c.name for c in client.get_collections().collections]
if COLLECTION_NAME not in existing:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )

def embed(text: str):
    return embedder.encode(text, normalize_embeddings=True).tolist()

UPSERT_BATCH_SIZE = 64

def upsert_in_batches(points):
    for i in range(0, len(points), UPSERT_BATCH_SIZE):
        batch = points[i:i + UPSERT_BATCH_SIZE]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch
        )


def ingest_qa_csv(filename: str, law: str):
    df = pd.read_csv(os.path.join(DATASET_DIR, filename)) 
    points = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Ingesting {filename}"):
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()

        text = f"Question: {question}\nAnswer: {answer}"

        metadata = {
            "law": law,
            "country": "india",
            "source_type": "qa",
            "authority_level": "secondary",
            "dataset": filename
        }

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embed(text),
                payload={"text": text, **metadata}
            )
        )

        if len(points) >= UPSERT_BATCH_SIZE:
            upsert_in_batches(points)
            points.clear()

    # Flush remaining
    upsert_in_batches(points)


def ingest_instruction_csv(filename: str, primary: bool):
  
    df = pd.read_csv(os.path.join(DATASET_DIR, filename))
    points = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Ingesting {filename}"):
        instruction = str(row["instruction"]).strip()
        input_col = str(row["input"]).strip()
        output = str(row["output"]).strip()

        text = (
            f"{instruction} {input_col}.\n"
            f"{output}"
        )

        metadata = {
            "law": "ipc",
            "country": "india",
            "source_type": "statute",
            "authority_level": "primary" if primary else "secondary",
            "dataset": filename
        }

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embed(text),
                payload={
                    "text": text,
                    **metadata
                }
            )
        )

    upsert_in_batches(points)

if __name__ == "__main__":

    # QA-style datasets
    ingest_qa_csv("ipc_qa.csv", "ipc")
    ingest_qa_csv("crpc_qa.csv", "crpc")
    ingest_qa_csv("csvjson.csv", "ipc")

    # Instruction-style datasets
    ingest_instruction_csv(
        "Laws and Constitution of India.csv",
        primary=False
    )

    ingest_instruction_csv(
        "Laws and Constitution of India_Cleaned.csv",
        primary=True
    )

    print("✅ Ingestion completed for ALL 5 CSV files with metadata.")
