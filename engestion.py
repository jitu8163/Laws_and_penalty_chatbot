from data_loader import load_csv_data
from split_embed import SplitEmbedder
from qdrant import QdrantStore

def run_ingestion():
    print("Loading CSV data...")
    documents, metadatas = load_csv_data()

    print("Splitting and embedding...")
    splitter = SplitEmbedder()
    chunks, embeddings, chunk_meta = splitter.split_and_embed(documents, metadatas)

    print("Storing in Qdrant...")
    store = QdrantStore()
    store.add(chunks, embeddings.tolist(), chunk_meta)

    print(f"Ingestion complete. Total chunks stored: {len(chunks)}")

if __name__ == "__main__":
    run_ingestion()
