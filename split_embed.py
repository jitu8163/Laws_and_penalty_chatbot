from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# Load embedding model (CPU-friendly and strong)
EMBED_MODEL_NAME = "intfloat/e5-small-v2"

class SplitEmbedder:
    def __init__(self, chunk_size=300, chunk_overlap=50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n", ".", ",", " ", "|"]
        )
        self.model = SentenceTransformer(EMBED_MODEL_NAME)

    def split(self, texts, metadatas):
        all_chunks = []
        all_metadata = []

        for text, meta in zip(texts, metadatas):
            chunks = self.splitter.split_text(text)
            for c in chunks:
                all_chunks.append(c)
                all_metadata.append(meta)

        return all_chunks, all_metadata

    def embed(self, texts):
        return self.model.encode(texts, show_progress_bar=True)

    def split_and_embed(self, texts, metadatas):
        chunks, chunk_meta = self.split(texts, metadatas)
        embeddings = self.embed(chunks)
        return chunks, embeddings, chunk_meta


if __name__ == "__main__":
    from data_loader import load_csv_data

    docs, meta = load_csv_data()
    se = SplitEmbedder()
    chunks, vectors, metas = se.split_and_embed(docs, meta)

    print("Total Chunks:", len(chunks))
    print("Vector shape:", len(vectors), len(vectors[0]))
    print("Sample Chunk:", chunks[0])
    print("Sample Metadata:", metas[0])
