import os
import pandas as pd

DATASET_DIR = "dataset"

def row_to_text(row: pd.Series) -> str:
    parts = []
    for col, val in row.items():
        parts.append(f"{col}: {val}")
    return " | ".join(parts)

def load_csv_data():
    documents = []
    metadatas = []

    for file in os.listdir(DATASET_DIR):
        if file.endswith(".csv"):
            path = os.path.join(DATASET_DIR, file)
            df = pd.read_csv(path)

            for idx, row in df.iterrows():
                text = row_to_text(row)
                documents.append(text)
                metadatas.append({
                    "source_file": file,
                    "row_id": idx
                })

    return documents, metadatas


if __name__ == "__main__":
    docs, meta = load_csv_data()
    print(f"Loaded {len(docs)} rows from CSV files.")
    print(docs[0])
    print(meta[0])
