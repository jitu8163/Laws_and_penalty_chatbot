from sentence_transformers import SentenceTransformer
from qdrant import get_qdrant_client, COLLECTION_NAME

model = SentenceTransformer('all-MiniLM-L6-v2')

def decompose_question(question: str):
    q = question.lower().strip()

    if " and " in q:
        left, right = q.split(" and ", 1)

        # Detect the shared intent prefix
        if "punishment for" in left:
            prefix = "punishment for"
            left_entity = left.replace(prefix, "").strip()
            right_entity = right.strip().rstrip("?")

            return [
                f"{prefix} {left_entity}",
                f"{prefix} {right_entity}"
            ]

        # fallback
        return [left.strip(), right.strip()]

    return [question]




def retrieve_context(query: str, k: int = 10):
    client = get_qdrant_client()
    vector = model.encode(query).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=k,
        with_payload=True
    )

    contexts = []

    for point in results.points:
        payload = point.payload or {}
        text = payload.get("text", "")
        contexts.append(text)

    return contexts
