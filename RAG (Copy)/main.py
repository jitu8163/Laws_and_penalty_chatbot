from fastapi import FastAPI, Query, HTTPException
from retrieval import ranked_docs
from llm import generate_answer

app = FastAPI(
    title="Indian Law RAG Chatbot",
    description="Legal assistant",
    version="1.0.0"
)

@app.post("/chat")
def chat(query: str = Query(...)):
    reranked_docs = ranked_docs(query)

    answer = generate_answer(
        query=query,
        reranked_docs=reranked_docs
    )

    return {"answer": answer}


@app.get("/health")
def health():
    return {"status": "ok"}
