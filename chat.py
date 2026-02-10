from fastapi import FastAPI
from pydantic import BaseModel

from retriever import retrieve_context, decompose_question
from prompt import build_prompt
from llm import GroqLLM

app = FastAPI(title="Laws & Penalties RAG API")

llm = GroqLLM()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    contexts: list[str]


@app.post("/ask", response_model=QueryResponse)
def ask(req: QueryRequest):
    sub_questions = decompose_question(req.question)
    final_answers = []
    all_contexts = []

    for q in sub_questions:
        contexts = retrieve_context(q, k=10)
        all_contexts.extend(contexts)

        prompt = build_prompt(contexts, q)
        ans = llm.generate(prompt)
        final_answers.append(ans)

    return {
        "answer": "\n\n".join(final_answers),
        "contexts": all_contexts
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("chat:app", host="0.0.0.0", port=8000, reload=True)
