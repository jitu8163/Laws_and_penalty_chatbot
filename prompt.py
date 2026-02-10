def build_prompt(context_chunks, question):
    context_text = "\n\n".join(context_chunks)

    prompt = f"""
You are a summarization engine for a legal retrieval system.

Your task is to answer the question based solely on the provided context.
- You'll get 10 context chunks that may contain relevant information.
- You have to carefully read through them and extract the necessary details among from all context_text and return only one or two best relevant sentences.
- Only the required information should be included in your answer never include the context in your answer. Only return what you extracted from the context.
- If you got more than one question, answer them all in the final answer by merging them accordingly Don't give separate.
- Synthesize it into a clear, concise answer.
- Do NOT use any external knowledge.
- Do NOT make assumptions.
- Do NOT generalize beyond the context.
- If the context does not contain the answer, say exactly:
  "Answer not found in the provided database."

Context:
{context_text}

Question:
{question}

Summarized Answer (only from context):
"""
    return prompt.strip()
