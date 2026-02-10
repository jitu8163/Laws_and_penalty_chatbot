import os
from dotenv import load_dotenv
from typing import List, Dict
from prompt import SYSTEM_PROMPT
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY missing in .env")

client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "llama-3.1-8b-instant"

def generate_answer(query: str, reranked_docs: List[Dict]) -> str:

    if not reranked_docs:
        return "No relevant legal documents were found."

    # Build context
    context_blocks = []
    for doc in reranked_docs:
        block = f""" [Law: {doc.get('law')} | Authority: {doc.get('authority_level')}] {doc.get('text')}"""
        context_blocks.append(block.strip())

    context = "\n\n".join(context_blocks)

    user_prompt = f"""
        Context:
        {context}

        Question:
        {query}
        """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )

    message = response.choices[0].message.content

    if message is None or not isinstance(message, str):
        return "The answer is not available in the provided legal documents."

    return message.strip()

