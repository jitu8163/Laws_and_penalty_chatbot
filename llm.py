import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant"  

class GroqLLM:
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in .env file")
        self.client = Groq(api_key=GROQ_API_KEY)

    def generate(self, prompt: str):
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=1.0,
            max_tokens=500
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    llm = GroqLLM()
    test_prompt = "Say hello in one sentence."
    print(llm.generate(test_prompt))
