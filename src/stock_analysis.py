# stock_analysis.py
import os
from openai import OpenAI

# Make sure OPENAI_API_KEY is set in your environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

client = OpenAI(api_key=api_key)

# Templates
INSTRUCTION_TEMPLATE = "You are an expert financial analyst. Answer clearly and concisely."
PROMPT_TEMPLATE = "Analyze the stock performance of NVDA compared to AMD over the last quarter."

def get_completion(
    instructions: str = INSTRUCTION_TEMPLATE,
    prompt: str = PROMPT_TEMPLATE,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Call OpenAI Responses API with instructions + prompt.
    """
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=prompt,
    )
    return response.output_text

if __name__ == "__main__":
    result = get_completion()
    print(result)
