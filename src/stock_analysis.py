# stock_analysis.py
import os
from openai import OpenAI
from src.constants import *

# Make sure OPENAI_API_KEY is set in your environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

client = OpenAI(api_key=api_key)

def get_completion(
    prompt: str = PROMPT_ANALYSIS,
    model: str = "gpt-5",
) -> str:
    """
    Call OpenAI Responses API with instructions + prompt.
    """
    #print(prompt)
    response = client.responses.create(
        model=model,
        input=prompt,
    )
    return response.output_text
