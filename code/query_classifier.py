import os

import openai
from dotenv import load_dotenv

load_dotenv()


def query_gpt(prompt: list[dict], model: str = "gpt-3.5-turbo") -> str:
    """
    Prompts the classifier (GPT-3.5 or GPT-4)
    :param prompt: Prompt used to query GPT
    :param model: GPT model to use
    :return: GPT's answer
    """

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    completion = client.chat.completions.create(
        model=model,
        messages=prompt
    )
    return completion.choices[0].message.content.lower().strip()
