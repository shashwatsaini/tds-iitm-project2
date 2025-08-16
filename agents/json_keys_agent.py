import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI


def get_json_keys(instructions_text: str):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        api_key=os.environ.get("GOOGLE_API_KEY"),
        max_retries=0
    )

    parser = JsonOutputParser()

    prompt_text = f"""
    You will be given a set of instructions or a CSV description.

    Your task:
    1. Identify all the keys the user expects in the output JSON.
    2. Return only a JSON list of keys, e.g., ["shape", "variance", "chart"].
    3. Do **not** provide any extra keys, commentary, or answers.

    Instructions:
    {instructions_text}

    Output strictly as a JSON list:
    """

    raw_output = llm.invoke(prompt_text)
    
    keys_json = parser.parse(raw_output.content)
    
    return keys_json
