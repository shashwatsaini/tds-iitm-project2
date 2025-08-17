import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

def build_json_output(instructions_text: str, results_list: list):
    """
    Given instructions (that define JSON keys) and a results list (from another workflow),
    align results with the keys in order and return a JSON object.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=os.environ.get("GOOGLE_API_KEY"),
        max_retries=0
    )
    parser = JsonOutputParser()

    prompt_text = f"""
    You are given:
    - Instructions that describe which JSON keys the user expects.
    - A list of results (in order).

    Your job:
    1. Extract only the JSON keys from the instructions (in order).
    2. Match each key to the corresponding item in the results list.
       (1st key → 1st result, 2nd key → 2nd result, etc.)
    3. Return a single JSON object with key-value pairs.
    4. Do not add commentary, only return valid JSON.

    Instructions:
    {instructions_text}

    Results:
    {results_list}

    Output strictly as a JSON object:
    """

    raw_output = llm.invoke(prompt_text)
    json_obj = parser.parse(raw_output.content)

    return json_obj