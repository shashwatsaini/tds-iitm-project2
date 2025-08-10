from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from agents.AgentState import AgentState

question_splitter_prompt = PromptTemplate.from_template("""
You will be given a block of text. It may contain one or more questions.

Your task:
1. Extract and list **only** the questions, one per line.
2. Remove the parts of questions that mention, but retain the main instruction:
   - Encoding an image
   - Encoding a file
   - Base64 encoding
   - Data URI formats (e.g., "data:image/png;base64,...")
3. Do not answer the questions or include any commentary.
4. Ignore any URLs or markdown.
5. Return the result as a JSON list of strings.

Text:
{input}

Output (as a JSON list of strings):
```json
["Question 1?", "Question 2?", "Question 3?"]
""")

def get_question_splitter_node(llm) -> Runnable:
    splitter_chain = question_splitter_prompt | llm | JsonOutputParser()

    def node(state: AgentState) -> AgentState:
        print(f"""
          \n-------------------------------------------------
          \n[Agent Log] Question Splitter Invoked
          \n-------------------------------------------------\n""")
        
        questions = splitter_chain.invoke({"input": state["input"]})

        print(f"[Question Splitter] Extracted Questions: {questions}")

        return {
            **state,
            "questions": questions,
            "current_question_index": 0,
            "answers": {},
            "next": "per_question_router"
        }

    return node
