from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from agents.AgentState import AgentState

question_splitter_prompt = PromptTemplate.from_template("""
You will be given a block of text. It may contain one or more questions.
Extract and list **only** the questions, one per line. 
Do not answer them or include any other commentary.
Ignore any URLs or markdown, and only return the clean questions.

Text:
{input}

Output (as a JSON list of strings):
```json
["Question 1?", "Question 2?", "Question 3?"]
```
""")

def get_question_splitter_node(llm) -> Runnable:
    splitter_chain = question_splitter_prompt | llm | JsonOutputParser()

    def node(state: AgentState) -> AgentState:
        print(f"""
          \n-------------------------------------------------
          \n[Agent Log] Question Splitter Invoked
          \n-------------------------------------------------\n""")
        
        questions = splitter_chain.invoke({"input": state["input"]})
        return {
            **state,
            "questions": questions,
            "current_question_index": 0,
            "answers": {},
            "next": "per_question_router"
        }

    return node
