import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate

from agents.AgentState import AgentState
from agents.csv_agent import get_csv_agent_node
from agents.generic_agent import get_generic_agent_node
from agents.question_splitter_agent import get_question_splitter_node

from tools.web_scraper_tool import web_scraper_tool
from tools.duckdb_query_tool import duckdb_tool
from tools.plot_tools import plot_tool, scatterplot_tool, scatterplot_regression_tool
from tools.code_executor import code_executor_tool

file_path = 'X_train.csv'
image_path = 'poster.jpeg'

# Base LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ.get('GOOGLE_API_KEY')
)

tools = [web_scraper_tool, duckdb_tool, plot_tool, scatterplot_tool, scatterplot_regression_tool, code_executor_tool]
tool_names = [tool.name for tool in tools]

REACT_PROMPT = PromptTemplate.from_template("""
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
                                            
    Action: the action to take, should be one of [{tool_names}]."""+ f""" If a URL is provided, I must always use '{tool_names[0]}'. If a DuckDB or paraquet is provided online, 
    I must always use '{tool_names[1]}' to connect and execute queries. If plotting is required, I must always use '{tool_names[2]}, and for scatter plot I must use '{tool_names[3]}.'
    For scatter plots with regression lines and analysis, I must always use '{tool_names[4]}. For advanced plotting, I must always use '{tool_names[5]}', which can execute any plotting code I send it.""" + """

    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: only return the raw answer to the original question, do not output any English

    Begin!

    Question: {input}
    
    {agent_scratchpad}
""")

# Router functions
def per_question_router(state: AgentState) -> AgentState:
    questions = state["questions"]
    index = state.get("current_question_index", 0)

    if index >= len(questions):
        return {"next": "end"}

    state["current_question"] = questions[index]
    state['next'] = 'router'
    return state

def accumulator(state: AgentState) -> AgentState:
    idx = state["current_question_index"]
    question = state["questions"][idx]
    answer = state["response"]

    state["answers"][question] = answer

    state["current_question_index"] += 1
    return state

def route_agent(state: AgentState) -> AgentState:
    input_text = state["input"].lower()
    if "csv" in input_text:
        return {"next": "csv_agent"}
    return {"next": "generic_agent"}

def router(state: AgentState) -> str:
    return route_agent(state)

def should_retry(state):
    output = state.get("response", "")
    retry_count = state.get("retry_count", 0)

    if retry_count >= 3:
        return "end"
    
    if "Traceback" and ("Error" in output or "Exception" in output):
        return "retry"

    return "end"

def retry_generic_agent(state):
    state["retry_count"] = state.get("retry_count", 0) + 1
    return generic_agent_node.invoke(state)

generic_agent_node = get_generic_agent_node(llm, REACT_PROMPT, tools)
csv_agent_node = get_csv_agent_node(llm, file_path)
splitter_node = get_question_splitter_node(llm)

# Build the graph
workflow = StateGraph(AgentState)

workflow.add_node("question_splitter", splitter_node)
workflow.add_node("per_question_router", per_question_router)
workflow.add_node("generic_agent", generic_agent_node)
workflow.add_node("csv_agent", csv_agent_node)
workflow.add_node("router", router)
workflow.add_node("retry_generic_agent", retry_generic_agent)
workflow.add_node("accumulator", accumulator)

workflow.set_entry_point("question_splitter")

workflow.add_conditional_edges(
    "question_splitter",
    lambda state: "per_question_router"
)

workflow.add_conditional_edges(
    "per_question_router",
    lambda state: state["next"],
    {
        "router": "router",
        "end": END
    }
)

workflow.add_conditional_edges(
    "router",
    lambda state: state["next"],
    {
        "csv_agent": "csv_agent",
        "generic_agent": "generic_agent"
    }
)

workflow.add_conditional_edges(
    "generic_agent",
    should_retry,
    {
        "retry": "retry_generic_agent",
        "end": "accumulator"
    }
)

workflow.add_edge("retry_generic_agent", "generic_agent")
workflow.add_edge("csv_agent", "accumulator")
workflow.add_edge("accumulator", "per_question_router")

graph = workflow.compile()

response = graph.invoke({"input": """
Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings containing the answer.

1. Top 5 highest grossing films of all time.
"""})

print(response['answers'])