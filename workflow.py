import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate

from agents.AgentState import AgentState
from agents.generic_agent import get_generic_agent_node
from agents.question_splitter_agent import get_question_splitter_node

from tools.web_scraper_tool import web_scraper_tool
from tools.duckdb_query_tool import duckdb_tool
from tools.plot_tools import plot_tool, scatterplot_tool, scatterplot_regression_tool
from tools.code_executor import code_executor_tool

from agents.FallbackLLM import FallbackLLM

load_dotenv()

# Base LLM
llm = FallbackLLM()

tools = [
    web_scraper_tool,
    # duckdb_tool,
    # plot_tool,
    # scatterplot_tool,
    # catterplot_regression_tool,
    code_executor_tool
]
tool_names = [tool.name for tool in tools]

REACT_PROMPT = PromptTemplate.from_template("""
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
                                            
    Action: the action to take, should be one of [{tool_names}]. I must direct all tools to save their outputs to the folder {output_dir}/<file name>
    {csv_text}""" + f"""

    If a URL is provided, I must always use '{tool_names[0]}' to load/process the data.

    I must use '{tool_names[1]}' for:
      - All CSV or Parquet file analysis (local or online)
      - All DuckDB queries or SQL on data
      - Any code-based data processing or transformation """ + """
      - All plotting with matplotlib or seaborn code to save plots to {output_dir}/<file name>.
    
    However, I should not encode the plots, and instead send them to {output_dir}/<file name>.

    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: only return the answer to the question you have been asked. 
    If the output is a file, I must return the file path only. 
    If the output is a number, I must return the number only.
    I must not encode any images or files in base64 — encoding will be done by a later system.

    Example reply:
        [1, "Titanic", 0.485782, "outputs/20250810_123456_abcd1234/plot.png"]

    These are unacceptable replies:
        [1, "Titanic", 0.485782, "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."]
        ["Answer is 1", "Answer is Titanic", "Correlation is 0.485782", "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."]
        [1, "Titanic", 0.485782, "The image is saved at outputs/20250810_123456_abcd1234/plot.png"]

    Begin!

    Question: {input}

    You will be given a specific question from above to answer, and you must ignore all others.
    
    {agent_scratchpad}
""")

generic_agent_node = get_generic_agent_node(llm, REACT_PROMPT, tools)
splitter_node = get_question_splitter_node(llm)

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

def create_workflow():

    workflow = StateGraph(AgentState)
    workflow.add_node("question_splitter", splitter_node)
    workflow.add_node("per_question_router", per_question_router)
    workflow.add_node("generic_agent", generic_agent_node)
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
    workflow.add_edge("accumulator", "per_question_router")

    graph = workflow.compile()

    return graph
