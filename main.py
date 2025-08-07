import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from typing import TypedDict, Literal, Union

from agents.AgentState import AgentState
from agents.csv_agent import get_csv_agent_node
from agents.generic_agent import get_generic_agent_node

from tools.web_scraper_tool import web_scraper_tool

file_path = 'X_train.csv'

# Base LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ.get('GOOGLE_API_KEY')
)

tools = [web_scraper_tool]
tool_names = [web_scraper_tool.name]

REACT_PROMPT = PromptTemplate.from_template("""
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]. If a URL is provided, I must always use 'WebScraper'
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: only return the raw answer to the original question, do not output any English

    Begin!

    Question: {input}
    {agent_scratchpad}
""")

# Router function
def route_agent(state: AgentState) -> Literal["csv_agent", "generic_agent"]:
    if state["agent_decision"] == "generic_agent":
        return "__end__"
    
    input_text = state["input"].lower()
    if "csv" in input_text or "column" in input_text or "row" in input_text:
        return "csv_agent"
    return "generic_agent"

generic_agent_node = get_generic_agent_node(llm, REACT_PROMPT, tools)
csv_agent_node = get_csv_agent_node(llm, file_path)

# Build the graph
workflow = StateGraph(AgentState)

workflow.set_entry_point("generic_agent")

workflow.add_node("generic_agent", generic_agent_node)
workflow.add_node("csv_agent", csv_agent_node)
workflow.add_node("router", lambda state: state)

workflow.add_edge("generic_agent", "router")
workflow.add_edge("csv_agent", END)
workflow.add_edge("generic_agent", END)

workflow.add_conditional_edges("router", route_agent, {
    "csv_agent": "csv_agent",
    "generic_agent": "generic_agent",
    "__end__": END
})

graph = workflow.compile()

response = graph.invoke({"input": """
                            Scrape the list of highest grossing films from Wikipedia. It is at the URL:
                            https://en.wikipedia.org/wiki/List_of_highest-grossing_films
                         
                            What's the correlation between the Rank and Peak?
                            """})
print(response["response"])
