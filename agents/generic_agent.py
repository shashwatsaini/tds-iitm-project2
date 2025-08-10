from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from typing import TypedDict, Literal

from agents.AgentState import AgentState

def get_generic_agent_node(llm, REACT_PROMPT, tools) -> Runnable:
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=create_react_agent(llm, tools=tools, prompt=REACT_PROMPT),
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    def call_generic_agent(state: AgentState) -> AgentState:
        print(f"""
          \n-------------------------------------------------
          \n[Agent Log] Generic Agent Invoked
          \n-------------------------------------------------\n""")
        
        current_question = state.get('current_question')
        result = agent_executor.invoke({
            "input": state["input"] + f'The current question I must answer, and ignore all others: {current_question}',
            "output_dir": state["output_dir"],
            "csv_dir": state['csv_dir'],
            "csv_text": state['csv_text']
        })
        
        return {
            **state,
            "input": state["input"],
            "agent_decision": "generic_agent",
            "response": result["output"]
        }

    return call_generic_agent
