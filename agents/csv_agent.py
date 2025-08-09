from langchain_experimental.agents.agent_toolkits import create_csv_agent
from typing import TypedDict, Literal

from agents.AgentState import AgentState

def get_csv_agent_node(llm, file_path):
    agent = create_csv_agent(llm, file_path, verbose=True, allow_dangerous_code=True)

    def call_csv_agent(state: AgentState) -> AgentState:
        print(f"""
          \n-------------------------------------------------
          \n[Agent Log] CSV Agent Invoked
          \n-------------------------------------------------\n""")
        
        full_input = f"{state['input']}\nIntermediate output from the previous agent: {state.get('response', '')}"
        
        result = agent.invoke(full_input)

        return {
            "input": state["input"],
            "agent_decision": "csv_agent",
            "response": result
        }

    return call_csv_agent
