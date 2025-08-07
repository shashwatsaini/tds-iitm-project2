# Define state structure
from typing import TypedDict, Literal, Union

class AgentState(TypedDict):
    input: str
    agent_decision: Literal["csv_agent", "generic_agent"]
    response: str