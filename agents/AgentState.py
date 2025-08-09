from typing import TypedDict, Literal, Union, List, Dict, Optional

class AgentState(TypedDict, total=False):
    input: str
    agent_decision: Literal["csv_agent", "generic_agent"]
    output_dir: str
    request_id: str
    response: str
    next: str
    retry_count: int

    encoded_image: Optional[str]

    # Question-splitting related fields
    questions: List[str]
    current_question: str
    current_question_index: int
    answers: Dict[str, str]
