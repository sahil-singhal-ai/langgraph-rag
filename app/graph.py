from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    """
    Defines the structure of state passed between agents.

    Python Fundamentals:
    - TypedDict specifies expected keys and their types
    - Acts as a contract for agent communication
    """
    question: str
    retriever: Any
    retrieved_docs: List[Any]
    answer: str

graph = StateGraph(AgentState)

graph.add_node("retriever", retriever_agent)
graph.add_node("answer", answer_agent)

graph.set_entry_point("retriever")
graph.add_edge("retriever", "answer")
graph.add_edge("answer", END)

rag_app = graph.compile()
