#centralize LLM initialization and inject into graph nodes

from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, END

from app.model import load_llm
from app.agents import retriever_agent, answer_agent
from app.loader import load_and_split_pdf
from app.embeddings import create_vector_store


class AgentState(TypedDict):
    question: str
    retriever: Any
    retrieved_docs: List[Any]
    answer: str


class RAGPipeline:
    def __init__(self):
        self.llm = load_llm()
        self._vector_cache = {}
        self.graph = self._build_graph()

    def _build_graph(self):
        def _retriever(state):
            return retriever_agent(state)

        def _answer(state):
            return answer_agent(state, self.llm)

        graph = StateGraph(AgentState)
        graph.add_node("retriever", _retriever)
        graph.add_node("answer", _answer)
        graph.set_entry_point("retriever")
        graph.add_edge("retriever", "answer")
        graph.add_edge("answer", END)

        return graph.compile()

    def _get_vectorstore(self, pdf_url, file_path):
        if pdf_url in self._vector_cache:
            return self._vector_cache[pdf_url]

        chunks = load_and_split_pdf(file_path)
        vs = create_vector_store(chunks)
        self._vector_cache[pdf_url] = vs
        return vs

    def run(self, pdf_url, question, file_path):
        retriever = self._get_vectorstore(pdf_url, file_path)\
            .as_retriever(search_kwargs={"k": 3})

        result = self.graph.invoke({
            "question": question,
            "retriever": retriever
        })
        return result["answer"]
