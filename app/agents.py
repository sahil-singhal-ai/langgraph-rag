def retriever_agent(state):
    """
    Retrieves relevant chunks from vectorstore.
    No LLM used here → fast.
    """
    question = state["question"]
    retriever = state["retriever"]

    docs = retriever.invoke(question)

    return {
        **state,
        "retrieved_docs": docs
    }

#answer agent generates final answer using llm
def answer_agent(state,llm):

    question = state["question"]
    docs = state["retrieved_docs"]

    context = "\n\n".join(doc.page_content[:1200] for doc in docs)

    prompt = f"""

You are a precise document question-answering system.
Answer ONLY from the provided context.
If the answer is not in the context, say: "Not found in document."

Context:
{context}

Question: {question}

Give only the answer. Do not repeat the question. Do not add extra text.

"""

    response = llm.invoke(prompt)

    return {
        **state,
        "answer": response
    }
