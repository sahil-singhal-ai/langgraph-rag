from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def create_vector_store(chunks):
    """
    Creates FAISS vector database from document chunks.

    Purpose: Enables semantic search - find relevant chunks for questions
    """
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore
