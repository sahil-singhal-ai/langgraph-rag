_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(...)
    return _embeddings

def create_vector_store(chunks):
    return FAISS.from_documents(chunks, get_embeddings())
