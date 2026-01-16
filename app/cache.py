VECTOR_CACHE = {}

def get_vectorstore(pdf_url, file_path):
    if pdf_url in VECTOR_CACHE:
        print("⚡ Using cached vectorstore")
        return VECTOR_CACHE[pdf_url]

    chunks = load_and_split_pdf(file_path)
    #print("FIRST CHUNK:\n", chunks[0].page_content[:2000])
    vectorstore = create_vector_store(chunks)

    VECTOR_CACHE[pdf_url] = vectorstore
    return vectorstore
