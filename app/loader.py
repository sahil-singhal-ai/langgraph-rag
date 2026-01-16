from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pdfplumber
from langchain_core.documents import Document


def load_and_split_pdf(file_path):
    """
    Loads PDF and splits into chunks for embedding.

    Python Fundamentals:
    - Function returns list of Document objects
    - chunk_size=1000: each chunk ~1000 characters
    - chunk_overlap=200: overlap prevents breaking context
    """
    documents=[]
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                documents.append(Document(page_content=text, metadata={"page": i}))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # Reduce from 1000 to shorten context window
        chunk_overlap=50 # Reduce from 200 to shorten context window
    )
    chunks = splitter.split_documents(documents)
    return chunks
