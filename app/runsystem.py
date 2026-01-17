#add system entrypoint with singleton pipeline initialization

import requests
from app.pipeline import RAGPipeline

_pipeline = None


def run_system(pdf_url: str, question: str):
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()

    response = requests.get(pdf_url, timeout=30)
    response.raise_for_status()

    file_path = "/content/uploaded.pdf"
    with open(file_path, "wb") as f:
        f.write(response.content)

    return _pipeline.run(pdf_url, question, file_path)
