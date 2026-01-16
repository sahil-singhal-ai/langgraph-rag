import requests
import gc
import torch
import time

def run_system(pdf_url, question, progress=None):

    t0 = time.time()

    if not pdf_url or not question:
        return "Error: Please provide both PDF URL and a question."

    if not pdf_url.startswith("http"):
        return "Error: Please provide a valid URL."

    try:
        # Download PDF
        print(f"📥 Downloading PDF: {pdf_url}")
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()

        file_path = "/content/uploaded.pdf"
        with open(file_path, "wb") as f:
            f.write(response.content)

        print("✓ PDF downloaded", time.time() - t0)

        # Vectorstore timing
        t1 = time.time()

        # Get vectorstore (cached)
        vectorstore = get_vectorstore(pdf_url, file_path)
        print("⏱ Vectorstore time:", time.time() - t1)

        # Create retriever
        t2 = time.time()
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
        print("⏱ Retriever setup:", time.time() - t2)

        # Run LangGraph app
        t3 = time.time()
        result = rag_app.invoke({
            "question": question,
            "retriever": retriever
        })
        print("⏱ LLM time:", time.time() - t3)

        print("TOTAL ⏱", time.time() - t0)
        print("Final state keys:", result.keys())
        return result["answer"]

    except Exception as e:
        return f"Error: {str(e)}"
