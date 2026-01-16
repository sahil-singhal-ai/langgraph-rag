import gradio as gr

interface = gr.Interface(
    fn=run_system,
    inputs=[
        gr.Textbox(
            label="PDF URL",
            placeholder="https://example.com/document.pdf",
            lines=1
        ),
        gr.Textbox(
            label="Your Question",
            placeholder="What is this document about?",
            lines=2
        )
    ],
    outputs=gr.Textbox(
        label="Answer",
        lines=10
    ),
    title="LangGraph RAG System (Fast & Clean)",
    description="Retriever → Answer Agent using LangGraph. Paste a PDF URL and ask a question.",
    examples=[
        [
            "https://thedocs.worldbank.org/en/doc/8bf0b62ec6bcb886d97295ad930059e9-0050012025/original/GEP-June-2025.pdf",
            "What is this report about"
        ],
        [
            "https://static.pib.gov.in/WriteReadData/specificdocs/documents/2025/jun/doc2025616570701.pdf",
            "What are the key economic indicators mentioned?"
        ]
    ],
    allow_flagging="never"
)

interface.launch(share=False, debug=True)
