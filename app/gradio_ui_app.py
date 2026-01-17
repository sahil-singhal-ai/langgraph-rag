
import gradio as gr
from app.runsystem import run_system

interface = gr.Interface(
    fn=run_system,
    inputs=[
        gr.Textbox(label="PDF URL"),
        gr.Textbox(label="Question")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="LangGraph RAG System",
    description="Retriever → Answer Agent using LangGraph"
)

interface.launch(share=false, debug=True)
