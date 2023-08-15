# Importing necessary libraries
import os, time
from pdf2image import convert_from_path
import gradio as gr
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# Load the model and initialize necessary components
embeddings = OpenAIEmbeddings()
chain = load_qa_chain(OpenAI(), chain_type="stuff")
os.environ["OPENAI_API_KEY"] = "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"

# Method to generate answers
def generate_answer(pdf_file, question):
    try:
        # Loading the uploaded pdf and extracting text out of it
        with open(pdf_file.name, "rb") as pdf_file_1:
            pdf = PdfReader(pdf_file_1)
            image = convert_from_path(pdf_file.name)
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        if text:
            # Creating semantic index on text
            docsearch = FAISS.from_texts([text], embeddings)
            # Calculating Similarity Score and Fetcheing Index for answer
            docs = docsearch.similarity_search(question)
            answer = chain.run(input_documents=docs, question=question)
            # Adding sleep component to avoid Infinite Time Error
            time.sleep(2.5)
            # Returning Pdf Image with Answer to Question
            return image[0], answer
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

# Define the Gradio interface
iface = gr.Interface(
    fn=generate_answer,
    inputs = [
        gr.inputs.File(label = "Upload PDF", type = "file"),
        gr.inputs.Textbox(label = "Question"),
    ],
    outputs = [
        gr.outputs.Image(type = "pil", label = "PDF Image"),
        gr.outputs.Textbox(label = "Answer"),
    ],
    title = "Invoice PDF's Question Answer App",
    description = "Upload a PDF, input a question, and get an answer!",
)

# Run the Gradio app
iface.launch(debug = True)