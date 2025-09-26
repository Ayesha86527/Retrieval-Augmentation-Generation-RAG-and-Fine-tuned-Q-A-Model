# 📚 Citable RAG Question-Answering System (Mistral 7B Fine-Tuned)

![Python](https://img.shields.io/badge/python-3.9%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-RAG-orange?logo=chainlink)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-lightgrey?logo=sqlite)
![HuggingFace](https://img.shields.io/badge/Model-Mistral%207B-yellow?logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-purple)


## 🌟 Overview
This project implements a high-performance **Retrieval-Augmented Generation (RAG)** system built on a fine-tuned **Mistral 7B Instruct** model.  

The core feature of this system is its ability to provide **concise, factual answers directly from uploaded documents**, always accompanied by a **precise citation format**:

[Chunk ID / Page Number / Timestamp]


The system is deployed as a **FastAPI** service, exposing robust endpoints for **document ingestion** and **real-time question answering**.


## ⚙️ Architecture Highlights

| Component         | Technology                               | Description                                                                 |
|-------------------|-------------------------------------------|-----------------------------------------------------------------------------|
| Generative Model  | Mistral 7B Instruct (Fine-Tuned with QLoRA) | Tuned for strict output compliance (≤3 sentences + mandatory citations).     |
| Vector Database   | FAISS                                     | Efficient retrieval of document chunks.                                     |
| RAG Framework     | LangChain                                 | Orchestrates document processing and query workflow.                        |
| Deployment        | FastAPI                                   | Provides two main API endpoints (`/upload/` and `/QA/`).                    |


## 🗂️ Project Structure

| File Name                  | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| `main.py`                  | FastAPI app and entry point. Handles `/upload/` and `/QA/` endpoints.       |
| `mistral_q_a_finetuned.py` | Fine-tuning scripts for Mistral 7B (custom metrics + data formatting).      |
| `utils.py`                 | Utility functions: chunking, metadata extraction, embeddings.               |
| `requirements.txt`         | Project dependencies (FastAPI, FAISS, LangChain, Torch, Transformers, etc.) |



## 🚀 Setup and Installation

### Prerequisites
- Python 3.9+
- `pip` package installer
- Hugging Face access token (if the model is private)

### Installation
```bash
# Clone repository
git clone [Your Repository URL]
cd [Your Repository Name]

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Environment Variables
export HUGGINGFACE_TOKEN=your_token_here

Running the API

Start the FastAPI app:

uvicorn main:app --reload


API Base: http://127.0.0.1:8000

Interactive Docs: http://127.0.0.1:8000/docs

# API Endpoints
1. Document Upload (Ingestion)

Processes a document → chunks → embeddings → indexes into FAISS.

Method: POST
Endpoint: /upload/

Description: Upload a PDF or TXT file for indexing.

Example (curl)
curl -X 'POST' \
  'http://127.0.0.1:8000/upload/' \
  -H 'accept: application/json' \
  -F 'file=@path/to/your/document.pdf;type=application/pdf'

2. Question Answering

Executes the full RAG pipeline: retrieval + context construction + generation.

Method: POST
Endpoint: /QA/

Request Body (JSON):

{
  "question": "What were the key findings of the report"
}


Example Response:

{
  "answer": "The model successfully refined the system prompt for RAG and was deployed to Hugging Face Spaces. [Chunk ID: 15 / Page Number: 2 / Timestamp: 00:03:45]"
}

