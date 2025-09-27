from utils import (document_loader,split_text,create_chunks,add_metadata,create_embeddings,
create_vector_store,retrieval,chat_completion,chat_model)
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel,Field
import logging
import os

app = FastAPI()

# Creating faiss index and metadata_records to be later accessed by all endpoints 
app.state.faiss_index = None
app.state.metadata_records = []

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request model
class QueryRequest(BaseModel):
    query: str = Field(max_length=1000)

# Response model
class QueryResponse(BaseModel):
    response: str
    status: str = "success"

# Root Endpoint
@app.get("/")
def read_root():
    return {"message": "Hello!"}

# Document Upload Endpoint
@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    docs = document_loader(file_path)
    os.remove(file_path)  # Clean-up file space
    text_splitter = split_text()
    chunks = create_chunks(docs, text_splitter)
    chunks = add_metadata(chunks)
    records, embeddings = create_embeddings(chunks)

    if app.state.faiss_index is None:
        app.state.faiss_index = create_vector_store(embeddings)
        app.state.metadata_records = records
    else:
        app.state.faiss_index.add(embeddings)
        app.state.metadata_records.extend(records)

    return {"message": "Document uploaded and indexed successfully"}

# Question/Answering Endpoint
@app.post("/QA/", response_model=QueryResponse)
def ask_endpoint(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if app.state.faiss_index is None:
      raise HTTPException(status_code=400, detail="No documents uploaded yet")
    try:
        logger.info(f"Processing query of length: {len(request.query)}")

        # Retrieving relevant Context
        context = retrieval(app.state.faiss_index, request.query, app.state.metadata_records)
        # Get AI response
        ai_response = chat_completion(chat_model,request.query,context)
        logger.info("Query processed successfully")
        return QueryResponse(response = ai_response)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")

        raise HTTPException(status_code=500, detail=str(e))
