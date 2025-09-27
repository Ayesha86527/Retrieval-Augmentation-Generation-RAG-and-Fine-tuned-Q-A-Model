from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import time
import faiss
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
import torch
import getpass
import os

# Configure env variables
if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Enter your token: ")


embedding_model=SentenceTransformer('paraphrase-MiniLM-L6-v2')


def load_chat_model():
  model_id = "Ayesha490/mistral-7b-qlora-merged-qa"
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16
    )
  pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=320,
    do_sample=False,
    repetition_penalty=1.03,
    temperature=0
    )
  llm = HuggingFacePipeline(pipeline=pipe)
  chat_model = ChatHuggingFace(llm=llm)
  return chat_model

chat_model=load_chat_model()

def document_loader(file_path):
  loader = UnstructuredFileLoader(file_path)
  docs = loader.load()
  return docs

def split_text():
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)

# Creating Chunks 
def create_chunks(docs, text_splitter):
    all_chunks = []
    for idx, doc in enumerate(docs, start=1):  # idx acts as fallback page number
        chunks = text_splitter.split_documents([doc])
        for chunk in chunks:
            # If original page_number exists, keep it else produce page numbers through idx
            page_number = doc.metadata.get("page_number", idx)

            chunk.metadata["page_number"] = page_number
            chunk.metadata["chunk_id"] = f"chunk_{len(all_chunks)+1}"  # unique ID
            chunk.metadata["timestamp"] = f"00:{idx:02}:{(len(all_chunks)%60):02}"  # dummy timestamp

            all_chunks.append(chunk)
    return all_chunks

# Adding required metadata to the chunks
def add_metadata(chunks):
    enriched = []
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    for i, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["timestamp"] = timestamp
        enriched.append(chunk)
    return enriched

# Creating Embeddings from the processed chunks along with the metadata
def create_embeddings(chunks):
    text_contents = [doc.page_content for doc in chunks]
    embeddings = embedding_model.encode(text_contents)

    records = []
    for i, (embedding, doc) in enumerate(zip(embeddings, chunks), start=1):
        record = {
            "chunk_id": doc.metadata["chunk_id"],
            "page_number": doc.metadata.get("page_number"),
            "timestamp": doc.metadata["timestamp"],
            "text": doc.page_content,
        }
        records.append(record)
    return records,embeddings


# Adding embeddimngs to the FAISS Vector Store
def create_vector_store(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

# Retrieving Top K Chunks from the vector store
def retrieval(index, user_prompt, records):
    query_embedding = embedding_model.encode([user_prompt])
    k=3
    distances, indices = index.search(query_embedding, k)
    retrieved_info = []
    for idx in indices[0]:
        record = records[idx]
        formatted = (
            f"Chunk ID: {record['chunk_id']}\n"
            f"Page Number: {record.get('page_number')}\n"
            f"Timestamp: {record['timestamp']}\n"
            f"Text: {record['text']}\n"
            "-------------------------"
        )
        retrieved_info.append(formatted)
    return "\n".join(retrieved_info)

# Passing Context, User Query and fine-tuned model to Generation Function
def chat_completion(chat_model,user_input,context):
    SYSTEM_PROMPT=f"""
    You are a helpful Q/A assistant.
    Follow these rules STRICTLY:
    - Answer MUST be ≤3 sentences.
    - Answer MUST always include a citation in EXACTLY this format: (Chunk ID/ Page Number/ Timestamp).
    - Output MUST be one single block of text, with no bullet points, no prefixes like “Answer:”, and no extra explanations.
    - Answer MUST be complete, precise, and directly based only on the provided context.
    - DO NOT return incomplete answers.
    - You will ALWAYS follow this output format:
      Answer. (Chunk_ID: Chunk ID: / Page: Page Number/ TimeStamp: Timestamp)
    Here is the context:
    ```{context}```
    """
    messages = [
        SystemMessage(content=f"{SYSTEM_PROMPT}"),
        HumanMessage(
        content= user_input
    ),
    ]
    ai_msg = chat_model.invoke(messages)

    return ai_msg.content.split("[/INST]")[-1].strip()
