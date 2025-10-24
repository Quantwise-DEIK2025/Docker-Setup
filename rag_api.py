import os
import json
import httpx 
import torch
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any

import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from lancedb.rerankers import ColbertReranker
from transformers import AutoTokenizer

DB_PATH = "./db"
TABLE_NAME = os.environ.get("TABLE_NAME", "my_table")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
EMBEDDING_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
RAG_CHAT_MODEL = os.environ.get("RAG_CHAT_MODEL", "gemma3:4b-it-qat") 

RAG_PROMPT_TEMPLATE = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer from the context, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}
"""

class DbHandler:
    """
    Convenience class to handle database operations for LanceDB.
    This version is simplified for *reading* only.
    """

    def __init__(self, db_path, embedding_model_name):
        self.db = lancedb.connect(db_path)
        self.reranker = ColbertReranker()

        # Check for CUDA availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"DbHandler initializing with device: {device}")

        self.embedding_model = get_registry().get("huggingface").create(
            name=embedding_model_name, 
            trust_remote_code=True, 
            device=device
        )
        
        self.ModelClass = self.create_model_class(self.embedding_model)

    def query_table(self, table_name, prompt, limit=3):
        """
        Queries a specified database table using a prompt and returns the top chunks.
        """
        try:
            table = self.db.open_table(table_name)
            
        except Exception as e:
            print(f"Error opening table {table_name}: {e}")
            print("This likely means the table doesn't exist. Run chunking.py first.")
            return [] 
            
        results_df = table.search(prompt, query_type="hybrid", vector_column_name="vector", fts_columns="text") \
                           .rerank(reranker=self.reranker) \
                           .limit(limit) \
                           .to_pandas()
        
        return results_df["original_text"].tolist()

    @staticmethod
    def create_model_class(embedding_model):
        """
        Factory function used for generating a schema.
        """
        class MyDocument(LanceModel):
            text: str = embedding_model.SourceField()
            vector: Vector(embedding_model.ndims()) = embedding_model.VectorField()
            original_text: str
            context: str
            document: str
            pages: list[int]
            id: str
        return MyDocument

app = FastAPI()

print("Initializing DbHandler...")
db_handler = DbHandler(db_path=DB_PATH, embedding_model_name=EMBEDDING_MODEL_NAME)
print("DbHandler initialized.")

client = httpx.AsyncClient(base_url=OLLAMA_HOST)


@app.post("/api/chat")
async def chat_endpoint(request: Request):
    """
    This endpoint mimics Ollama's /api/chat endpoint.
    It intercepts the request, performs RAG, and then
    streams the response from the *actual* Ollama service.
    """
    try:
        request_data = await request.json()
        print(f"Received request for model: {request_data.get('model')}")

        user_messages = [msg for msg in request_data["messages"] if msg["role"] == "user"]
        if not user_messages:
            raise ValueError("No user message found in the request")
        
        last_user_prompt = user_messages[-1]["content"]
        print(f"Last user prompt: {last_user_prompt}")

        print(f"Querying database for: {last_user_prompt}")
        context_chunks = db_handler.query_table(TABLE_NAME, last_user_prompt, limit=3)
        
        if context_chunks:
            context_str = "\n\n---\n\n".join(context_chunks)
            print(f"Found context: {context_str[:200]}...") 
        else:
            context_str = "No relevant context found."
            print("No context found from database.")

        augmented_prompt = RAG_PROMPT_TEMPLATE.format(context=context_str, question=last_user_prompt)

        ollama_messages = request_data["messages"][:-1]
        ollama_messages.append({"role": "user", "content": augmented_prompt})

        ollama_payload = {
            "model": RAG_CHAT_MODEL, 
            "messages": ollama_messages,
            "stream": True,
            "options": request_data.get("options", {}) 
        }
        
        print(f"Sending augmented prompt to Ollama model: {RAG_CHAT_MODEL}")

        async def stream_ollama_response():
            try:
                async with client.stream(
                    "POST",
                    "/api/chat",
                    json=ollama_payload,
                    timeout=600 
                ) as response:
                    async for chunk in response.aiter_bytes():
                        yield chunk
                        
            except httpx.RequestError as e:
                print(f"ERROR: Could not connect to Ollama at {OLLAMA_HOST}: {e}")
                error_message = {
                    "model": RAG_CHAT_MODEL,
                    "created_at": "now",
                    "message": {
                        "role": "assistant",
                        "content": f"Sorry, I couldn't connect to the language model. Error: {e}"
                    },
                    "done": True
                }
                yield (json.dumps(error_message) + '\n').encode('utf-8')

        return StreamingResponse(stream_ollama_response(), media_type="application/x-ndjson")

    except Exception as e:
        print(f"ERROR: An error occurred in /api/chat endpoint: {e}")
        return {
            "error": f"An internal error occurred: {str(e)}"
        }

@app.api_route("/api/{path:path}", methods=["GET", "POST", "HEAD", "PUT", "DELETE", "PATCH"])
async def proxy_ollama(request: Request, path: str):
    """
    Proxies all other /api/ requests directly to the Ollama service.
    This handles /api/show, /api/pull, etc.
    **SPECIAL HANDLING for /api/tags**: We filter out the chunker model.
    """
    
    if path == "tags" and request.method == "GET":
        print("Proxying request for: /api/tags (with filtering)")
        try:
            resp = await client.get("/api/tags")
            resp.raise_for_status() 
            
            data = resp.json()
            
            if "models" in data:
                original_models = data["models"]
                filtered_models = [
                    model for model in original_models 
                    if "chunker_full_doc" not in model.get("name", "")
                ]
                data["models"] = filtered_models
                print(f"Filtered models. Original: {len(original_models)}, Filtered: {len(filtered_models)}")
            
            return JSONResponse(content=data)
            
        except httpx.RequestError as e:
            print(f"ERROR: Proxy request to Ollama /api/tags failed: {e}")
            return Response(status_code=502, content=f"Failed to proxy request to Ollama: {e}")
        except Exception as e:
            print(f"ERROR: Failed to parse or filter /api/tags response: {e}")
            return Response(status_code=500, content=f"Failed to filter model list: {e}")

    print(f"Proxying request for: /api/{path}")
    
    headers = {name: value for name, value in request.headers.items() if name.lower() not in ("host", "user-agent")}
    
    body = await request.body()
    
    try:
        req = client.build_request(
            method=request.method,
            url=f"/api/{path}",
            headers=headers,
            content=body,
            params=request.query_params
        )
        
        r = await client.send(req, stream=True)

        return StreamingResponse(
            r.aiter_bytes(),
            status_code=r.status_code,
            media_type=r.headers.get("content-type"),
            headers={name: value for name, value in r.headers.items() if name.lower() not in ("content-encoding", "transfer-encoding")}
        )

    except httpx.RequestError as e:
        print(f"ERROR: Proxy request to Ollama failed: {e}")
        return Response(status_code=502, content=f"Failed to proxy request to Ollama: {e}")

@app.get("/")
def health_check():
    """A simple health check endpoint."""

    return {"status": "ok", "message": "RAG API is running"}
